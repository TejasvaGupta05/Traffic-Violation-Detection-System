"""
Helmet Detector — Roboflow cloud-based detection for motorcycle violations.

Detects:
  • No helmet
  • Wrong lane (rear-facing motorcycle)
  • Triple riding (more than 2 people)

Also extracts license plate numbers via OCR.space API.

Usage:
    def on_status(msg):
        print(msg)

    detector = HelmetDetector(roboflow_api_key, ocr_api_key, status_callback=on_status)
    results  = detector.detect(bgr_frame)
"""

import os
import re
import json
import logging
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  OCR helper
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_license_plate(image_path: str, api_key: str, status_cb=None) -> str:
    """
    Send a cropped license-plate image to OCR.space and return the
    cleaned plate number string.  Returns '' on failure.
    """
    if not api_key:
        return ""
    try:
        if status_cb:
            status_cb("📤 Sending license plate to OCR.space…")
        payload = {
            "isOverlayRequired": True,
            "apikey": api_key,
            "language": "eng",
            "OCREngine": 2,
        }
        with open(image_path, "rb") as f:
            resp = requests.post(
                "https://api.ocr.space/parse/image",
                files={image_path: f},
                data=payload,
                timeout=15,
            )
        data = json.loads(resp.content.decode())

        lines = data["ParsedResults"][0]["TextOverlay"]["Lines"]
        raw = "".join(line["LineText"] for line in lines)
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", raw)
        if status_cb:
            if cleaned:
                status_cb(f"✅ OCR result: {cleaned}")
            else:
                status_cb("⚠ OCR returned empty result")
        return cleaned
    except Exception as exc:
        logger.debug("OCR failed for %s: %s", image_path, exc)
        if status_cb:
            status_cb(f"❌ OCR failed: {exc}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
#  Drawing helper
# ──────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = {
    "helmet":         "blue",
    "motorcyclist":   "green",
    "license_plate":  "red",
    "face":           "darkmagenta",
    "front":          "darkgoldenrod",
    "rear":           "darkorchid",
}


def _draw_detections(predictions: list[dict], img: Image.Image) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image (returns a copy)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        cls = pred["class"]
        conf = pred["confidence"]
        color = _CLASS_COLORS.get(cls, "black")

        if cls == "motorcyclist":
            draw.rectangle([x1, y1, x2, y1 + 14], fill=color)
            lbl_pos = (x1 + 5, y1 + 2)
        else:
            draw.rectangle([x1, y1 - 14, x2, y1], fill=color)
            lbl_pos = (x1 + 5, y1 - 12)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text(lbl_pos, f"{cls} ({conf:.2f})", fill="white")

    return img


# ──────────────────────────────────────────────────────────────────────────────
#  HelmetDetector
# ──────────────────────────────────────────────────────────────────────────────

class HelmetDetector:
    """
    Wraps three Roboflow cloud models to detect motorcycle-related
    traffic violations in a single frame.

    Thread-safe: each call to detect() is stateless and self-contained.
    """

    def __init__(self, roboflow_api_key: str, ocr_api_key: str = "",
                 status_callback=None):
        """
        Initialize and download model metadata from Roboflow.

        Parameters
        ----------
        roboflow_api_key : str
            Roboflow API key.
        ocr_api_key : str
            OCR.space API key (optional — license plate OCR is skipped if empty).
        status_callback : callable, optional
            Called with (message: str) at each stage for live progress reporting.
        """
        self._ocr_key = ocr_api_key
        self._status_cb = status_callback or (lambda msg: None)
        self._models_ready = False
        self._init_error = None

        try:
            self._status_cb("🔄 Importing Roboflow SDK…")
            from roboflow import Roboflow

            self._status_cb("🔑 Authenticating with Roboflow API…")
            rf = Roboflow(api_key=roboflow_api_key)

            # Model 1 — helmet + motorcyclist + license plate detection
            self._status_cb("📦 Loading Model 1/3: Helmet Detection…")
            self._m_helmet = (
                rf.workspace()
                .project("helmet-detection-project")
                .version(13)
                .model
            )
            self._status_cb("✅ Model 1/3 loaded: Helmet Detection")

            # Model 2 — face detection
            self._status_cb("📦 Loading Model 2/3: Face Detection…")
            self._m_face = (
                rf.workspace()
                .project("face-detection-mik1i")
                .version(21)
                .model
            )
            self._status_cb("✅ Model 2/3 loaded: Face Detection")

            # Model 3 — two-wheeler lane (front/rear) detection
            self._status_cb("📦 Loading Model 3/3: Lane Detection…")
            self._m_lane = (
                rf.workspace()
                .project("two-wheeler-lane-detection")
                .version(3)
                .model
            )
            self._status_cb("✅ Model 3/3 loaded: Lane Detection")

            self._models_ready = True
            self._status_cb("🟢 All 3 Roboflow models ready!")
            logger.info("HelmetDetector: all 3 Roboflow models loaded successfully")

        except Exception as exc:
            self._init_error = str(exc)
            self._status_cb(f"❌ Model init failed: {exc}")
            logger.error("HelmetDetector init failed: %s", exc)

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._models_ready

    @property
    def init_error(self) -> str | None:
        return self._init_error

    # ------------------------------------------------------------------
    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Analyze a single BGR frame for helmet / lane / triple-riding violations.

        Returns
        -------
        list of dicts, each with keys:
            violation_type : str   — 'no_helmet', 'wrong_lane', 'triple_riding'
            bbox           : tuple — (x1, y1, x2, y2) in original frame coords
            license_plate  : str   — OCR result (may be '')
            confidence     : float — motorcyclist detection confidence
            snapshot_pil   : PIL.Image — annotated crop of the motorcyclist
        """
        if not self._models_ready:
            return []

        self._status_cb("📸 Preparing frame for analysis…")

        # Convert to PIL for saving as temp file (Roboflow needs a file path)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)

        # Save temp frame
        tmp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "violations",
        )
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_frame = os.path.join(tmp_dir, "_helmet_tmp_frame.jpg")
        pil_frame.save(tmp_frame, quality=85)

        try:
            return self._process_frame(pil_frame, tmp_frame)
        except Exception as exc:
            self._status_cb(f"❌ Detection error: {exc}")
            logger.error("HelmetDetector.detect error: %s", exc)
            return []
        finally:
            # Clean up temp files
            for f in [tmp_frame,
                      os.path.join(tmp_dir, "_helmet_tmp_motor.jpg"),
                      os.path.join(tmp_dir, "_helmet_tmp_lp.jpg")]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    def _process_frame(self, pil_frame: Image.Image, image_path: str) -> list[dict]:
        """Core detection logic (adapted from traffic_new/main.py)."""
        violations: list[dict] = []

        # ── Step 1: detect helmets, motorcyclists, license plates ─────────
        self._status_cb("📤 Sending frame to Roboflow (helmet model)…")
        r1 = self._m_helmet.predict(image_path, confidence=40, overlap=40)
        pred1 = r1.json()["predictions"]

        num_motorcyclists = sum(1 for p in pred1 if p["class"] == "motorcyclist")
        num_helmets_total = sum(1 for p in pred1 if p["class"] == "helmet")
        num_plates = sum(1 for p in pred1 if p["class"] == "license_plate")
        self._status_cb(
            f"🔍 Found: {num_motorcyclists} motorcyclist(s), "
            f"{num_helmets_total} helmet(s), {num_plates} plate(s)"
        )

        if num_motorcyclists == 0:
            self._status_cb("ℹ No motorcyclists detected — skipping frame")
            return []

        motor_idx = 0
        for pr1 in pred1:
            if pr1["class"] != "motorcyclist":
                continue
            motor_idx += 1

            # ── Motorcyclist bounding box ─────────────────────────────────
            mx, my, mw, mh = pr1["x"], pr1["y"], pr1["width"], pr1["height"]
            mx1, my1 = int(mx - mw / 2), int(my - mh / 2)
            mx2, my2 = int(mx + mw / 2), int(my + mh / 2)

            self._status_cb(
                f"🏍 Analyzing motorcyclist {motor_idx}/{num_motorcyclists} "
                f"({pr1['confidence']:.0%} conf)…"
            )

            motor_crop = pil_frame.crop((mx1, my1, mx2, my2))
            tmp_dir = os.path.dirname(image_path)
            tmp_motor = os.path.join(tmp_dir, "_helmet_tmp_motor.jpg")
            motor_crop.save(tmp_motor)

            # ── Flags ─────────────────────────────────────────────────────
            helmet_detected = False
            face_detected = False
            rear_detected = False
            more_than_two = False
            num_faces = 0
            num_helmets = 0

            # ── Step 2: lane check (front/rear) ──────────────────────────
            self._status_cb("📤 Checking lane direction (lane model)…")
            r3 = self._m_lane.predict(tmp_motor, confidence=10, overlap=10)
            lane_json = r3.json()

            # Keep only highest-confidence lane prediction
            if lane_json["predictions"]:
                best = max(lane_json["predictions"], key=lambda p: p["confidence"])
                lane_json["predictions"] = [best]
                self._status_cb(f"   Lane: {best['class']} ({best['confidence']:.0%})")
            else:
                self._status_cb("   Lane: no direction detected")

            for lp in lane_json["predictions"]:
                if lp["class"] == "rear":
                    rx, ry = lp["x"], lp["y"]
                    if mx1 < rx < mx2 and my1 < ry < my2:
                        rear_detected = True
                        break

            # ── Step 3: face detection ────────────────────────────────────
            self._status_cb("📤 Scanning for faces (face model)…")
            r2 = self._m_face.predict(tmp_motor, confidence=40, overlap=30)
            pred2 = r2.json()["predictions"]

            for fp in pred2:
                if fp["class"] != "face":
                    continue
                fx, fy = fp["x"], fp["y"]
                fw, fh_val = fp["width"], fp["height"]

                if mx1 < fx < mx2 and my1 < fy < my2:
                    num_faces += 1

                    # Check overlap with helmet boxes to avoid double-counting
                    for hp in pred1:
                        if hp["class"] != "helmet":
                            continue
                        hx, hy = hp["x"], hp["y"]
                        hw, hh = hp["width"], hp["height"]

                        ow1 = max(fx, hx)
                        oh1 = max(fy, hy)
                        ow2 = min(fx + fw, hx + hw)
                        oh2 = min(fy + fh_val, hy + hh)
                        overlap_area = max(0, ow2 - ow1) * max(0, oh2 - oh1)
                        face_area = fw * fh_val

                        if face_area > 0 and overlap_area / face_area > 0.6:
                            num_faces -= 1
                            break

            if num_faces > 0:
                face_detected = True
            self._status_cb(f"   Faces detected: {num_faces}")

            # ── Step 4: helmet check ──────────────────────────────────────
            for hp in pred1:
                if hp["class"] == "helmet":
                    hx, hy = hp["x"], hp["y"]
                    if mx1 < hx < mx2 and my1 < hy < my2:
                        helmet_detected = True
                        num_helmets += 1

            self._status_cb(
                f"   Helmets on rider: {num_helmets}  |  "
                f"Helmet: {'YES' if helmet_detected else 'NO'}  |  "
                f"Rear: {'YES' if rear_detected else 'NO'}"
            )

            # ── Step 5: triple riding ─────────────────────────────────────
            if num_faces + num_helmets > 2:
                more_than_two = True
                self._status_cb(f"   ⚠ Triple riding detected! ({num_faces + num_helmets} people)")

            # ── Step 6: build annotated image ─────────────────────────────
            r4 = self._m_helmet.predict(tmp_motor, confidence=60, overlap=40)
            all_preds = (
                r4.json()["predictions"]
                + r2.json()["predictions"]
                + lane_json["predictions"]
            )
            annotated = _draw_detections(all_preds, motor_crop)

            # ── Step 7: check for violations ──────────────────────────────
            is_violation = (
                not helmet_detected or face_detected or rear_detected or more_than_two
            )

            if not is_violation:
                self._status_cb(f"   ✅ No violations for motorcyclist {motor_idx}")
                continue

            # Determine violation types
            violation_types: list[str] = []
            if not helmet_detected or face_detected:
                violation_types.append("no_helmet")
            if rear_detected:
                violation_types.append("wrong_lane")
            if more_than_two:
                violation_types.append("triple_riding")

            self._status_cb(
                f"   🚨 VIOLATION(S): {', '.join(violation_types)}"
            )

            # ── Step 8: license plate OCR ─────────────────────────────────
            license_plate = ""
            for lp_pred in pred1:
                if lp_pred["class"] != "license_plate":
                    continue
                lpx, lpy = lp_pred["x"], lp_pred["y"]
                lpw, lph = lp_pred["width"], lp_pred["height"]

                if mx1 < lpx < mx2 and my1 < lpy < my2:
                    lp_x1 = int(lpx - lpw / 2)
                    lp_y1 = int(lpy - lph / 2)
                    lp_x2 = int(lpx + lpw / 2)
                    lp_y2 = int(lpy + lph / 2)

                    lp_crop = pil_frame.crop((lp_x1, lp_y1, lp_x2, lp_y2))
                    tmp_lp = os.path.join(os.path.dirname(image_path), "_helmet_tmp_lp.jpg")
                    lp_crop.save(tmp_lp)
                    license_plate = _ocr_license_plate(tmp_lp, self._ocr_key, self._status_cb)
                    break

            # ── Emit one violation entry per type ─────────────────────────
            for vtype in violation_types:
                violations.append({
                    "violation_type": vtype,
                    "bbox": (mx1, my1, mx2, my2),
                    "license_plate": license_plate,
                    "confidence": pr1["confidence"],
                    "snapshot_pil": annotated,
                })

        if violations:
            self._status_cb(f"🚨 Frame analysis complete — {len(violations)} violation(s) found")
        else:
            self._status_cb("✅ Frame analysis complete — no violations")

        return violations
