# coding: utf-8
"""
traffic_violation_gui.py
========================
Smart E-Challan — Traffic Violation Detection System
Overspeeding Detection with Tkinter GUI

Run:
    python traffic_violation_gui.py

Dependencies (pip install if missing):
    pip install Pillow opencv-python tensorflow numpy
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import queue
import threading
import datetime
import time
import tarfile
import urllib.request

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

# ── Project path fix ──────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_DIR)
sys.path.insert(0, _DIR)

# ── Project imports ────────────────────────────────────────────────────────────
from utils import label_map_util
from utils.speed_estimator import SpeedEstimator
from utils.violation_logger import (
    init_db, log_violation, fetch_all_violations,
    clear_violations, get_violation_count
)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_NAME   = "ssd_mobilenet_v1_coco_11_06_2017"
MODEL_FILE   = MODEL_NAME + ".tar.gz"
DOWNLOAD_URL = "http://download.tensorflow.org/models/object_detection/" + MODEL_FILE
PATH_TO_CKPT = os.path.join(_DIR, MODEL_NAME, "frozen_inference_graph.pb")
PATH_TO_LABELS = os.path.join(_DIR, "data", "mscoco_label_map.pbtxt")
NUM_CLASSES  = 90

VIOLATIONS_DIR = os.path.join(_DIR, "violations")

# ── Vehicle COCO classes of interest ─────────────────────────────────────────
VEHICLE_CLASSES = {3: "Car", 4: "Motorbike", 6: "Bus", 8: "Truck"}

# ── Dark-theme palette ────────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_CARD     = "#161b22"
C_CARD2    = "#21262d"
C_BORDER   = "#30363d"
C_ACCENT   = "#1f6feb"          # blue
C_GREEN    = "#238636"          # safe / start
C_WARN     = "#d29922"          # yellow
C_RED      = "#da3633"          # violation / danger
C_TEXT     = "#f0f6fc"
C_MUTED    = "#8b949e"
C_CYAN     = "#58a6ff"

FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_STAT   = ("Segoe UI", 22, "bold")
FONT_SMALL  = ("Segoe UI", 9)


# ═════════════════════════════════════════════════════════════════════════════
class TrafficViolationApp:
    """Main application window."""

    # ── Construction ─────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart E-Challan  ·  Traffic Violation Detection")
        self.root.configure(bg=C_BG)
        self.root.state("zoomed")          # maximise on Windows
        self.root.minsize(1100, 650)

        # ── State variables ───────────────────────────────────────────────
        self.video_path       = ""
        self.speed_limit_kmh  = 0.0        # set via dialog at startup
        self.pixels_per_meter = 20.0
        self.video_fps        = 25.0
        self.roi              = None        # (x, y, w, h)

        self.running  = False
        self.paused   = False
        self._stop_evt = threading.Event()

        self.frame_q    = queue.Queue(maxsize=3)
        self.event_q    = queue.Queue()

        # violation de-duplication: vid -> last violation timestamp
        self._vio_cooldown: dict[int, float] = {}
        self._COOLDOWN_S = 6.0             # seconds between consecutive logs per vehicle

        # Tkinter observable vars
        self._var_speed_limit  = tk.StringVar(value="—")
        self._var_violations   = tk.IntVar(value=0)
        self._var_fps          = tk.StringVar(value="—")
        self._var_avg_speed    = tk.StringVar(value="—")
        self._var_status       = tk.StringVar(value="⬤  Idle")
        self._var_video_name   = tk.StringVar(value="No video selected")

        # ── Init persistent storage ───────────────────────────────────────
        init_db()
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_styles()
        self._build_layout()

        # ── Kick off loops ────────────────────────────────────────────────
        self.root.after(200,  self._startup_ask_speed_limit)
        self.root.after(33,   self._gui_tick)        # ~30 fps GUI refresh
        self.root.after(3000, self._log_refresh_tick) # periodic log reload

    # ── Style ─────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",       background=C_BG)
        s.configure("Card.TFrame",  background=C_CARD)
        s.configure("Card2.TFrame", background=C_CARD2)

        s.configure("Treeview",
                    background=C_CARD2, foreground=C_TEXT,
                    fieldbackground=C_CARD2, bordercolor=C_BORDER,
                    rowheight=22, font=FONT_SMALL)
        s.configure("Treeview.Heading",
                    background=C_CARD, foreground=C_TEXT,
                    relief="flat", font=FONT_BOLD)
        s.map("Treeview",
              background=[("selected", C_ACCENT)],
              foreground=[("selected", C_TEXT)])

        s.configure("Vertical.TScrollbar",
                    background=C_CARD2, troughcolor=C_BG,
                    bordercolor=C_BORDER, arrowcolor=C_MUTED)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        # ── Header bar ────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C_CARD, height=52)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="🚦  Smart E-Challan  ·  Traffic Violation Detection",
                 bg=C_CARD, fg=C_TEXT, font=FONT_TITLE).pack(side="left", padx=18, pady=12)

        tk.Label(hdr, textvariable=self._var_status,
                 bg=C_CARD, fg=C_WARN, font=FONT_BOLD).pack(side="right", padx=18)

        tk.Label(hdr, textvariable=self._var_video_name,
                 bg=C_CARD, fg=C_MUTED, font=FONT_SMALL).pack(side="right", padx=6)

        # ── Body (left video  |  right panel) ─────────────────────────────
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)

        # Left — video canvas
        left = tk.Frame(body, bg=C_CARD)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        tk.Label(left, text="📹  Live Detection Feed",
                 bg=C_CARD, fg=C_TEXT, font=FONT_BOLD).pack(anchor="w", padx=10, pady=(8, 2))

        self.canvas = tk.Canvas(left, bg="#000000", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        # Right panel — fixed width
        right = tk.Frame(body, bg=C_BG, width=390)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_stat_cards(right)
        self._build_log_panel(right)
        self._build_snapshot_panel(right)

        # ── Toolbar ───────────────────────────────────────────────────────
        self._build_toolbar()

    # ── Stat cards ────────────────────────────────────────────────────────────
    def _build_stat_cards(self, parent):
        grid = tk.Frame(parent, bg=C_BG)
        grid.pack(fill="x", pady=(0, 6))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        stats = [
            ("Speed Limit",  self._var_speed_limit, "km/h", C_CYAN, 0, 0),
            ("Violations ⚠",  self._var_violations,  "",     C_RED,  0, 1),
            ("Current FPS",  self._var_fps,          "",     C_TEXT, 1, 0),
            ("Avg Speed",    self._var_avg_speed,    "km/h", C_TEXT, 1, 1),
        ]
        for label, var, unit, fg, row, col in stats:
            card = tk.Frame(grid, bg=C_CARD, padx=10, pady=8)
            card.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            tk.Label(card, text=label, bg=C_CARD, fg=C_MUTED, font=FONT_SMALL).pack(anchor="w")
            tk.Label(card, textvariable=var, bg=C_CARD, fg=fg,
                     font=FONT_STAT).pack(anchor="w")
            if unit:
                tk.Label(card, text=unit, bg=C_CARD, fg=C_MUTED,
                         font=FONT_SMALL).pack(anchor="w")

    # ── Violation log panel ────────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        frm = tk.Frame(parent, bg=C_CARD)
        frm.pack(fill="both", expand=True, pady=(0, 6))

        hdr = tk.Frame(frm, bg=C_CARD)
        hdr.pack(fill="x", padx=8, pady=(8, 3))
        tk.Label(hdr, text="📋  Violation Log",
                 bg=C_CARD, fg=C_TEXT, font=FONT_BOLD).pack(side="left")

        clear_btn = tk.Button(hdr, text="Clear All", bg=C_RED, fg="white",
                              font=FONT_SMALL, relief="flat", padx=7, pady=2,
                              cursor="hand2", command=self._clear_log)
        clear_btn.pack(side="right")

        cols = ("ID", "Timestamp", "Vehicle", "Speed", "Limit")
        self.tree = ttk.Treeview(frm, columns=cols, show="headings", height=9)
        for col, w, anc in [
            ("ID",        34,  "center"),
            ("Timestamp", 128, "w"),
            ("Vehicle",   68,  "center"),
            ("Speed",     60,  "center"),
            ("Limit",     54,  "center"),
        ]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor=anc, stretch=False)

        vsb = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self.tree.pack(fill="both", padx=(8, 0), pady=(0, 8))

        self.tree.bind("<<TreeviewSelect>>", self._on_log_row_click)

        # tag for violation rows
        self.tree.tag_configure("vio", foreground=C_RED)

    # ── Snapshot preview panel ────────────────────────────────────────────────
    def _build_snapshot_panel(self, parent):
        frm = tk.Frame(parent, bg=C_CARD)
        frm.pack(fill="x", pady=(0, 0))

        tk.Label(frm, text="📸  Snapshot Preview  (click a log row)",
                 bg=C_CARD, fg=C_TEXT, font=FONT_BOLD).pack(anchor="w", padx=8, pady=(8, 4))

        self.snap_lbl = tk.Label(frm, bg="#0a0a0a",
                                 text="No snapshot selected",
                                 fg=C_MUTED, font=FONT_SMALL,
                                 width=48, height=11)
        self.snap_lbl.pack(padx=8, pady=(0, 8))

    # ── Toolbar ───────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=C_CARD2, height=56)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        def btn(parent, text, bg, cmd, side="left", pad_l=12, state="normal"):
            b = tk.Button(parent, text=text, bg=bg, fg=C_TEXT,
                          activebackground=bg, activeforeground=C_TEXT,
                          font=FONT_BOLD, relief="flat", padx=12, pady=8,
                          cursor="hand2", bd=0, command=cmd, state=state)
            b.pack(side=side, padx=(pad_l, 4), pady=10)
            return b

        btn(bar, "📂  Open Video",       C_ACCENT,  self._open_video)
        btn(bar, "🚀  Set Speed Limit",  C_WARN,    self._ask_speed_limit)
        self._start_btn = btn(bar, "▶  Start Detection", C_GREEN, self._start_detection)
        self._pause_btn = btn(bar, "⏸  Pause",           C_CARD,  self._toggle_pause, state="disabled")
        self._stop_btn  = btn(bar, "⏹  Stop",            C_RED,   self._stop_detection, state="disabled")

        btn(bar, "📁  Violations Folder", C_CARD, self._open_violations_folder, side="right", pad_l=4)
        btn(bar, "🔄  Refresh Log",       C_CARD, self._refresh_log,             side="right", pad_l=4)

    # ── Startup speed-limit prompt ────────────────────────────────────────────
    def _startup_ask_speed_limit(self):
        self._ask_speed_limit(startup=True)

    def _ask_speed_limit(self, startup=False):
        title = ("Welcome — Please Set Speed Limit" if startup
                 else "Speed Limit Configuration")
        prompt = ("Enter the speed limit for this zone (km/h):" if startup
                  else "Update speed limit (km/h):")
        val = simpledialog.askfloat(
            title, prompt,
            initialvalue=self.speed_limit_kmh if self.speed_limit_kmh > 0 else 40.0,
            minvalue=1.0, maxvalue=300.0,
            parent=self.root,
        )
        if val is not None:
            self.speed_limit_kmh = val
            self._var_speed_limit.set(f"{val:.0f}")

    # ── Open video file ────────────────────────────────────────────────────────
    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            self.video_path = path
            self._var_video_name.set(f"📽  {os.path.basename(path)}")
            self._set_status("Video loaded", C_GREEN)
            # Preview first frame
            cap = cv2.VideoCapture(path)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                self._paint_frame(cv2.resize(frame, (1280, 720)))

    # ── ROI selector (Tkinter Toplevel, no cv2 window) ──────────────────────
    def _select_roi(self, first_frame: np.ndarray):
        """
        Show first_frame in a Toplevel; user clicks+drags to draw ROI.
        Returns (x, y, w, h) in original frame coordinates, or None.
        """
        result = [None]

        oh, ow = first_frame.shape[:2]
        MAX_W, MAX_H = 900, 580
        scale = min(MAX_W / ow, MAX_H / oh, 1.0)
        dw, dh = int(ow * scale), int(oh * scale)

        img_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((dw, dh), Image.LANCZOS)

        top = tk.Toplevel(self.root)
        top.title("Select Monitoring Zone  —  Click & Drag")
        top.configure(bg=C_BG)
        top.resizable(False, False)
        top.grab_set()

        tk.Label(
            top,
            text="🖱  Click and drag to define the speed-monitoring zone, then click Confirm.",
            bg=C_BG, fg=C_TEXT, font=FONT_BODY,
        ).pack(pady=8, padx=12)

        cnv = tk.Canvas(top, width=dw, height=dh, bg="black", highlightthickness=1,
                        highlightbackground=C_BORDER, cursor="crosshair")
        cnv.pack(padx=12)

        imgtk = ImageTk.PhotoImage(img_pil)
        cnv.create_image(0, 0, anchor="nw", image=imgtk)
        cnv._img = imgtk

        state = {"start": None, "rect": None}

        def on_press(e):
            state["start"] = (e.x, e.y)
            if state["rect"]:
                cnv.delete(state["rect"])
                state["rect"] = None

        def on_drag(e):
            if state["start"]:
                if state["rect"]:
                    cnv.delete(state["rect"])
                x0, y0 = state["start"]
                state["rect"] = cnv.create_rectangle(
                    x0, y0, e.x, e.y,
                    outline="#00ff88", width=2, dash=(5, 3),
                )

        def on_release(e):
            if state["start"]:
                x0, y0 = state["start"]
                x1, y1 = e.x, e.y
                rx = int(min(x0, x1) / scale)
                ry = int(min(y0, y1) / scale)
                rw = int(abs(x1 - x0) / scale)
                rh = int(abs(y1 - y0) / scale)
                result[0] = (rx, ry, rw, rh)

        cnv.bind("<ButtonPress-1>",   on_press)
        cnv.bind("<B1-Motion>",       on_drag)
        cnv.bind("<ButtonRelease-1>", on_release)

        btn_row = tk.Frame(top, bg=C_BG)
        btn_row.pack(pady=10)

        def confirm():
            if result[0] is None or result[0][2] < 20 or result[0][3] < 20:
                messagebox.showwarning(
                    "Invalid ROI",
                    "Please draw a larger region of interest.",
                    parent=top,
                )
                return
            top.destroy()

        def use_full():
            result[0] = (0, 0, ow, oh)
            top.destroy()

        tk.Button(btn_row, text="✅  Confirm ROI",
                  bg=C_GREEN, fg=C_TEXT, font=FONT_BOLD,
                  relief="flat", padx=14, pady=7, cursor="hand2",
                  command=confirm).pack(side="left", padx=8)

        tk.Button(btn_row, text="⏭  Use Full Frame",
                  bg=C_CARD2, fg=C_TEXT, font=FONT_BODY,
                  relief="flat", padx=14, pady=7, cursor="hand2",
                  command=use_full).pack(side="left", padx=8)

        top.wait_window()
        return result[0]

    # ── Detection controls ────────────────────────────────────────────────────
    def _start_detection(self):
        # Guard: speed limit
        if self.speed_limit_kmh <= 0:
            messagebox.showerror("Speed Limit Required",
                                 "Please set a speed limit before starting.")
            self._ask_speed_limit()
            return

        # Guard: video
        if not self.video_path or not os.path.exists(self.video_path):
            messagebox.showerror("No Video",
                                 "Please open a valid video file first.")
            return

        if self.running:
            return

        # Calibration: lane width
        lane_w = simpledialog.askfloat(
            "Speed Calibration",
            "Approximate lane width in metres visible in the video.\n"
            "(Typical road lane: 3.5 m.  Used to convert pixels → km/h)",
            initialvalue=3.5,
            minvalue=0.5, maxvalue=30.0,
            parent=self.root,
        )
        if lane_w is None:
            lane_w = 3.5

        # Read first frame
        cap = cv2.VideoCapture(self.video_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        ok, first_frame = cap.read()
        cap.release()

        if not ok or first_frame is None:
            messagebox.showerror("Video Error",
                                 "Cannot read from the selected video file.")
            return

        first_frame = cv2.resize(first_frame, (1280, 720))

        # ROI selection
        self._set_status("Draw monitoring zone…", C_WARN)
        roi = self._select_roi(first_frame)
        if roi is None:
            self._set_status("Detection cancelled", C_MUTED)
            return
        self.roi = roi

        # Calibrate pixels_per_meter from ROI width
        roi_px_width = roi[2] if roi[2] > 0 else first_frame.shape[1]
        self.pixels_per_meter = roi_px_width / lane_w

        # Reset state
        self.running = True
        self.paused  = False
        self._stop_evt.clear()
        self._vio_cooldown.clear()
        self._var_violations.set(0)

        self._start_btn.config(state="disabled")
        self._pause_btn.config(state="normal", text="⏸  Pause")
        self._stop_btn.config(state="normal")
        self._set_status("Loading model…", C_WARN)

        t = threading.Thread(target=self._detection_thread, daemon=True)
        t.start()

    def _toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self._pause_btn.config(text="▶  Resume")
            self._set_status("Paused", C_WARN)
        else:
            self._pause_btn.config(text="⏸  Pause")
            self._set_status("Running", C_GREEN)

    def _stop_detection(self):
        self.running = False
        self._stop_evt.set()
        self._set_status("Stopped", C_MUTED)
        self._start_btn.config(state="normal")
        self._pause_btn.config(state="disabled", text="⏸  Pause")
        self._stop_btn.config(state="disabled")
        self._var_fps.set("—")

    # ── Detection thread ──────────────────────────────────────────────────────
    def _detection_thread(self):
        # ── 1. Ensure model exists ─────────────────────────────────────────
        if not os.path.exists(PATH_TO_CKPT):
            self.event_q.put(("status", "Downloading model…", C_WARN))
            try:
                if not os.path.exists(os.path.join(_DIR, MODEL_FILE)):
                    urllib.request.urlretrieve(
                        DOWNLOAD_URL,
                        os.path.join(_DIR, MODEL_FILE),
                    )
                self.event_q.put(("status", "Extracting model…", C_WARN))
                tf_tar = tarfile.open(os.path.join(_DIR, MODEL_FILE))
                for m in tf_tar.getmembers():
                    if "frozen_inference_graph.pb" in os.path.basename(m.name):
                        tf_tar.extract(m, _DIR)
                tf_tar.close()
            except Exception as exc:
                self.event_q.put(("error", f"Model download/extract failed:\n{exc}"))
                self.running = False
                return

        # ── 2. Load TF graph ───────────────────────────────────────────────
        try:
            det_graph = tf.Graph()
            with det_graph.as_default():
                gdef = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, "rb") as f:
                    gdef.ParseFromString(f.read())
                tf.import_graph_def(gdef, name="")
        except Exception as exc:
            self.event_q.put(("error", f"Model load failed:\n{exc}"))
            self.running = False
            return

        # ── 3. Load label map ──────────────────────────────────────────────
        label_map      = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories     = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # ── 4. Initialise speed estimator ──────────────────────────────────
        speed_est = SpeedEstimator(
            pixels_per_meter=self.pixels_per_meter,
            fps=self.video_fps,
            smooth_window=12,
        )

        self.event_q.put(("status", "Running", C_GREEN))

        # ── 5. Open video ──────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.video_path)

        frame_times: list[float] = []
        INFER_EVERY = 2          # run TF inference every N-th frame
        frame_idx   = 0
        last_boxes   = np.zeros((0, 4), dtype=np.float32)
        last_scores  = np.zeros((0,),   dtype=np.float32)
        last_classes = np.zeros((0,),   dtype=np.int32)

        ry, rx, rh, rw = (self.roi[1], self.roi[0],
                          self.roi[3], self.roi[2])  # unpacked for clarity

        with det_graph.as_default():
            with tf.Session(graph=det_graph) as sess:
                t_img   = det_graph.get_tensor_by_name("image_tensor:0")
                t_boxes = det_graph.get_tensor_by_name("detection_boxes:0")
                t_scores= det_graph.get_tensor_by_name("detection_scores:0")
                t_cls   = det_graph.get_tensor_by_name("detection_classes:0")
                t_num   = det_graph.get_tensor_by_name("num_detections:0")

                while self.running and not self._stop_evt.is_set():
                    if self.paused:
                        time.sleep(0.05)
                        continue

                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        self.event_q.put(("status", "✔ Video completed", C_CYAN))
                        self.running = False
                        break

                    frame = cv2.resize(frame, (1280, 720))
                    frame_idx += 1
                    h, w = frame.shape[:2]

                    # TF inference (every INFER_EVERY frames)
                    if frame_idx % INFER_EVERY == 0:
                        exp = np.expand_dims(frame, axis=0)
                        boxes, scores, classes, _ = sess.run(
                            [t_boxes, t_scores, t_cls, t_num],
                            feed_dict={t_img: exp},
                        )
                        last_boxes   = np.squeeze(boxes)
                        last_scores  = np.squeeze(scores)
                        last_classes = np.squeeze(classes).astype(np.int32)

                    # ── Extract vehicle detections inside ROI ──────────────
                    vehicle_cents: list[tuple] = []
                    cent_to_info:  dict[tuple, tuple] = {}

                    n = min(len(last_boxes), 60)
                    for i in range(n):
                        if i >= len(last_scores) or last_scores[i] < 0.40:
                            break
                        cls_id = int(last_classes[i])
                        if cls_id not in VEHICLE_CLASSES:
                            continue

                        ymin, xmin, ymax, xmax = last_boxes[i]
                        L = int(xmin * w);  R = int(xmax * w)
                        T = int(ymin * h);  B = int(ymax * h)
                        cx = (L + R) // 2
                        cy = (T + B) // 2

                        # Inside ROI?
                        if (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh):
                            key = (cx, cy)
                            vehicle_cents.append(key)
                            cent_to_info[key] = (cls_id, L, T, R, B)

                    # ── Speed estimation ───────────────────────────────────
                    speeds = speed_est.update(vehicle_cents)

                    # Build reverse map centroid -> obj_id
                    cent_to_vid = {v: k for k, v in speed_est.tracker.objects.items()}

                    violations_this_frame: list[dict] = []
                    all_speeds: list[float] = list(speeds.values())

                    for cent, info in cent_to_info.items():
                        cls_id, L, T, R, B = info
                        vid = cent_to_vid.get(cent)
                        spd = speeds.get(vid, 0.0) if vid is not None else 0.0
                        is_vio = (spd > self.speed_limit_kmh) and (spd > 3.0)

                        color = (0, 40, 220) if is_vio else (50, 210, 80)
                        cv2.rectangle(frame, (L, T), (R, B), color, 2)

                        cls_name = VEHICLE_CLASSES.get(cls_id, "Vehicle")
                        label    = f"{cls_name}  {spd:.1f} km/h"
                        (tw, th), bl = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                        cv2.rectangle(frame,
                                      (L, T - th - bl - 6), (L + tw + 4, T),
                                      color, -1)
                        cv2.putText(frame, label, (L + 2, T - bl - 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                        # VIOLATION
                        if is_vio and vid is not None:
                            now = time.time()
                            if now - self._vio_cooldown.get(vid, 0.0) > self._COOLDOWN_S:
                                self._vio_cooldown[vid] = now
                                snap = self._save_snapshot(
                                    frame, L, T, R, B, vid, cls_id, spd)
                                ts = log_violation(
                                    vid, cls_name,
                                    round(spd, 1), self.speed_limit_kmh,
                                    snap,
                                )
                                violations_this_frame.append({
                                    "vid": vid, "class": cls_name,
                                    "speed": spd, "snap": snap,
                                    "ts": ts,
                                })

                    # ── Draw overlays ──────────────────────────────────────
                    # ROI rectangle
                    cv2.rectangle(frame,
                                  (rx, ry), (rx + rw, ry + rh),
                                  (0, 200, 255), 2)
                    cv2.putText(frame, "Monitoring Zone",
                                (rx + 4, ry + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 200, 255), 1, cv2.LINE_AA)

                    # Timestamp
                    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, ts_str, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (255, 255, 255), 2, cv2.LINE_AA)

                    # Speed-limit watermark
                    cv2.putText(frame,
                                f"Speed Limit: {self.speed_limit_kmh:.0f} km/h",
                                (10, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 200, 255), 2, cv2.LINE_AA)

                    # VIOLATION FLASH banner
                    if violations_this_frame:
                        cv2.rectangle(frame, (0, 0), (w, 5), (0, 0, 255), -1)
                        cv2.rectangle(frame, (0, h - 5), (w, h), (0, 0, 255), -1)
                        cv2.putText(frame, "⚠ VIOLATION DETECTED",
                                    (w // 2 - 220, h - 18),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.9,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    # FPS counter
                    elapsed = time.time() - t0
                    frame_times.append(elapsed)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    fps = 1.0 / (sum(frame_times) / len(frame_times) + 1e-9)
                    cv2.putText(frame,
                                f"FPS: {fps:.1f}",
                                (w - 120, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (255, 220, 0), 2, cv2.LINE_AA)

                    # Push to GUI queue (non-blocking, drop if full)
                    try:
                        self.frame_q.put_nowait((
                            frame.copy(), fps,
                            [s for s in all_speeds if s > 0],
                            violations_this_frame,
                        ))
                    except queue.Full:
                        pass

        cap.release()
        if not self._stop_evt.is_set():
            self.event_q.put(("done",))

    # ── Snapshot capture ──────────────────────────────────────────────────────
    def _save_snapshot(self, frame, L, T, R, B, vid, cls_id, speed):
        ts_str    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename  = f"violation_v{vid}_{ts_str}.jpg"
        out_path  = os.path.join(VIOLATIONS_DIR, filename)

        snap = frame.copy()

        # Thick red border around the vehicle
        pad = 8
        cv2.rectangle(snap,
                      (max(L - pad, 0), max(T - pad, 0)),
                      (min(R + pad, snap.shape[1] - 1), min(B + pad, snap.shape[0] - 1)),
                      (0, 0, 255), 4)

        # Info overlay
        cls_name = VEHICLE_CLASSES.get(cls_id, "Vehicle")
        lines = [
            "OVER-SPEED VIOLATION",
            f"{cls_name}  |  {speed:.1f} km/h",
            f"Limit: {self.speed_limit_kmh:.0f} km/h",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        y0 = max(T - 10 - len(lines) * 26, 10)
        for line in lines:
            (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y = y0 + th
            cv2.rectangle(snap, (L, y0 - 2), (L + tw + 6, y + bl + 2), (0, 0, 180), -1)
            cv2.putText(snap, line, (L + 3, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += th + 8

        cv2.imwrite(out_path, snap)
        return out_path

    # ── GUI tick (runs every ~33 ms on main thread) ───────────────────────────
    def _gui_tick(self):
        # Process events from detection thread
        while not self.event_q.empty():
            try:
                evt = self.event_q.get_nowait()
                if evt[0] == "status":
                    _, msg, color = evt
                    self._set_status(msg, color)
                elif evt[0] == "done":
                    self.running = False
                    self._start_btn.config(state="normal")
                    self._pause_btn.config(state="disabled", text="⏸  Pause")
                    self._stop_btn.config(state="disabled")
                    self._refresh_log()
                elif evt[0] == "error":
                    messagebox.showerror("Detection Error", evt[1])
                    self.running = False
                    self._start_btn.config(state="normal")
                    self._pause_btn.config(state="disabled")
                    self._stop_btn.config(state="disabled")
            except queue.Empty:
                break

        # Process frames
        latest_pkg = None
        while not self.frame_q.empty():
            try:
                latest_pkg = self.frame_q.get_nowait()
            except queue.Full:
                break

        if latest_pkg is not None:
            frame, fps, active_speeds, violations = latest_pkg

            self._var_fps.set(f"{fps:.1f}")
            if active_speeds:
                self._var_avg_speed.set(f"{np.mean(active_speeds):.1f}")
            else:
                self._var_avg_speed.set("—")

            if violations:
                cur = self._var_violations.get()
                self._var_violations.set(cur + len(violations))
                self._refresh_log()

            self._paint_frame(frame)

        self.root.after(33, self._gui_tick)

    # ── Periodic log refresh ──────────────────────────────────────────────────
    def _log_refresh_tick(self):
        self._refresh_log()
        self.root.after(3000, self._log_refresh_tick)

    def _refresh_log(self):
        rows = fetch_all_violations()
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            vid_id, ts, cls, spd, lim, snap = row
            self.tree.insert(
                "", "end",
                iid=str(vid_id),
                values=(vid_id, ts, cls, f"{spd:.1f}", f"{lim:.0f}"),
                tags=("vio",),
            )

    def _clear_log(self):
        if messagebox.askyesno(
            "Clear Violations",
            "Delete all violation records from the database?\nSnapshot files will NOT be deleted.",
            parent=self.root,
        ):
            clear_violations()
            self._var_violations.set(0)
            self.tree.delete(*self.tree.get_children())
            self.snap_lbl.config(image="", text="No snapshot selected", fg=C_MUTED)

    # ── Log row click → snapshot preview ─────────────────────────────────────
    def _on_log_row_click(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        for row in fetch_all_violations():
            if str(row[0]) == str(iid):
                self._show_snapshot(row[5])
                return

    def _show_snapshot(self, path: str):
        if not path or not os.path.exists(path):
            self.snap_lbl.config(image="", text="Snapshot not found", fg=C_MUTED)
            return
        try:
            img = Image.open(path)
            img.thumbnail((370, 210), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            self.snap_lbl.config(image=imgtk, text="")
            self.snap_lbl._imgtk = imgtk
        except Exception:
            self.snap_lbl.config(image="", text="Cannot load snapshot", fg=C_RED)

    # ── Canvas frame painter ──────────────────────────────────────────────────
    def _paint_frame(self, bgr_frame: np.ndarray):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = bgr_frame.shape[:2]
        scale = min(cw / w, ch / h)
        dw, dh = int(w * scale), int(h * scale)

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=imgtk)
        self.canvas._imgtk = imgtk

    # ── Status helper ─────────────────────────────────────────────────────────
    def _set_status(self, msg: str, color: str = C_WARN):
        self.root.after(0, lambda: (
            self._var_status.set(f"⬤  {msg}"),
        ))
        # also change the label colour
        for w in self.root.winfo_children():
            for child in getattr(w, "winfo_children", lambda: [])():
                if getattr(child, "cget", None) and hasattr(child, "_name"):
                    pass  # traverse not needed — we do it via a tag approach

        # Simpler: find the status label in the header by textvariable
        # We stored it in __init__ as convenience
        try:
            self._status_lbl.config(fg=color)
        except AttributeError:
            pass

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _open_violations_folder(self):
        import subprocess
        os.makedirs(VIOLATIONS_DIR, exist_ok=True)
        subprocess.Popen(f'explorer "{VIOLATIONS_DIR}"')

    def on_close(self):
        self._stop_detection()
        self.root.destroy()


# ═════════════════════════════════════════════════════════════════════════════
def main():
    os.chdir(_DIR)
    root = tk.Tk()
    app  = TrafficViolationApp(root)

    # Give the status label a direct reference so _set_status can recolour it
    for child in list(root.winfo_children()[0].winfo_children()):
        if isinstance(child, tk.Label) and child.cget("textvariable"):
            try:
                if str(child.cget("textvariable")) == str(app._var_status):
                    app._status_lbl = child
            except Exception:
                pass

    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
