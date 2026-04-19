"""Label map utility functions for the TF Object Detection API (compat version)."""

import logging

# Minimal proto-like container so we don't need protobuf installed
class _Item:
    def __init__(self, id_, name, display_name=None):
        self.id = id_
        self.name = name
        self.display_name = display_name or name


class _LabelMap:
    def __init__(self):
        self.item = []


def load_labelmap(path):
    """Loads a label map (.pbtxt) file and returns a _LabelMap object."""
    label_map = _LabelMap()
    try:
        with open(path, 'r') as f:
            content = f.read()
        # Simple parser for pbtxt label map format
        import re
        items = re.findall(r'item\s*\{([^}]+)\}', content, re.DOTALL)
        for item_str in items:
            id_match = re.search(r'id\s*:\s*(\d+)', item_str)
            name_match = re.search(r'display_name\s*:\s*"([^"]+)"', item_str)
            if not name_match:
                name_match = re.search(r'name\s*:\s*"([^"]+)"', item_str)
            if id_match and name_match:
                label_map.item.append(_Item(
                    id_=int(id_match.group(1)),
                    name=name_match.group(1),
                    display_name=name_match.group(1)
                ))
    except Exception as e:
        logging.warning(f"Could not load label map from {path}: {e}")
    return label_map


def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    """Converts a label map to a list of category dicts."""
    categories = []
    for item in label_map.item:
        if item.id < 1 or item.id > max_num_classes:
            continue
        name = item.display_name if use_display_name else item.name
        categories.append({'id': item.id, 'name': name})
    return categories


def create_category_index(categories):
    """Creates a dict mapping category id to category dict."""
    return {cat['id']: cat for cat in categories}
