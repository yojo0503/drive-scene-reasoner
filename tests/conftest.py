import json
import pytest
from PIL import Image
from drivescene.data import CAMERAS


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "nuscenes"
    source = {}
    for n in range(3):
        paths = {}
        for cam in CAMERAS:
            relative = f"samples/{cam}/{n}.jpg"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 32), (n, 20, 30)).save(path)
            paths[cam] = "../nuscenes/" + relative
        source[f"scene{n}"] = {"scene_description": "ANNOTATION_SECRET", "custom_scene": 7,
            "key_frames": {f"frame{n}": {"image_paths": paths,
                "key_object_infos": {"<c1,CAM_FRONT,12.0,13.0>": {"2d_bbox": [1, 2, 20, 22]}},
                "QA": {"perception": [{"Q": "Is a car visible?", "A": "Yes.", "tag": [0],
                    "C": "group1", "con_up": [], "con_down": [1], "cluster": 2, "layer": 0, "future_field": 9}],
                    "prediction": [{"Q": "Will it move?", "A": "No.", "con_up": [0], "tag": [1]}],
                    "planning": [], "behavior": []}}}}
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    return source_path, root, source
