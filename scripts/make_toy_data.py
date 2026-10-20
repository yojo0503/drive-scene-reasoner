"""Generate a synthetic compatibility fixture; not DriveLM data or a driving benchmark."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/toy")
    args = p.parse_args()
    out = Path(args.output)
    from PIL import Image, ImageDraw

    dataset = {}
    for n in range(3):
        scene, frame = f"toy_scene_{n}", f"toy_frame_{n}"
        paths = {}
        for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            rel = Path("samples") / cam / f"{scene}_{cam}.png"
            path = out / "nuscenes" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            im = Image.new("RGB", (320, 180), (205, 215, 225))
            d = ImageDraw.Draw(im)
            d.text((10, 15), f"SYNTHETIC {scene} {cam}", fill=(10, 10, 10))
            d.rectangle((110, 80, 220, 135), outline=(30, 80, 170), width=4)
            im.save(path)
            paths[cam] = str(Path("../nuscenes") / rel)
        dataset[scene] = {
            "scene_description": "Synthetic fixture only, no real driving scene.",
            "key_frames": {frame: {
                "image_paths": paths,
                "QA": {
                    "perception": [{"Q": "Which data fixture is this?", "A": "A synthetic test fixture."}],
                    "prediction": [{"Q": "Is future motion inferable from this synthetic image?", "A": "No, insufficient visual evidence."}],
                    "planning": [], "behavior": [],
                },
            }},
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "drivelm_train.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Created synthetic data at {out}. It is not DriveLM data.")


if __name__ == "__main__":
    main()
