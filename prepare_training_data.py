"""Extract one task from two SHA256-verified official OpenVLA RLDS shards."""
import hashlib
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "datasets/training_subset"
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def main():
    source = json.loads((DEST / "raw/manifest.json").read_text())
    episodes = []
    for entry in source["files"]:
        if ".tfrecord-" not in entry["path"]:
            continue
        path = DEST / "raw" / Path(entry["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["lfs"]["oid"]
        for record_id, record in enumerate(tf.compat.v1.io.tf_record_iterator(str(path))):
            features = tf.train.Example.FromString(record).features.feature
            instructions = features["steps/language_instruction"].bytes_list.value
            if instructions[0].decode() != TASK:
                continue
            assert all(text.decode() == TASK for text in instructions)
            actions = np.asarray(features["steps/action"].float_list.value, dtype=np.float32).reshape(-1, 7)
            images = []
            for encoded in features["steps/observation/image"].bytes_list.value:
                # RLDS images are already rotated upright by the official builder.
                image = tf.io.decode_jpeg(encoded, channels=3)
                image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
                image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
                image = tf.image.convert_image_dtype(image, tf.float32)
                offset = (1 - np.sqrt(0.9)) / 2
                image = tf.image.crop_and_resize(image[None], [[offset, offset, 1-offset, 1-offset]], [0], (224, 224))[0]
                images.append(tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True).numpy())
            assert len(images) == len(actions) and np.isfinite(actions).all()
            episodes.append({"images": np.stack(images), "actions": actions,
                             "id": f"{path.name}:{record_id}", "sha256": hashlib.sha256(record).hexdigest()})
    assert len(episodes) >= 4, "Need at least 3 training trajectories and one validation trajectory"
    assert len({e["sha256"] for e in episodes}) == len(episodes)
    manifest = {"task": TASK, "source": source, "preprocessing": "upright RLDS JPEG -> Lanczos3 224 -> center crop area 0.9",
                "validation_scope": "Held out only from this local update; base checkpoint was already trained on LIBERO-Spatial.",
                "episodes": []}
    for index, episode in enumerate(episodes):
        split = "validation" if index >= len(episodes) - 2 else "train"
        name = f"{split}_episode_{index:02}.npz"
        np.savez_compressed(DEST / name, images=episode["images"], actions=episode["actions"])
        manifest["episodes"].append({"file": name, "split": split, "steps": len(episode["actions"]),
                                     "source_record": episode["id"], "record_sha256": episode["sha256"]})
    Image.fromarray(episodes[0]["images"][0]).save(DEST / "training_sample.png")
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"train_episodes": len(episodes)-2, "validation_episodes": 2,
                      "total_frames": sum(len(e["actions"]) for e in episodes)}, indent=2))


if __name__ == "__main__":
    main()
