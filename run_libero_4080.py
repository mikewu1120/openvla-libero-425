"""A local, single-GPU LIBERO-Spatial evaluation using the official OpenVLA model.

Preprocessing and gripper conversion follow openvla/experiments/robot.
This 4-bit Windows smoke run is not the paper's full BF16/A100 benchmark.
"""
import argparse
import importlib
import json
import os
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".libero")
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
sys.path.insert(0, str(ROOT / "LIBERO"))

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image


def configure_libero():
    base = ROOT / "LIBERO" / "libero" / "libero"
    paths = {"benchmark_root": base, "bddl_files": base / "bddl_files",
             "init_states": base / "init_files", "assets": base / "assets",
             "datasets": ROOT / "datasets"}
    paths["datasets"].mkdir(exist_ok=True)
    config = ROOT / ".libero"
    config.mkdir(exist_ok=True)
    (config / "config.yaml").write_text(
        yaml.safe_dump({k: v.as_posix() for k, v in paths.items()}), encoding="utf-8")


def load_model():
    # Import only the official standalone HF classes, avoiding training-only imports.
    package = types.ModuleType("openvla_hf")
    package.__path__ = [str(ROOT / "openvla" / "prismatic" / "extern" / "hf")]
    sys.modules[package.__name__] = package
    model_cls = importlib.import_module("openvla_hf.modeling_prismatic").OpenVLAForActionPrediction
    processing = importlib.import_module("openvla_hf.processing_prismatic")
    from transformers import AutoTokenizer, BitsAndBytesConfig
    checkpoint = ROOT / "models" / "openvla-7b-finetuned-libero-spatial"
    processor = processing.PrismaticProcessor(
        processing.PrismaticImageProcessor.from_pretrained(checkpoint),
        AutoTokenizer.from_pretrained(checkpoint, local_files_only=True))
    model = model_cls.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    return model, processor


def preprocess(obs):
    # Official JPEG round trip + Lanczos3 resize + 90%-area center crop.
    import tensorflow as tf
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    offset = (1 - np.sqrt(0.9)) / 2
    image = tf.image.crop_and_resize(image[None], [[offset, offset, 1-offset, 1-offset]], [0], (224, 224))[0]
    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-only", action="store_true", help="Check scene rendering without loading weights")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adapter", type=Path, help="Optional local LoRA adapter from a training run")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "libero_spatial_task0")
    args = parser.parse_args()
    if not 0 <= args.task_id < 10 or not 1 <= args.episodes <= 50 or args.steps < 1:
        parser.error("task-id must be 0..9; episodes 1..50; steps >= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    configure_libero()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    print(f"Task: {task.language}", flush=True)
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
                             camera_heights=256, camera_widths=256)
    env.seed(0)  # Matches the official evaluator, independently of model seed.
    summary = {"suite": "libero_spatial", "task_id": args.task_id, "task": task.language,
               "seed": args.seed, "sim_only": args.sim_only, "quantization": "NF4",
               "attention": "sdpa", "step_limit": args.steps, "episodes": [],
               "full_paper_benchmark": False, "adapter": str(args.adapter) if args.adapter else None}
    try:
        model = processor = None
        if not args.sim_only:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in this environment")
            summary["gpu"] = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()
            print("Loading official checkpoint in 4-bit...", flush=True)
            model, processor = load_model()
            if args.adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
                model.eval()
            print("Model loaded.", flush=True)
        for episode in range(args.episodes):
            env.reset()
            obs = env.set_init_state(initial_states[episode])
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).save(args.output / f"episode_{episode}_initial.png")
            video = args.output / f"episode_{episode}.mp4"
            actions, latencies = [], []
            success = False
            with imageio.get_writer(str(video), fps=20) as writer:
                for step in range(args.steps):
                    writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    if args.sim_only:
                        action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=float)
                    else:
                        image = preprocess(obs)
                        if step == 0:
                            image.save(args.output / f"episode_{episode}_model_input.png")
                        prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
                        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        with torch.inference_mode():
                            raw = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
                        torch.cuda.synchronize()
                        latencies.append(time.perf_counter() - start)
                        if raw.shape != (7,) or not np.isfinite(raw).all():
                            raise ValueError(f"Invalid predicted action: {raw}")
                        action = raw.copy()
                        action[-1] = -np.sign(2 * action[-1] - 1)
                        actions.append({"step": step, "raw": raw.tolist(), "executed": action.tolist(), "inference_seconds": latencies[-1]})
                    obs, _, done, _ = env.step(action.tolist())
                    if step % 20 == 0:
                        print(f"Episode {episode}, step {step}: {action.tolist()}", flush=True)
                    if done:
                        success = True
                        break
                writer.append_data(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).save(args.output / f"episode_{episode}_final.png")
            (args.output / f"episode_{episode}_actions.json").write_text(json.dumps(actions, indent=2), encoding="utf-8")
            result = {"episode": episode, "steps": step + 1, "success": success,
                      "mean_inference_seconds": float(np.mean(latencies)) if latencies else None,
                      "video": str(video)}
            summary["episodes"].append(result)
            print(json.dumps(result), flush=True)
        if not args.sim_only:
            summary["peak_torch_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            summary["peak_torch_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        env.close()


if __name__ == "__main__":
    main()
