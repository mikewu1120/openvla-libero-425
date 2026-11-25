"""Small, reproducible QLoRA continuation of the existing LIBERO-Spatial model."""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

from run_libero_4080 import ROOT, load_model
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Demonstrations:
    def __init__(self, processor, stats, folder=ROOT / "datasets/training_subset"):
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.processor = processor
        action_cls = import_file("official_action_tokenizer", ROOT / "openvla/prismatic/vla/action_tokenizer.py").ActionTokenizer
        prompt_cls = import_file("official_prompt_builder", ROOT / "openvla/prismatic/models/backbones/llm/prompting/base_prompter.py").PurePromptBuilder
        action_tokenizer = action_cls(processor.tokenizer)
        self.samples = {"train": [], "validation": []}
        self.sample_tasks = {"train": [], "validation": []}
        low, high = np.array(stats["q01"]), np.array(stats["q99"])
        mask = np.array(stats.get("mask", [True] * 7))
        for episode in self.manifest["episodes"]:
            with np.load(folder / episode["file"]) as data:
                images, actions = data["images"], data["actions"].copy()
            # Exact official LIBERO gripper standardization, then existing checkpoint q01/q99.
            actions[:, -1] = 1 - np.clip(actions[:, -1], 0, 1)
            actions = np.where(mask, np.clip(2 * (actions - low) / (high - low + 1e-8) - 1, -1, 1), actions)
            for image, action in zip(images, actions):
                prompt = prompt_cls("openvla")
                instruction = episode.get("instruction", self.manifest["task"])
                prompt.add_turn("human", f"What action should the robot take to {instruction}?")
                prompt.add_turn("gpt", action_tokenizer(action))
                ids = processor.tokenizer(prompt.get_prompt(), add_special_tokens=True).input_ids
                expected = processor.tokenizer.vocab_size - np.digitize(np.clip(action, -1, 1), action_tokenizer.bins)
                assert ids[-1] == processor.tokenizer.eos_token_id
                assert ids[-8:-1] == expected.tolist(), "Action token round-trip failed"
                self.samples[episode["split"]].append((image, torch.tensor(ids)))
                self.sample_tasks[episode["split"]].append(episode.get("task_id", 0))

    def batch(self, split, index):
        image, ids = self.samples[split][index]
        labels = ids.clone()
        labels[:-8] = -100  # Seven action tokens and EOS; no prompt supervision.
        pixels = self.processor.image_processor.apply_transform(Image.fromarray(image))
        return {"input_ids": ids[None].to("cuda"), "labels": labels[None].to("cuda"),
                "attention_mask": torch.ones_like(ids)[None].to("cuda"),
                "pixel_values": pixels[None].to("cuda", dtype=torch.bfloat16)}


def score(model, dataset, split, count):
    model.eval()
    indices = np.linspace(0, len(dataset.samples[split])-1, min(count, len(dataset.samples[split])), dtype=int)
    losses, correct, total = [], 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in indices:
            batch = dataset.batch(split, index)
            output = model(**batch, use_cache=False)
            losses.append(output.loss.item())
            # Visual tokens are inserted after BOS. Align causal logits with text targets.
            patches = output.logits.shape[1] - batch["input_ids"].shape[1]
            predictions = output.logits[:, patches:-1].argmax(-1)
            labels = batch["labels"][:, 1:]
            mask = labels > dataset.processor.tokenizer.vocab_size - 257
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    return {"loss": float(np.mean(losses)), "action_token_accuracy": correct / total, "frames": len(indices)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Optimizer updates")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--eval-frames", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "training_runs/qlora_task0")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/training_subset")
    args = parser.parse_args()
    if args.steps < 1 or args.accumulation < 1 or args.eval_frames < 1:
        parser.error("steps, accumulation and eval-frames must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output to preserve prior runs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    free, _ = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError(f"Only {free / 1024**3:.2f} GiB free; close games/other GPU jobs before training")
    args.output.mkdir(parents=True)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    start = time.perf_counter()
    base, processor = load_model()
    dataset = Demonstrations(processor, base.norm_stats["libero_spatial"]["action"], args.data)
    base = prepare_model_for_kbit_training(base, gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    base.language_model.config.use_cache = False
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                          target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    report = {"status": "running", "base": str(ROOT / "models/openvla-7b-finetuned-libero-spatial"),
              "task": dataset.manifest["task"], "seed": 7, "rank": 8, "lora_alpha": 16,
              "target_modules": ["q_proj", "v_proj"], "learning_rate": 1e-4,
              "quantization": "NF4", "attention": "sdpa", "gradient_checkpointing": True,
              "batch_size": 1, "accumulation": args.accumulation, "planned_updates": args.steps,
              "trainable_parameters": sum(p.numel() for p in parameters),
              "dataset_frames": {k: len(v) for k, v in dataset.samples.items()},
              "data_directory": str(args.data),
              "sampling": "round-robin tasks, randomized frames within task",
              "validation_scope": dataset.manifest["validation_scope"]}
    torch.cuda.reset_peak_memory_stats()
    try:
        report["before"] = {split: score(model, dataset, split, args.eval_frames) for split in dataset.samples}
        print("BEFORE", json.dumps(report["before"]), flush=True)
        task_ids = sorted(set(dataset.sample_tasks["train"]))
        pools = {task: np.flatnonzero(np.array(dataset.sample_tasks["train"]) == task) for task in task_ids}
        orders = {task: [] for task in task_ids}
        samples_seen = {task: 0 for task in task_ids}
        microstep = 0
        with (args.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, args.steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for _ in range(args.accumulation):
                    task = task_ids[microstep % len(task_ids)]
                    if not orders[task]:
                        orders[task] = np.random.permutation(pools[task]).tolist()
                    batch = dataset.batch("train", orders[task].pop())
                    samples_seen[task] += 1
                    microstep += 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = model(**batch, use_cache=False)
                        loss = output.loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    losses.append(loss.item())
                    (loss / args.accumulation).backward()
                    del output, loss, batch
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                if norm.item() == 0:
                    raise RuntimeError("Zero adapter gradients")
                optimizer.step()
                row = {"update": step, "loss": float(np.mean(losses)), "gradient_norm": norm.item(),
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                       "elapsed_seconds": time.perf_counter() - start}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                report["completed_updates"] = step
                report["sampled_frames_per_task"] = samples_seen.copy()
        optimizer.zero_grad(set_to_none=True)
        report["after"] = {split: score(model, dataset, split, args.eval_frames) for split in dataset.samples}
        delta = sum((p.detach().cpu() - initial[name]).abs().sum().item()
                    for name, p in model.named_parameters() if p.requires_grad)
        assert delta > 0, "Adapter weights did not change"
        report["adapter_weight_l1_change"] = delta
        model.save_pretrained(args.output / "adapter", safe_serialization=True)
        report["status"] = "complete"
        print("AFTER", json.dumps(report["after"]), flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
