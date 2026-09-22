"""Qualify the frozen Cosmos Policy with its official 16-action horizon."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark
from PIL import Image, ImageDraw

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig, TASK_MAX_STEPS, prepare_observation


PREFIX = 16
SETTLE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--seeds", required=True, help="Comma-separated LIBERO initial-state indices.")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def policy_paths() -> tuple[str, str, str]:
    snapshot = os.environ.get("CFWAM_POLICY_SNAPSHOT")
    if snapshot:
        root = Path(snapshot)
        return (
            str(root / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
            str(root / "libero_dataset_statistics.json"),
            str(root / "libero_t5_embeddings.pkl"),
        )
    root = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    return root, f"{root}/libero_dataset_statistics.json", f"{root}/libero_t5_embeddings.pkl"


def build_policy_config() -> PolicyEvalConfig:
    checkpoint, statistics, embeddings = policy_paths()
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=checkpoint,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=statistics,
        t5_text_embeddings_path=embeddings,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=PREFIX,
        num_open_loop_steps=PREFIX,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=5,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 22), fill=(0, 0, 0))
    draw.text((4, 4), text, fill="white")
    return np.asarray(canvas)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = __import__("yaml").safe_load(args.task_config.read_text(encoding="utf-8"))
    task_id = int(raw.get("libero_task_id", str(raw["task_id"]).rsplit("_", 1)[-1]))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cfg = build_policy_config()
    stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, cosmos_config = get_model(cfg)
    trained_chunk = int(cosmos_config.dataloader_train.dataset.chunk_size)
    if trained_chunk != PREFIX:
        raise ValueError(f"checkpoint trained chunk is {trained_chunk}, expected {PREFIX}")
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(task_id)
    records = []
    for seed in seeds:
        run_dir = args.output / f"task{task_id}_seed{seed}_official16_clean"
        run_dir.mkdir(parents=True, exist_ok=True)
        env, description = get_libero_env(task, model_family="cosmos", resolution=256)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(task_id)[seed])
        for _ in range(SETTLE):
            observation, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
        primary_video, wrist_video, decisions = [], [], []
        executed_total = wam_calls = 0
        success = terminal = False
        while executed_total < TASK_MAX_STEPS["libero_10"] and not terminal:
            query_time = SETTLE + executed_total
            payload = prepare_observation(observation, resize_size=256, flip_images=cfg.flip_images)
            started = time.perf_counter()
            result = get_action(
                cfg, model, stats, payload, description,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            wam_calls += 1
            applied = 0
            for action in np.asarray(result["actions"], dtype=np.float32)[:PREFIX]:
                observation, _, done, info = env.step(action.tolist())
                executed_total += 1; applied += 1
                primary_video.append(caption(np.flipud(observation["agentview_image"]), f"official16 t={SETTLE + executed_total}"))
                wrist_video.append(caption(np.flipud(observation["robot0_eye_in_hand_image"]), f"official16 wrist t={SETTLE + executed_total}"))
                if done or info.get("success", False) or executed_total >= TASK_MAX_STEPS["libero_10"]:
                    success = bool(done or info.get("success", False))
                    terminal = True; break
            decisions.append({"t_query": query_time, "t_real": SETTLE + executed_total, "actions_executed": applied, "value": float(result["value_prediction"]), "wam_latency_ms": latency_ms})
        imageio.mimsave(run_dir / "full_episode_primary.mp4", primary_video, fps=10, macro_block_size=1)
        imageio.mimsave(run_dir / "full_episode_wrist.mp4", wrist_video, fps=10, macro_block_size=1)
        record = {"task_id": task_id, "seed": seed, "description": description, "trained_chunk_size": trained_chunk, "executed_prefix": PREFIX, "success": success, "environment_steps": executed_total, "wam_calls": wam_calls, "decisions": decisions}
        (run_dir / "official16_clean_log.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append({key: value for key, value in record.items() if key != "decisions"})
        print(json.dumps(records[-1]), flush=True)
    (args.output / f"task{task_id}_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
