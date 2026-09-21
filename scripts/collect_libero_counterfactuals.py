"""Collect one aligned four-step Cosmos Policy + LIBERO counterfactual record.

This is deliberately a data collector, not an online policy replacement.  The
frozen Cosmos Policy provides planned actions, imagined futures and value;
LIBERO ground truth is used only after execution to write offline labels.
Run one record first in the visible Ubuntu desktop, then scale only after a
human review of the files and intervention behaviour.
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from libero.libero import benchmark
from PIL import Image

from cfwam.graph import BeliefGraph, TaskGraphSpec
from cfwam.protocol import DevelopmentSplit, offline_mask_for_cause
from cfwam.records import OfflineLabelRecord, OnlineStepRecord, write_aligned_record
from cfwam.types import Cause, NodeKind
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    PolicyEvalConfig, prepare_observation,
)

PREFIX, SETTLE = 4, 10
NOISE_DELTA = np.asarray([0.12, -0.12, 0.0], dtype=np.float32)
SHIFT_DELTA = np.asarray([0.25, -0.12, 0.02], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True, help="Reviewed task YAML")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--phase", choices=["approach", "grasp", "transport"], default="approach")
    parser.add_argument("--condition", choices=[item.value for item in Cause], default="normal")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="Deterministic multiplier for action-noise diagnostics; must be positive.")
    parser.add_argument("--output", default="outputs/cfwam_v1_records")
    return parser.parse_args()


def save_image(array: np.ndarray, path: Path) -> str:
    Image.fromarray(array).save(path)
    return path.name


def images_from_observation(observation):
    return (
        np.flipud(observation["agentview_image"]),
        np.flipud(observation["robot0_eye_in_hand_image"]),
    )


def proprio_from_observation(observation):
    return np.concatenate((observation["robot0_gripper_qpos"], observation["robot0_eef_pos"], observation["robot0_eef_quat"]))


def find_object_qpos(env, hint: str) -> int:
    """Find the position part of a free-joint by reviewed semantic name."""
    for joint_id in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(joint_id) or ""
        if hint.lower() in name.lower():
            return int(env.sim.model.jnt_qposadr[joint_id])
    raise RuntimeError(f"No LIBERO joint matched reviewed object hint: {hint!r}")


def apply_shift(env, object_hint: str):
    address = find_object_qpos(env, object_hint)
    env.sim.data.qpos[address:address + 3] += SHIFT_DELTA
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def occlude(model_input):
    patched = deepcopy(model_input)
    image = patched["primary_image"].copy()
    height, width = image.shape[:2]
    image[height // 4:height * 3 // 4, width // 3:width * 2 // 3] = 0
    patched["primary_image"] = image
    return patched


def build_config():
    # A local HF snapshot is required for reliable offline/cloud collection;
    # otherwise the official helper interprets the model-relative asset paths
    # as Hub URLs even when the checkpoint itself is already cached.
    snapshot = os.environ.get("CFWAM_POLICY_SNAPSHOT")
    if snapshot:
        root = Path(snapshot)
        ckpt_path = str(root / "Cosmos-Policy-LIBERO-Predict2-2B.pt")
        dataset_stats_path = str(root / "libero_dataset_statistics.json")
        t5_text_embeddings_path = str(root / "libero_t5_embeddings.pkl")
    else:
        ckpt_path = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
        dataset_stats_path = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
        t5_text_embeddings_path = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=ckpt_path,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=dataset_stats_path,
        t5_text_embeddings_path=t5_text_embeddings_path,
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, chunk_size=PREFIX, num_open_loop_steps=PREFIX,
        trained_with_image_aug=True, use_jpeg_compression=True, flip_images=True,
        num_denoising_steps_action=5, num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )


def main():
    args = parse_args()
    if args.noise_scale <= 0:
        raise ValueError("--noise-scale must be positive")
    raw = json.loads(json.dumps({}))  # explicit: semantic config is read below, not simulator GT
    spec = TaskGraphSpec.from_yaml(args.task_config)
    import yaml
    with open(args.task_config, encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    task_id = int(raw.get("libero_task_id", str(raw["task_id"]).rsplit("_", 1)[-1]))
    phase_step = int(raw["phase_steps"].get(args.phase, raw["phase_steps"][f"phase_{args.phase}"]))
    object_hint = str(raw.get("intervention_object_joint_hint", ""))
    if args.condition in {Cause.OBJECT_SHIFT.value, Cause.UNKNOWN.value} and not object_hint:
        raise ValueError("object shift / unknown needs intervention_object_joint_hint in reviewed YAML")

    # On an 8 GB WSL GPU, reserve model memory before EGL creates its render
    # context.  The policy sees the same observations; this only avoids a
    # renderer-first allocation race during checkpoint loading.
    cfg = build_config()
    stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    print("Loading frozen Cosmos Policy checkpoint...", flush=True)
    model, _ = get_model(cfg)

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(task_id)
    env, description = get_libero_env(task, model_family="cosmos", resolution=256)
    env.reset()
    observation = env.set_init_state(suite.get_task_init_states(task_id)[args.episode_id])
    for _ in range(SETTLE):
        observation, _, _, _ = env.step(get_libero_dummy_action("cosmos"))

    # Move deterministically to the reviewed phase boundary in full 4-step chunks.
    elapsed = SETTLE
    while elapsed < phase_step:
        inp = prepare_observation(observation, resize_size=256, flip_images=cfg.flip_images)
        warmup = get_action(cfg, model, stats, inp, description,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True)
        for action in np.asarray(warmup["actions"], dtype=np.float32)[: min(PREFIX, phase_step - elapsed)]:
            observation, _, done, info = env.step(action.tolist())
            elapsed += 1
            if done or info.get("success", False):
                break
        if done or info.get("success", False):
            raise RuntimeError("Episode ended before the selected semantic phase boundary")

    primary_current, wrist_current = images_from_observation(observation)
    proprio_current = proprio_from_observation(observation)
    model_input = prepare_observation(observation, resize_size=256, flip_images=cfg.flip_images)
    candidate_input = occlude(model_input) if args.condition in {
        Cause.VISUAL_OCCLUSION.value, Cause.UNKNOWN.value
    } else model_input
    result = get_action(cfg, model, stats, candidate_input, description,
                        num_denoising_steps_action=cfg.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=True)
    planned = np.asarray(result["actions"], dtype=np.float32)[:PREFIX]

    if args.condition in {Cause.OBJECT_SHIFT.value, Cause.UNKNOWN.value}:
        apply_shift(env, object_hint)

    executed = planned.copy()
    if args.condition in {Cause.ACTION_NOISE.value, Cause.UNKNOWN.value}:
        executed[:, :3] = np.clip(executed[:, :3] + NOISE_DELTA * args.noise_scale, -1.0, 1.0)
    for action in executed:
        observation, _, done, info = env.step(action.tolist())
        if done or info.get("success", False):
            break
    primary_real, wrist_real = images_from_observation(observation)
    proprio_real = proprio_from_observation(observation)

    suffix = f"_noise{args.noise_scale:g}" if args.condition == Cause.ACTION_NOISE.value else ""
    output = Path(args.output) / f"task{task_id}_seed{args.episode_id:02d}_{args.phase}_{args.condition}{suffix}"
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "primary_current": save_image(primary_current, output / "primary_current.png"),
        "wrist_current": save_image(wrist_current, output / "wrist_current.png"),
        "primary_predicted": save_image(result["future_image_predictions"]["future_image"], output / "primary_predicted.png"),
        "wrist_predicted": save_image(result["future_image_predictions"]["future_wrist_image"], output / "wrist_predicted.png"),
        "primary_real": save_image(primary_real, output / "primary_real_t_plus_4.png"),
        "wrist_real": save_image(wrist_real, output / "wrist_real_t_plus_4.png"),
    }
    np.save(output / "planned_actions.npy", planned)
    np.save(output / "executed_actions.npy", executed)
    np.save(output / "action_execution_delta.npy", executed - planned)
    np.save(output / "proprio_current.npy", proprio_current)
    np.save(output / "proprio_real_t_plus_4.npy", proprio_real)
    graph = BeliefGraph(spec)
    graph.refresh(graph.beliefs, timestamp=elapsed, provenance="current_observation")
    cause = Cause(args.condition)
    # These names are simulator-derived supervision metadata only. They are
    # written after online data capture and never supplied to the policy.
    object_nodes = [node.node_id for node in spec.nodes if node.kind is NodeKind.OBJECT]
    target_nodes = [node.node_id for node in spec.nodes if node.kind is NodeKind.TARGET]
    shifted_object = next((node for node in object_nodes if object_hint.replace("_", "") in node.replace("_", "")),
                          object_nodes[0] if object_nodes else None)
    target_node = target_nodes[0] if target_nodes else None
    online = OnlineStepRecord(
        run_id=output.name, task_id=spec.task_id, split=DevelopmentSplit().subset(args.episode_id),
        phase=args.phase, episode_id=args.episode_id, t_query=elapsed, t_real=elapsed + len(executed),
        requested_prefix=PREFIX, executed_prefix=len(executed), primary_current=files["primary_current"],
        wrist_current=files["wrist_current"], primary_prediction=files["primary_predicted"],
        wrist_prediction=files["wrist_predicted"], primary_actual=files["primary_real"],
        wrist_actual=files["wrist_real"], action_plan="planned_actions.npy", action_executed="executed_actions.npy",
        action_execution_delta="action_execution_delta.npy", proprio_current="proprio_current.npy",
        proprio_actual="proprio_real_t_plus_4.npy", value_prediction=float(result["value_prediction"]),
        belief_snapshot=graph.snapshot(), intervention_description="hidden from online policy; simulator used offline only",
    )
    offline = OfflineLabelRecord(
        run_id=output.name, cause=cause.value,
        affected_nodes=offline_mask_for_cause(cause, shifted_object, target_node),
        intervention_parameters={"shift_delta": SHIFT_DELTA.tolist() if args.condition in {"object_shift", "unknown"} else None,
                                 "noise_delta": (NOISE_DELTA * args.noise_scale).tolist() if args.condition == "action_noise" else (NOISE_DELTA.tolist() if args.condition == "unknown" else None),
                                 "noise_scale": args.noise_scale if args.condition == "action_noise" else None,
                                 "occlusion": args.condition in {"visual_occlusion", "unknown"}},
    )
    write_aligned_record(output, online, offline)
    (output / "manifest.json").write_text(json.dumps({"task_id": task_id, "episode_id": args.episode_id,
        "phase": args.phase, "condition": args.condition, "files": files}, indent=2), encoding="utf-8")
    print(f"Saved aligned 4-step record: {output}", flush=True)


if __name__ == "__main__":
    main()
