"""Run one native-16 temporal-guard full-episode rollout in LIBERO.

The runner adds validation-frozen strong-unknown detection and episode-local
weak-evidence hysteresis. Cosmos always decodes and executes its native 16-step
action chunk so the predicted future and real observation share the same
horizon. A first weak signal produces a four-step guarded hold
with zero Cartesian delta and the previous gripper command; a strong signal or
two consecutive weak action segments produces the existing global safe stop.
The hold is logged separately and is never counted as independent attribution
evidence. Simulator truth remains offline-only.
"""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import time
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from libero.libero import benchmark
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel

from build_training_tensors import SLOTS, canonical_map, graph_tensors
from cfwam.decision import decode_attribution
from cfwam.graph import BeliefGraph, TaskGraphSpec
from cfwam.model import CounterfactualAttributor
from cfwam.recovery import RecoveryRouter
from cfwam.runtime import AbstainPolicy, CausalConsistencyGuard, TemporalAbstainGate
from cfwam.types import AttributionResult, Cause
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    PolicyEvalConfig, TASK_MAX_STEPS, prepare_observation,
)

PREFIX, SETTLE = 16, 10
# The v1 attributor was trained with four action-delta rows. Preserve that
# input ABI while the controller itself uses the checkpoint's native horizon.
RESIDUAL_ACTION_STEPS = 4
SHIFT_DELTA = np.asarray([0.25, -0.12, 0.02], dtype=np.float64)
NOISE_DELTA = np.asarray([0.12, -0.12, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--training-tensors", required=True, type=Path,
                        help="Used only to recover the fixed residual/model dimensions.")
    parser.add_argument("--episode-id", type=int, default=32,
                        help="Use a held-out, non-training initial state for an integration rollout.")
    parser.add_argument("--phase", choices=("approach", "grasp", "transport"), default="approach")
    parser.add_argument("--condition", choices=("normal", "visual_occlusion", "object_shift", "action_noise", "unknown"), default="normal")
    parser.add_argument("--intervention-block", type=int, default=2)
    parser.add_argument("--max-policy-steps", type=int, default=None,
                        help="Optional validation-calibration budget; defaults to the LIBERO task maximum.")
    parser.add_argument("--dino-device", choices=("cpu", "cuda"), default="cpu",
                        help="CPU default avoids concurrent WAM/DINO GPU residency on small GPUs.")
    parser.add_argument("--observe-only", action="store_true",
                        help="Record counterfactual recovery decisions without applying them or safety-stopping.")
    parser.add_argument("--approach", required=True, choices=("A_binary_global", "B_uniform_subgraph", "C_attribution_global", "D_dependency_aware"))
    parser.add_argument("--binary-mae-threshold", type=float, default=13.5)
    parser.add_argument("--hold-steps", type=int, default=None, help="Must match the validation-frozen configuration.")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()

def policy_paths() -> tuple[str, str, str]:
    snapshot = os.environ.get("CFWAM_POLICY_SNAPSHOT")
    if snapshot:
        root = Path(snapshot)
        return (str(root / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
                str(root / "libero_dataset_statistics.json"),
                str(root / "libero_t5_embeddings.pkl"))
    root = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    return root, f"{root}/libero_dataset_statistics.json", f"{root}/libero_t5_embeddings.pkl"


def build_policy_config() -> PolicyEvalConfig:
    ckpt_path, stats_path, embeddings_path = policy_paths()
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only", ckpt_path=ckpt_path,
        config_file="cosmos_policy/config/config.py", dataset_stats_path=stats_path,
        t5_text_embeddings_path=embeddings_path, use_wrist_image=True, use_proprio=True,
        normalize_proprio=True, unnormalize_actions=True, chunk_size=PREFIX,
        num_open_loop_steps=PREFIX, trained_with_image_aug=True, use_jpeg_compression=True,
        flip_images=True, num_denoising_steps_action=5,
        num_denoising_steps_future_state=1, num_denoising_steps_value=1,
    )


def occlude(model_input: dict) -> dict:
    patched = deepcopy(model_input)
    image = patched["primary_image"].copy(); height, width = image.shape[:2]
    image[height // 4:height * 3 // 4, width // 3:width * 2 // 3] = 0
    patched["primary_image"] = image
    return patched


def find_object_qpos(env, hint: str) -> int:
    for joint_id in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(joint_id) or ""
        if hint.lower() in name.lower():
            return int(env.sim.model.jnt_qposadr[joint_id])
    raise RuntimeError(f"No free-joint matches reviewed object hint {hint!r}")


def images(observation):
    return np.flipud(observation["agentview_image"]), np.flipud(observation["robot0_eye_in_hand_image"])


def image_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    if actual.shape[:2] != predicted.shape[:2]:
        actual = np.asarray(Image.fromarray(actual).resize((predicted.shape[1], predicted.shape[0]), Image.Resampling.LANCZOS))
    return float(np.mean(np.abs(predicted.astype(np.float32) - actual.astype(np.float32))))


def proprio(observation) -> np.ndarray:
    return np.concatenate((observation["robot0_gripper_qpos"], observation["robot0_eef_pos"], observation["robot0_eef_quat"])).astype(np.float32)


def feature(model, processor, predicted: np.ndarray, actual: np.ndarray, device: str) -> torch.Tensor:
    pil = [Image.fromarray(image).convert("RGB") for image in (predicted, actual)]
    inputs = processor(images=pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        tokens = model(**inputs).last_hidden_state[:, 0].float().cpu()
    return tokens[1] - tokens[0]


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(canvas); draw.rectangle((0, 0, canvas.width, 22), fill=(0, 0, 0)); draw.text((4, 4), text, fill="white")
    return np.asarray(canvas)


def save_png(image: np.ndarray, path: Path) -> str:
    Image.fromarray(image).save(path); return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:

    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
    task_id = int(raw.get("libero_task_id", str(raw["task_id"]).rsplit("_", 1)[-1]))
    spec = TaskGraphSpec.from_yaml(args.task_config)
    graph = BeliefGraph(spec); router = RecoveryRouter(graph)
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    if thresholds.get("status") != "frozen_validation_temporal_guard":
        raise ValueError("online control requires validation-frozen temporal-guard thresholds")
    abstain = AbstainPolicy(float(thresholds["min_known_probability"]), float(thresholds["max_entropy"]), float(thresholds["max_residual_energy"]))
    prototype = torch.load(args.training_tensors, map_location="cpu", weights_only=False)[0]
    attributor = CounterfactualAttributor(prototype["residual"].numel(), prototype["node_features"].shape[-1], 5).cuda().eval()
    attributor.load_state_dict(torch.load(args.checkpoint, map_location="cuda", weights_only=True))
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(args.dino_device).eval()

    cfg = build_policy_config(); stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path); print("Loading frozen Cosmos Policy", flush=True)
    wam, cosmos_config = get_model(cfg)
    trained_chunk = int(cosmos_config.dataloader_train.dataset.chunk_size)
    if trained_chunk != PREFIX:
        raise ValueError(f"checkpoint trained chunk is {trained_chunk}, expected {PREFIX}")
    suite = benchmark.get_benchmark_dict()["libero_10"](); task = suite.get_task(task_id)
    env, description = get_libero_env(task, model_family="cosmos", resolution=256)
    env.reset(); observation = env.set_init_state(suite.get_task_init_states(task_id)[args.episode_id])
    for _ in range(SETTLE): observation, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
    graph.refresh(graph.beliefs, SETTLE, "current_observation")
    _, node_valid, edge_mask, mapping = graph_tensors(raw, args.phase)
    slot_to_actual = {slot: actual for actual, slot in canonical_map(raw).items()}
    node_ids = [slot_to_actual.get(slot, slot) for slot in SLOTS]
    temporal_gate = TemporalAbstainGate(
        float(thresholds["strong_unknown_probability"]),
        int(thresholds.get("weak_confirmations", 2)),
    )
    causal_guard = CausalConsistencyGuard(
        float(thresholds["action_residual_l2_threshold"]),
        float(thresholds["object_shift_energy_delta_threshold"]),
    )
    frozen_hold_steps = int(thresholds.get("hold_steps", 4))
    if args.hold_steps is not None and args.hold_steps != frozen_hold_steps:
        raise ValueError("--hold-steps must match the validation-frozen configuration")
    hold_steps = frozen_hold_steps
    threshold_hash = sha256_file(args.thresholds)
    checkpoint_hash = sha256_file(args.checkpoint)
    edge_index, edge_type = prototype["edge_index"], prototype["edge_type"]
    phase_feature, _, _, _ = graph_tensors(raw, args.phase)
    object_hint = str(raw.get("intervention_object_joint_hint", ""))
    object_qpos = find_object_qpos(env, object_hint) if args.condition in {"object_shift", "unknown"} else None
    object_nodes = {
        node.node_id for node in spec.nodes
        if node.node_id.startswith("object_") or node.node_id.startswith("target_")
    }
    action_correction = np.zeros(3, dtype=np.float32)
    primary_video, wrist_video, records = [], [], []
    wam_calls, global_refreshes, invalidated_total, guarded_reobserves = 0, 0, 0, 0
    planned_action_steps_total, guard_hold_steps_total = 0, 0
    latency_ms = []
    max_policy_steps = args.max_policy_steps or TASK_MAX_STEPS["libero_10"]
    executed_total, block, success, safety_stopped, terminal = 0, 0, False, False, False
    previous_residual_energy = None
    while executed_total < max_policy_steps and not terminal:
        block += 1; query_time = SETTLE + executed_total
        primary_current, wrist_current = images(observation); proprio_current = proprio(observation)
        input_payload = prepare_observation(observation, resize_size=256, flip_images=cfg.flip_images)
        injected = block == args.intervention_block
        policy_input = occlude(input_payload) if injected and args.condition in {"visual_occlusion", "unknown"} else input_payload
        query_started = time.perf_counter()
        result = get_action(cfg, wam, stats, policy_input, description,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True)
        latency_ms.append((time.perf_counter() - query_started) * 1000.0)
        wam_calls += 1
        planned = np.asarray(result["actions"], dtype=np.float32)[:PREFIX]
        if injected and args.condition in {"object_shift", "unknown"}:
            env.sim.data.qpos[object_qpos:object_qpos + 3] += SHIFT_DELTA; env.sim.forward(); env._post_process(); env._update_observables(force=True)
        # Separate an intentional controller correction from an exogenous
        # execution deviation.  Attribution must compare what was actually
        # executed with what the controller commanded, otherwise its own local
        # correction is misdiagnosed as fresh action noise on the next block.
        commanded = planned.copy()
        commanded[:, :3] = np.clip(commanded[:, :3] + action_correction, -1.0, 1.0)
        executed = commanded.copy()
        if injected and args.condition in {"action_noise", "unknown"}:
            executed[:, :3] = np.clip(executed[:, :3] + NOISE_DELTA, -1.0, 1.0)
        actual_steps = 0
        for action in executed:
            observation, _, done, info = env.step(action.tolist()); actual_steps += 1; executed_total += 1
            primary, wrist = images(observation)
            primary_video.append(caption(primary, f"block={block} t={SETTLE + executed_total}"))
            wrist_video.append(caption(wrist, f"wrist block={block} t={SETTLE + executed_total}"))
            if done or info.get("success", False) or executed_total >= max_policy_steps:
                success = bool(done or info.get("success", False))
                terminal = True; break
        planned_action_steps_total += actual_steps
        primary_real, wrist_real = images(observation); proprio_real = proprio(observation)
        p_res = feature(dino, dino_processor, result["future_image_predictions"]["future_image"], primary_real, args.dino_device)
        w_res = feature(dino, dino_processor, result["future_image_predictions"]["future_wrist_image"], wrist_real, args.dino_device)
        executed_delta = executed[:actual_steps] - commanded[:actual_steps]
        residual_delta = executed_delta[:RESIDUAL_ACTION_STEPS]
        action_delta = torch.from_numpy(residual_delta).float().flatten()
        if residual_delta.shape[0] < RESIDUAL_ACTION_STEPS:
            action_delta = torch.cat((action_delta, torch.zeros((RESIDUAL_ACTION_STEPS - residual_delta.shape[0]) * planned.shape[1])))
        residual = torch.cat((p_res.flatten(), w_res.flatten(), torch.from_numpy(proprio_real - proprio_current), action_delta, torch.tensor([float(result["value_prediction"])], dtype=torch.float32))).unsqueeze(0).cuda()
        dynamic_features = phase_feature.clone()
        for slot, value in {"camera_primary": float(torch.linalg.vector_norm(p_res)), "camera_wrist": float(torch.linalg.vector_norm(w_res)), "effector_gripper": float(np.linalg.norm(proprio_real - proprio_current)), "action_segment_4": float(torch.linalg.vector_norm(action_delta))}.items():
            dynamic_features[SLOTS.index(slot), 7] = value
        with torch.inference_mode():
            output = attributor(residual, dynamic_features.unsqueeze(0).cuda(), edge_index.cuda(), edge_type.cuda(), edge_mask.unsqueeze(0).cuda(), node_valid.unsqueeze(0).cuda())
        model_attribution = decode_attribution(output, residual, node_ids, abstain)
        raw_cause = max(model_attribution.cause_probabilities, key=model_attribution.cause_probabilities.get)
        action_residual_l2 = float(np.linalg.norm(residual_delta[:, :3]))
        energy_delta = (
            0.0 if previous_residual_energy is None
            else float(model_attribution.residual_energy - previous_residual_energy)
        )
        consistency = causal_guard.apply(
            model_attribution, action_residual_l2, energy_delta, object_nodes,
        )
        attribution = consistency.attribution
        previous_residual_energy = model_attribution.residual_energy
        max_known_probability = max(
            probability for cause, probability in attribution.cause_probabilities.items()
            if cause.value != "unknown"
        )
        abstain_reasons = []
        if raw_cause.value == "unknown":
            abstain_reasons.append("raw_unknown")
        if max_known_probability < abstain.min_known_probability:
            abstain_reasons.append("low_known_probability")
        if attribution.entropy > abstain.max_entropy:
            abstain_reasons.append("high_entropy")
        if attribution.residual_energy > abstain.max_residual_energy:
            abstain_reasons.append("high_residual_energy")
        pixel_mae = image_mae(result["future_image_predictions"]["future_image"], primary_real)
        all_nodes = [node.node_id for node in spec.nodes]
        large_nodes = [node_id for node_id in all_nodes if not node_id.startswith("phase_")]
        timestamp = SETTLE + executed_total
        guard_level = "not_applicable"
        weak_streak = 0
        strong_unknown = False
        unknown_probability = float(attribution.cause_probabilities.get(Cause.UNKNOWN, 0.0))
        hold_steps_executed = 0
        if args.approach == "A_binary_global":
            mismatch = pixel_mae > args.binary_mae_threshold
            action = "global_refresh" if mismatch else "continue"
            invalidated = graph.invalidate(all_nodes, "binary_global", timestamp) if mismatch else []
            refresh_wam, safety_stop = mismatch, False
        elif args.approach == "B_uniform_subgraph":
            mismatch = pixel_mae > args.binary_mae_threshold
            action = "large_subgraph_refresh" if mismatch else "continue"
            invalidated = graph.invalidate(large_nodes, "uniform_subgraph", timestamp) if mismatch else []
            refresh_wam, safety_stop = mismatch, False
        elif args.approach == "C_attribution_global":
            if attribution.abstained or attribution.cause.value == "unknown":
                action, invalidated, refresh_wam, safety_stop = "safe_stop_global_refresh", graph.invalidate(all_nodes, "unknown", timestamp), True, True
            elif attribution.cause.value == "normal":
                action, invalidated, refresh_wam, safety_stop = "continue", [], False, False
            else:
                action, invalidated, refresh_wam, safety_stop = "global_refresh", graph.invalidate(all_nodes, "attribution_global", timestamp), True, False
        else:
            immediate_causal_unknown = consistency.reason in {
                "simultaneous_visual_and_execution_evidence",
                "model_unknown_with_execution_evidence",
            }
            guard = temporal_gate.update(
                attribution, immediate_safe_stop=immediate_causal_unknown,
            )
            guard_level = guard.level.value
            weak_streak = guard.weak_streak
            strong_unknown = guard.level.value == "strong"
            unknown_probability = guard.unknown_probability
            active_router = RecoveryRouter(deepcopy(graph)) if args.observe_only else router
            if guard.force_safe_stop:
                forced_unknown = AttributionResult(
                    Cause.UNKNOWN, attribution.cause_probabilities, attribution.affected_nodes,
                    attribution.residual_energy, attribution.entropy, True,
                )
                decision = active_router.route(forced_unknown, timestamp)
            elif guard.guarded_reobserve:
                decision = active_router.guarded_reobserve(timestamp)
            else:
                decision = active_router.route(attribution, timestamp)
            action = decision.action.value
            invalidated = list(decision.invalidated_nodes)
            refresh_wam = decision.refresh_wam
            safety_stop = action == "safe_stop_global_refresh"
            if action == "guarded_reobserve":
                guarded_reobserves += 1
        if action in {"global_refresh", "safe_stop_global_refresh"}:
            global_refreshes += 1
        invalidated_total += len(invalidated)
        if not args.observe_only and action == "local_action_correction":
            action_correction = -np.mean(executed_delta[:, :3], axis=0)
        else:
            action_correction[:] = 0
        query_dir = args.output / f"query_{block:03d}"; query_dir.mkdir(exist_ok=True)
        files = {"primary_current": save_png(primary_current, query_dir / "primary_current.png"), "wrist_current": save_png(wrist_current, query_dir / "wrist_current.png"), "primary_predicted": save_png(result["future_image_predictions"]["future_image"], query_dir / "primary_predicted.png"), "wrist_predicted": save_png(result["future_image_predictions"]["future_wrist_image"], query_dir / "wrist_predicted.png"), "primary_real": save_png(primary_real, query_dir / "primary_real_t_plus_16.png"), "wrist_real": save_png(wrist_real, query_dir / "wrist_real_t_plus_16.png")}
        np.save(query_dir / "planned_actions.npy", planned)
        np.save(query_dir / "commanded_actions.npy", commanded[:actual_steps])
        np.save(query_dir / "executed_actions.npy", executed[:actual_steps])
        np.save(query_dir / "controller_action_delta.npy", commanded[:actual_steps] - planned[:actual_steps])
        np.save(query_dir / "action_execution_delta.npy", executed_delta)
        row = {"approach": args.approach, "block": block, "t_query": query_time, "t_real": SETTLE + executed_total, "intervention_injected": injected, "raw_cause": raw_cause.value, "cause": attribution.cause.value, "causal_consistency_reason": consistency.reason, "action_residual_l2": action_residual_l2, "residual_energy_delta": energy_delta, "abstained": attribution.abstained, "abstain_reasons": abstain_reasons, "max_known_probability": max_known_probability, "unknown_probability": unknown_probability, "guard_level": guard_level, "weak_streak": weak_streak, "strong_unknown": strong_unknown, "hold_steps_requested": hold_steps if action == "guarded_reobserve" else 0, "hold_steps_executed": 0, "entropy": attribution.entropy, "residual_energy": attribution.residual_energy, "prediction_real_mae": pixel_mae, "binary_mae_threshold": args.binary_mae_threshold, "cause_probabilities": attribution.cause_probabilities, "affected_nodes": sorted(attribution.affected_nodes), "recovery_action": action, "would_safety_stop": safety_stop, "invalidated_nodes": sorted(invalidated), "refresh_wam": refresh_wam, "action_correction_next": action_correction.tolist(), "value": float(result["value_prediction"]), "wam_latency_ms": latency_ms[-1], "files": files}
        if not args.observe_only and action == "guarded_reobserve" and not terminal:
            hold_action = np.zeros(planned.shape[1], dtype=np.float32)
            hold_action[-1] = float(executed[max(actual_steps - 1, 0), -1] if actual_steps else planned[0, -1])
            np.save(query_dir / "guard_hold_action.npy", hold_action)
            for hold_index in range(hold_steps):
                if executed_total >= max_policy_steps:
                    terminal = True; break
                observation, _, done, info = env.step(hold_action.tolist())
                executed_total += 1; hold_steps_executed += 1; guard_hold_steps_total += 1
                primary, wrist = images(observation)
                primary_video.append(caption(primary, f"guarded hold={hold_index + 1}/{hold_steps} t={SETTLE + executed_total}"))
                wrist_video.append(caption(wrist, f"wrist guarded hold={hold_index + 1}/{hold_steps} t={SETTLE + executed_total}"))
                if done or info.get("success", False):
                    success = bool(done or info.get("success", False))
                    terminal = True; break
            row["hold_steps_executed"] = hold_steps_executed
            row["t_after_hold"] = SETTLE + executed_total
        (query_dir / "decision.json").write_text(json.dumps(row, indent=2), encoding="utf-8"); records.append(row)
        if not args.observe_only and safety_stop:
            safety_stopped = True; break
    imageio.mimsave(args.output / "full_episode_primary.mp4", primary_video, fps=10, macro_block_size=1)
    imageio.mimsave(args.output / "full_episode_wrist.mp4", wrist_video, fps=10, macro_block_size=1)
    (args.output / "online_decision_log.json").write_text(json.dumps({"method": args.approach, "online_only": True, "observe_only": args.observe_only, "control_horizon": PREFIX, "prediction_alignment_horizon": PREFIX, "attributor_action_residual_steps": RESIDUAL_ACTION_STEPS, "episode_id": args.episode_id, "task_id": task_id, "condition": args.condition, "success": success, "safety_stopped": safety_stopped, "would_safety_stop_count": sum(int(row["would_safety_stop"]) for row in records), "environment_steps": executed_total, "planned_action_steps": planned_action_steps_total, "guard_hold_steps": guard_hold_steps_total, "max_policy_steps": max_policy_steps, "wam_calls": wam_calls, "global_refreshes": global_refreshes, "guarded_reobserves": guarded_reobserves, "invalidated_node_total": invalidated_total, "threshold_config_sha256": threshold_hash, "checkpoint_sha256": checkpoint_hash, "mean_wam_latency_ms": float(np.mean(latency_ms)) if latency_ms else 0.0, "p95_wam_latency_ms": float(np.percentile(latency_ms, 95)) if latency_ms else 0.0, "decisions": records}, indent=2), encoding="utf-8")
    (args.output / "offline_evaluation_metadata.json").write_text(json.dumps({"condition": args.condition, "intervention_block": args.intervention_block, "warning": "post-hoc metadata; not opened by online router"}, indent=2), encoding="utf-8")
    print(json.dumps({"success": success, "safety_stopped": safety_stopped, "observe_only": args.observe_only, "would_safety_stop_count": sum(int(row["would_safety_stop"]) for row in records), "blocks": block, "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
