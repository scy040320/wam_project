"""Run one real multi-segment CF-WAM local-repair rollout in LIBERO.

This is the first online integration of the frozen Cosmos Policy, DINOv2-S
residual encoder, trained attribution GAT and reviewed recovery router.  It
queries the WAM every four executed actions.  Simulator ground truth is never
read by the online decision path: the requested intervention is written only
to a separate post-hoc metadata file after the rollout ends.

The script intentionally reports decisions and safety stops, not a claim of
physical rollback.  A later A/B/C/D runner must use identical seeds and a
fixed budget to make success-rate claims.
"""

from __future__ import annotations

import argparse
import json
import os
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
from cfwam.runtime import AbstainPolicy
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    PolicyEvalConfig, TASK_MAX_STEPS, prepare_observation,
)

PREFIX, SETTLE = 4, 10
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
    parser.add_argument("--dino-device", choices=("cpu", "cuda"), default="cpu",
                        help="CPU default avoids concurrent WAM/DINO GPU residency on small GPUs.")
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


def main() -> int:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
    task_id = int(raw.get("libero_task_id", str(raw["task_id"]).rsplit("_", 1)[-1]))
    spec = TaskGraphSpec.from_yaml(args.task_config)
    graph = BeliefGraph(spec); router = RecoveryRouter(graph)
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    if thresholds.get("status") != "frozen_for_held_out_test":
        raise ValueError("Only validation-frozen thresholds may be used for online rollout")
    abstain = AbstainPolicy(float(thresholds["min_known_probability"]), float(thresholds["max_entropy"]), float(thresholds["max_residual_energy"]))
    prototype = torch.load(args.training_tensors, map_location="cpu", weights_only=False)[0]
    attributor = CounterfactualAttributor(prototype["residual"].numel(), prototype["node_features"].shape[-1], 5).cuda().eval()
    attributor.load_state_dict(torch.load(args.checkpoint, map_location="cuda", weights_only=True))
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(args.dino_device).eval()

    cfg = build_policy_config(); stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path); print("Loading frozen Cosmos Policy", flush=True)
    wam, _ = get_model(cfg)
    suite = benchmark.get_benchmark_dict()["libero_10"](); task = suite.get_task(task_id)
    env, description = get_libero_env(task, model_family="cosmos", resolution=256)
    env.reset(); observation = env.set_init_state(suite.get_task_init_states(task_id)[args.episode_id])
    for _ in range(SETTLE): observation, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
    graph.refresh(graph.beliefs, SETTLE, "current_observation")
    _, node_valid, edge_mask, mapping = graph_tensors(raw, args.phase)
    slot_to_actual = {slot: actual for actual, slot in canonical_map(raw).items()}
    node_ids = [slot_to_actual.get(slot, slot) for slot in SLOTS]
    edge_index, edge_type = prototype["edge_index"], prototype["edge_type"]
    phase_feature, _, _, _ = graph_tensors(raw, args.phase)
    object_hint = str(raw.get("intervention_object_joint_hint", ""))
    object_qpos = find_object_qpos(env, object_hint) if args.condition in {"object_shift", "unknown"} else None
    action_correction = np.zeros(3, dtype=np.float32)
    primary_video, wrist_video, records = [], [], []
    executed_total, block, success, safety_stopped = 0, 0, False, False
    while executed_total < TASK_MAX_STEPS["libero_10"] and not success:
        block += 1; query_time = SETTLE + executed_total
        primary_current, wrist_current = images(observation); proprio_current = proprio(observation)
        input_payload = prepare_observation(observation, resize_size=256, flip_images=cfg.flip_images)
        injected = block == args.intervention_block
        policy_input = occlude(input_payload) if injected and args.condition in {"visual_occlusion", "unknown"} else input_payload
        result = get_action(cfg, wam, stats, policy_input, description,
                            num_denoising_steps_action=cfg.num_denoising_steps_action,
                            generate_future_state_and_value_in_parallel=True)
        planned = np.asarray(result["actions"], dtype=np.float32)[:PREFIX]
        if injected and args.condition in {"object_shift", "unknown"}:
            env.sim.data.qpos[object_qpos:object_qpos + 3] += SHIFT_DELTA; env.sim.forward(); env._post_process(); env._update_observables(force=True)
        executed = planned.copy(); executed[:, :3] = np.clip(executed[:, :3] + action_correction, -1.0, 1.0)
        if injected and args.condition in {"action_noise", "unknown"}:
            executed[:, :3] = np.clip(executed[:, :3] + NOISE_DELTA, -1.0, 1.0)
        actual_steps = 0
        for action in executed:
            observation, _, done, info = env.step(action.tolist()); actual_steps += 1; executed_total += 1
            primary, wrist = images(observation)
            primary_video.append(caption(primary, f"block={block} t={SETTLE + executed_total}"))
            wrist_video.append(caption(wrist, f"wrist block={block} t={SETTLE + executed_total}"))
            if done or info.get("success", False) or executed_total >= TASK_MAX_STEPS["libero_10"]:
                success = bool(info.get("success", False)); break
        primary_real, wrist_real = images(observation); proprio_real = proprio(observation)
        p_res = feature(dino, dino_processor, result["future_image_predictions"]["future_image"], primary_real, args.dino_device)
        w_res = feature(dino, dino_processor, result["future_image_predictions"]["future_wrist_image"], wrist_real, args.dino_device)
        action_delta = torch.from_numpy(executed[:actual_steps] - planned[:actual_steps]).float().flatten()
        if actual_steps < PREFIX: action_delta = torch.cat((action_delta, torch.zeros((PREFIX - actual_steps) * planned.shape[1])))
        residual = torch.cat((p_res.flatten(), w_res.flatten(), torch.from_numpy(proprio_real - proprio_current), action_delta, torch.tensor([float(result["value_prediction"])], dtype=torch.float32))).unsqueeze(0).cuda()
        dynamic_features = phase_feature.clone()
        for slot, value in {"camera_primary": float(torch.linalg.vector_norm(p_res)), "camera_wrist": float(torch.linalg.vector_norm(w_res)), "effector_gripper": float(np.linalg.norm(proprio_real - proprio_current)), "action_segment_4": float(torch.linalg.vector_norm(action_delta))}.items():
            dynamic_features[SLOTS.index(slot), 7] = value
        with torch.inference_mode():
            output = attributor(residual, dynamic_features.unsqueeze(0).cuda(), edge_index.cuda(), edge_type.cuda(), edge_mask.unsqueeze(0).cuda(), node_valid.unsqueeze(0).cuda())
        attribution = decode_attribution(output, residual, node_ids, abstain)
        decision = router.route(attribution, SETTLE + executed_total)
        if decision.action.value == "local_action_correction":
            action_correction = -np.mean(executed[:actual_steps, :3] - planned[:actual_steps, :3], axis=0)
        else:
            action_correction[:] = 0
        query_dir = args.output / f"query_{block:03d}"; query_dir.mkdir(exist_ok=True)
        files = {"primary_current": save_png(primary_current, query_dir / "primary_current.png"), "wrist_current": save_png(wrist_current, query_dir / "wrist_current.png"), "primary_predicted": save_png(result["future_image_predictions"]["future_image"], query_dir / "primary_predicted.png"), "wrist_predicted": save_png(result["future_image_predictions"]["future_wrist_image"], query_dir / "wrist_predicted.png"), "primary_real": save_png(primary_real, query_dir / "primary_real_t_plus_4.png"), "wrist_real": save_png(wrist_real, query_dir / "wrist_real_t_plus_4.png")}
        np.save(query_dir / "planned_actions.npy", planned); np.save(query_dir / "executed_actions.npy", executed[:actual_steps]); np.save(query_dir / "action_execution_delta.npy", executed[:actual_steps] - planned[:actual_steps])
        row = {"block": block, "t_query": query_time, "t_real": SETTLE + executed_total, "intervention_injected": injected, "cause": attribution.cause.value, "abstained": attribution.abstained, "cause_probabilities": attribution.cause_probabilities, "affected_nodes": sorted(attribution.affected_nodes), "recovery_action": decision.action.value, "invalidated_nodes": sorted(decision.invalidated_nodes), "refresh_wam": decision.refresh_wam, "action_correction_next": action_correction.tolist(), "value": float(result["value_prediction"]), "files": files}
        (query_dir / "decision.json").write_text(json.dumps(row, indent=2), encoding="utf-8"); records.append(row)
        if decision.action.value == "safe_stop_global_refresh":
            safety_stopped = True; break
    imageio.mimsave(args.output / "full_episode_primary.mp4", primary_video, fps=10, macro_block_size=1)
    imageio.mimsave(args.output / "full_episode_wrist.mp4", wrist_video, fps=10, macro_block_size=1)
    (args.output / "online_decision_log.json").write_text(json.dumps({"method": "cfwam_dependency_aware_v1", "online_only": True, "episode_id": args.episode_id, "task_id": task_id, "success": success, "safety_stopped": safety_stopped, "policy_steps": executed_total, "decisions": records}, indent=2), encoding="utf-8")
    (args.output / "offline_evaluation_metadata.json").write_text(json.dumps({"condition": args.condition, "intervention_block": args.intervention_block, "warning": "post-hoc metadata; not opened by online router"}, indent=2), encoding="utf-8")
    print(json.dumps({"success": success, "safety_stopped": safety_stopped, "blocks": block, "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
