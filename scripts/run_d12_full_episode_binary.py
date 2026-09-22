"""Complete-episode custom binary diagnostic baseline for Cosmos Policy + LIBERO.

This is NOT an implementation claim about Cosmos Policy's original controller.
It is an explicit diagnostic wrapper: after each executed 16-action block, the
prediction/real-image MAE selects either ``continue`` or ``global_refresh``.
The latter adds a fresh WAM query, has no causal label at decision time, and
never claims to roll back physical simulator state.
"""
import csv
import json
import os
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark
from PIL import Image, ImageChops, ImageDraw

from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig, TASK_MAX_STEPS, prepare_observation

OUT = Path(os.environ.get("D12_FULL_OUTPUT_DIR", "outputs/d12_full_episode_binary_v1"))
OUT.mkdir(parents=True, exist_ok=True)
SUITE, TASK_ID, SETTLE, PREFIX = "libero_10", 0, 10, 16
EPISODE_IDS = [int(v) for v in os.environ.get("D12_EPISODE_IDS", "0").split(",") if v.strip()]
CONDITIONS = [v.strip() for v in os.environ.get("D12_CONDITIONS", "clean,visual_occlusion,object_shift,action_noise").split(",") if v.strip()]
# The threshold is part of the diagnostic-method card.  Allow an explicitly
# logged override for a no-trigger control; never tune it on the same rollout.
THRESHOLD = float(os.environ.get("D12_MAE_THRESHOLD", "13.50"))
SHIFT_DELTA = np.asarray([0.25, -0.12, 0.02], dtype=np.float32)
NOISE_DELTA = np.asarray([0.12, -0.12, 0.0], dtype=np.float32)

def mae(predicted, actual):
    actual_224 = np.asarray(Image.fromarray(actual).resize((predicted.shape[1], predicted.shape[0]), Image.Resampling.LANCZOS))
    return float(np.mean(np.abs(predicted.astype(np.float32) - actual_224.astype(np.float32))))


def find_object_qpos(env, hint: str = "tomato"):
    """Resolve the reviewed free-joint by name rather than relying on a qpos offset."""
    for joint_id in range(env.sim.model.njnt):
        name = env.sim.model.joint_id2name(joint_id) or ""
        if hint.lower() in name.lower():
            return int(env.sim.model.jnt_qposadr[joint_id])
    raise RuntimeError(f"No LIBERO free joint matched object hint {hint!r}")


def model_paths():
    snapshot = os.environ.get("CFWAM_POLICY_SNAPSHOT")
    if not snapshot:
        return (
            "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
            "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
            "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        )
    root = Path(snapshot)
    return (str(root / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
            str(root / "libero_dataset_statistics.json"),
            str(root / "libero_t5_embeddings.pkl"))

def draw_frame(image, caption):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 22), fill=(0, 0, 0))
    draw.text((4, 4), caption, fill=(255, 255, 255))
    return np.asarray(canvas)

def save_alignment(folder, idx, current, predicted, actual, err, decision):
    target = (predicted.shape[1], predicted.shape[0])
    current = Image.fromarray(current).resize(target, Image.Resampling.LANCZOS)
    actual = Image.fromarray(actual).resize(target, Image.Resampling.LANCZOS)
    predicted = Image.fromarray(predicted)
    diff = ImageChops.difference(predicted, actual)
    canvas = Image.new("RGB", (target[0] * 4, target[1] + 26), "black")
    for col, (image, label) in enumerate(zip([current, predicted, actual, diff], ["current", "predicted future", "real after 16", f"MAE={err:.2f}: {decision}"])):
        canvas.paste(image, (col * target[0], 26))
        ImageDraw.Draw(canvas).text((col * target[0] + 4, 5), label, fill="white")
    canvas.save(folder / f"query_{idx:02d}_four_panel.png")

suite = benchmark.get_benchmark_dict()[SUITE]()
task = suite.get_task(TASK_ID)
ckpt_path, dataset_stats_path, t5_text_embeddings_path = model_paths()
cfg = PolicyEvalConfig(config="cosmos_predict2_2b_480p_libero__inference_only", ckpt_path=ckpt_path, config_file="cosmos_policy/config/config.py", dataset_stats_path=dataset_stats_path, t5_text_embeddings_path=t5_text_embeddings_path, use_wrist_image=True, use_proprio=True, normalize_proprio=True, unnormalize_actions=True, chunk_size=PREFIX, num_open_loop_steps=PREFIX, trained_with_image_aug=True, use_jpeg_compression=True, flip_images=True, num_denoising_steps_action=5, num_denoising_steps_future_state=1, num_denoising_steps_value=1)
print("Loading frozen Cosmos Policy checkpoint", flush=True)
stats = load_dataset_stats(cfg.dataset_stats_path)
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
model, _ = get_model(cfg)

def query(observation, description, occlude=False):
    model_input = prepare_observation(observation, resize_size=256, flip_images=cfg.flip_images)
    if occlude:
        model_input = deepcopy(model_input)
        image = model_input["primary_image"].copy(); height, width = image.shape[:2]
        image[height // 4:height * 3 // 4, width // 3:width * 2 // 3] = 0
        model_input["primary_image"] = image
    result = get_action(cfg, model, stats, model_input, description, num_denoising_steps_action=cfg.num_denoising_steps_action, generate_future_state_and_value_in_parallel=True)
    return model_input, result

records = []
(OUT / "method_card.json").write_text(json.dumps({"method":"custom_binary_diagnostic_baseline_v1", "not_cosmos_original_mechanism":True, "decision_rule":"MAE(predicted_future, real_after_16) > frozen_threshold => global_refresh; else continue", "threshold":THRESHOLD, "physical_rollback_claim":False, "task":{"suite":SUITE,"task_id":TASK_ID}}, indent=2), encoding="utf-8")
for episode_id in EPISODE_IDS:
    for condition in CONDITIONS:
        folder = OUT / f"seed{episode_id:02d}_{condition}"; folder.mkdir(parents=True, exist_ok=True)
        env, description = get_libero_env(task, model_family="cosmos", resolution=256)
        env.reset(); obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[episode_id])
        for _ in range(SETTLE): obs, _, _, _ = env.step(get_libero_dummy_action("cosmos"))
        executed, queries, success, terminal, first_block, pending_refresh = 0, 0, False, False, True, None
        frames, wrist_frames, decisions = [], [], []
        object_qpos = find_object_qpos(env) if condition == "object_shift" else None
        while executed < TASK_MAX_STEPS[SUITE] and not terminal:
            if pending_refresh is None:
                queries += 1
                current_input, result = query(obs, description, occlude=(condition == "visual_occlusion" and first_block))
            else:
                # The prior binary decision explicitly discarded the normal
                # continuation and supplied this fresh global plan instead.
                current_input, result = pending_refresh
                pending_refresh = None
            if condition == "object_shift" and first_block:
                env.sim.data.qpos[object_qpos:object_qpos + 3] += SHIFT_DELTA
                env.sim.forward(); env.check_success(); env._post_process(); env._update_observables(force=True)
            applied = 0
            for action in np.asarray(result["actions"], dtype=np.float32)[:PREFIX]:
                actual_action = action.copy()
                if condition == "action_noise": actual_action[:3] = np.clip(actual_action[:3] + NOISE_DELTA, -1.0, 1.0)
                obs, _, done, info = env.step(actual_action.tolist())
                executed += 1; applied += 1
                frames.append(draw_frame(np.flipud(obs["agentview_image"]), f"{condition}  t={SETTLE + executed}  block={queries}"))
                wrist_frames.append(draw_frame(np.flipud(obs["robot0_eye_in_hand_image"]), f"{condition} wrist  t={SETTLE + executed}  block={queries}"))
                # LIBERO can terminate an unsuccessful episode at its own
                # time limit.  That must end this diagnostic rollout too;
                # continuing to call env.step afterwards created misleading
                # one-action pseudo-blocks in the old D12 record.
                if done or info.get("success", False) or executed >= TASK_MAX_STEPS[SUITE]:
                    success = bool(done or info.get("success", False))
                    terminal = True
                    break
            actual = np.flipud(obs["agentview_image"]); predicted = result["future_image_predictions"]["future_image"]
            error = mae(predicted, actual); decision = "global_refresh" if error > THRESHOLD else "continue"
            save_alignment(folder, queries, current_input["primary_image"], predicted, actual, error, decision)
            entry = {"episode_id":episode_id,"condition":condition,"block":queries,"t_query":SETTLE+executed-applied,"t_real":SETTLE+executed,"actions_executed":applied,"prediction_real_mae":error,"threshold":THRESHOLD,"decision":decision,"value":float(result["value_prediction"])}
            if decision == "global_refresh" and not success and executed < TASK_MAX_STEPS[SUITE]:
                refreshed_input, refreshed = query(obs, description, occlude=False)
                pending_refresh = (refreshed_input, refreshed)
                entry["refresh_value"] = float(refreshed["value_prediction"]); entry["extra_wam_refresh_call"] = True
            else: entry["extra_wam_refresh_call"] = False
            decisions.append(entry); first_block = False
        imageio.mimsave(folder / "full_episode_primary.mp4", frames, fps=10, macro_block_size=1)
        imageio.mimsave(folder / "full_episode_wrist.mp4", wrist_frames, fps=10, macro_block_size=1)
        result_doc = {"episode_id":episode_id,"condition":condition,"success":success,"terminal":terminal,"policy_steps":executed,"normal_query_count":queries,"extra_global_refresh_calls":sum(x["extra_wam_refresh_call"] for x in decisions),"primary_video":"full_episode_primary.mp4","wrist_video":"full_episode_wrist.mp4","decisions":decisions}
        (folder / "decision_log.json").write_text(json.dumps(result_doc, indent=2), encoding="utf-8")
        records.append({k:result_doc[k] for k in result_doc if k != "decisions"})
        print(f"episode={episode_id} condition={condition} success={success} steps={executed} queries={queries}", flush=True)
with (OUT / "episode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
print(f"Done: {OUT.resolve()}", flush=True)
