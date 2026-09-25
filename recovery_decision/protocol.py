"""Public extraction of the frozen stage-coverage contract (offline GT only)."""
import argparse
import itertools
import json
from pathlib import Path

ARMS = ("continue", "reobserve4", "replan")
CONDITIONS = ("normal", "visual_occlusion", "object_shift", "action_noise")
STAGES = ("approach", "grasp_attempt")


def eligible(step):
    return step >= 4 and 16 - step % 16 >= 5


def select_stages(records):
    """Select chronological offline anchors, never an online model input.

    finger_contact is the historical trace key for gripper/hand contact,
    not a claim of force closure or successful grasp.
    """
    result = {}
    previous = -1
    for row in records:
        if row["step"] <= previous:
            raise ValueError("Trace must have strictly increasing steps")
        previous = row["step"]
        if not eligible(row["step"]):
            continue
        distance = row["distance_m"]
        if ("approach" not in result and .08 < distance <= .14
                and row["approaching"] and not row["finger_contact"]):
            result["approach"] = row
        if ("grasp_attempt" not in result and (row["finger_contact"]
                or (row["closing_command"] and distance <= .08))):
            result["grasp_attempt"] = row
    if set(result) != set(STAGES):
        raise ValueError("Missing stage: stop, do not replace seed")
    if result["approach"]["step"] >= result["grasp_attempt"]["step"]:
        raise ValueError("Invalid stage order")
    return result


def masked_at(step, onset, condition):
    return condition == "visual_occlusion" and onset <= step < onset + 8


def noisy_at(step, onset, condition):
    return condition == "action_noise" and onset <= step < onset + 4


def pair_class(executed, altered, terminal):
    if not 0 <= executed <= 16:
        raise ValueError("Expected actual executed length within one native chunk")
    if altered:
        return "invalid_action_condition_changed"
    if executed < 16:
        return "terminal_short" if terminal else "discarded_tail"
    return "aligned_endpoint"


def group_split(seed):
    if seed not in range(12):
        raise ValueError("Seed outside frozen development design")
    return "train" if seed < 8 else "validation" if seed < 10 else "development_test"


def build_manifest():
    rows = [dict(task=t, seed=s, stage=p, condition=c, arm=a, split=group_split(s))
            for t, s, p, c, a in itertools.product(
                range(2), range(12), STAGES, CONDITIONS, ARMS)]
    return {
        "protocol": "paired_stage_treatments_v2", "development_only": True,
        "online_attribution": False, "horizon": 16, "branch_delay_steps": 4,
        "minimum_remaining_actions": 1, "shift_m": [.0625, -.03, .005],
        "noise_action_delta": [.12, -.12, 0], "noise_duration_steps": 4,
        "primary_occlusion_steps": 8, "wrist_occlusion": False,
        "budget_seconds": 21600, "episode_timeout_seconds": 900,
        "disk_reserve_gib": 40, "prefix_count": 24, "branch_count": 576,
        "note": "Development seeds may have historical exposure; not final held-out evaluation.",
        "branches": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(build_manifest(), stream, indent=2)
