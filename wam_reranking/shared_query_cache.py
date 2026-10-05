"""Outcome-free immutable query cache for paired policy executions.

One worker owns each task. Identical deployment request hashes share the same
generated actions, predicted images and value across arms. This removes query
numeric drift from comparisons; logical query budgets are still billed to EACH
arm. No success, simulator state or interventions enter cache keys or payloads.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SharedQueryCache:
    def __init__(self, root, model_signature):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_signature = str(model_signature)

    def request(self, observation_sha256, seed, task):
        record = {"observation_sha256": str(observation_sha256), "seed": int(seed),
                  "task_language_binding": str(task), "model_signature": self.model_signature,
                  "schema": "shared_deployment_query_v1"}
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        return digest, record

    def get_or_generate(self, observation_sha256, seed, task, generate):
        key, request = self.request(observation_sha256, seed, task)
        folder = self.root / key
        hit = folder.exists()
        if not hit:
            result, seconds = generate()
            temp = self.root / (key + ".pending")
            temp.mkdir(exist_ok=False)
            arrays = {"actions": result["actions"],
                      "predicted_primary": result["future_image_predictions"]["future_image"],
                      "predicted_wrist": result["future_image_predictions"]["future_wrist_image"]}
            for name, value in arrays.items():
                array = np.asarray(value)
                if array.dtype.hasobject or not np.isfinite(array).all():
                    raise ValueError("invalid query array")
                np.save(temp / (name + ".npy"), array, allow_pickle=False)
            value = float(result["value_prediction"])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid query value")
            metadata = {"request": request, "value": value, "generation_seconds": float(seconds),
                        "sha256": {p.name: file_sha(p) for p in temp.glob("*.npy")}}
            (temp / "query.json").write_text(json.dumps(metadata, sort_keys=True))
            temp.rename(folder)
        metadata = json.loads((folder / "query.json").read_text())
        if metadata["request"] != request or set(metadata["sha256"]) != {
                "actions.npy", "predicted_primary.npy", "predicted_wrist.npy"}:
            raise RuntimeError("query cache identity drift")
        for name, sha in metadata["sha256"].items():
            if file_sha(folder / name) != sha:
                raise RuntimeError("query cache payload changed")
        result = {"actions": np.load(folder / "actions.npy", allow_pickle=False),
                  "future_image_predictions": {
                      "future_image": np.load(folder / "predicted_primary.npy", allow_pickle=False),
                      "future_wrist_image": np.load(folder / "predicted_wrist.npy", allow_pickle=False)},
                  "value_prediction": metadata["value"]}
        return result, metadata["generation_seconds"], key, hit
