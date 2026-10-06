"""Train-only shared-latent recovery residual on a frozen candidate backbone.

Observed recovery labels enter masked auxiliary heads; their gradients and
success-first pairwise gradients update the SAME projection used for ranking.
This establishes a learning connection, not a claim of attribution benefit.
Success/outcome/label records are accepted by ``fit`` only, never ``score``.
No frozen V8 model or historical deployment is changed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from . import recovery_contract
from .contracts import PREDICATES

MODEL_SCHEMA = "shared_latent_recovery_residual_v1"
EXPECTED_FEATURE_SCHEMA = "cause_dependency_recovery_features_v3_libero_command"
LATENT_WIDTH = 8
RESIDUAL_CAP = .1
RANK_LOSS_WEIGHT = 1.
AUXILIARY_LOSS_WEIGHT = .25
L2_WEIGHT = .001
LEARNING_RATE = .03


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _feature_contract(names, schema=EXPECTED_FEATURE_SCHEMA):
    names = tuple(str(name) for name in names)
    current_schema = getattr(recovery_contract, "FEATURE_SCHEMA", None)
    if schema != EXPECTED_FEATURE_SCHEMA or current_schema != schema:
        raise ValueError("recovery residual feature VERSION mismatch; legacy features cannot be promoted")
    if names != tuple(recovery_contract.FEATURE_NAMES):
        raise ValueError("recovery residual feature NAMES/order mismatch")
    return names


def _vector(value, width, name):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid finite {name} feature vector")
    return vector


def _sigmoid(value):
    clipped = np.clip(value, -60., 60.)
    return 1. / (1. + np.exp(-clipped))


def _require_train_role(row, branch):
    if row.get("split") != "train":
        raise ValueError(f"{branch} fit accepts explicitly marked TRAIN only")
    reserved = {"qc", "clean_b", "confirmation", "qualification", "test", "val"}
    for key in ("role", "dataset_role"):
        if str(row.get(key, "")).lower() in reserved:
            raise ValueError(f"{branch} QC/assessment role cannot be relabeled as train by split alone")
    if row.get("is_qc", False):
        raise ValueError(f"{branch} QC rows cannot enter recovery training")


@dataclass(frozen=True)
class RecoveryResidualModel:
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    support_min: np.ndarray
    support_max: np.ndarray
    shared_projection: np.ndarray
    ranking_weights: np.ndarray
    auxiliary_weights: np.ndarray
    auxiliary_bias: np.ndarray
    training_metadata: Mapping[str, object]
    training_curve: tuple[Mapping[str, float], ...] = ()
    residual_cap: float = RESIDUAL_CAP
    feature_schema: str = EXPECTED_FEATURE_SCHEMA
    schema: str = MODEL_SCHEMA

    def __post_init__(self):
        names = _feature_contract(self.feature_names, self.feature_schema)
        targets = tuple(str(name) for name in self.target_names)
        if not targets or len(set(targets)) != len(targets) or any(not name for name in targets):
            raise ValueError("auxiliary target names must be nonempty and unique")
        if self.schema != MODEL_SCHEMA or self.residual_cap != RESIDUAL_CAP:
            raise ValueError("recovery residual model/cap contract mismatch")
        width = len(names)
        shapes = {
            "mean": (width,), "scale": (width,),
            "support_min": (width,), "support_max": (width,),
            "shared_projection": (width, LATENT_WIDTH),
            "ranking_weights": (LATENT_WIDTH,),
            "auxiliary_weights": (LATENT_WIDTH, len(targets)),
            "auxiliary_bias": (len(targets),),
        }
        for name, shape in shapes.items():
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid recovery model {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.scale <= 0) or np.any(self.support_min > self.support_max):
            raise ValueError("invalid normalization/support bounds")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "target_names", targets)
        if not isinstance(self.training_metadata, Mapping):
            raise ValueError("missing audited training metadata")
        if self.training_metadata.get("input_roles") != ["train"]:
            raise ValueError("recovery model metadata must attest train-only inputs")
        # JSON validation also prevents hidden arrays/nonfinite metadata fields.
        _canonical_json(dict(self.training_metadata))
        _canonical_json(list(self.training_curve))

    def _latent(self, features):
        _feature_contract(self.feature_names, self.feature_schema)
        values = _vector(features, len(self.feature_names), "recovery")
        outside = bool(np.any(values < self.support_min - 1e-8)
                       or np.any(values > self.support_max + 1e-8))
        latent = np.tanh(((values - self.mean) / self.scale) @ self.shared_projection)
        return latent, outside

    def score(self, features: np.ndarray) -> float:
        """Deployable outcome-free bounded correction, not a total score."""
        latent, outside = self._latent(features)
        if outside:
            return 0.
        return float(self.residual_cap * np.tanh(latent @ self.ranking_weights))

    def score_decomposition(self, features, backbone_score):
        """Expose when support or zero weights suppress a recovery correction."""
        if not np.isfinite(backbone_score):
            raise ValueError("frozen backbone score must be finite")
        latent, outside = self._latent(features)
        rank_logit = float(latent @ self.ranking_weights)
        correction = 0. if outside else float(self.residual_cap * np.tanh(rank_logit))
        return dict(
            frozen_backbone_score=float(backbone_score),
            recovery_rank_logit=rank_logit, cause_recovery_residual=correction,
            total_score=float(backbone_score) + correction,
            residual_cap=self.residual_cap, outside_train_support=outside,
            # These are predicted silver recovery factors, not certificates.
            auxiliary_probabilities=dict(zip(self.target_names,
                _sigmoid(latent @ self.auxiliary_weights + self.auxiliary_bias).tolist())),
            auxiliary_predictions_are_not_current_belief_evidence=True,
        )

    def zero_residual_control(self):
        """Keep identical preprocessing/projection while masking the rank head."""
        from dataclasses import replace
        return replace(self, ranking_weights=np.zeros(LATENT_WIDTH))

    def to_record(self):
        record = dict(
            schema=self.schema, feature_schema=self.feature_schema,
            feature_names=list(self.feature_names), target_names=list(self.target_names),
            mean=self.mean.tolist(), scale=self.scale.tolist(),
            support_min=self.support_min.tolist(), support_max=self.support_max.tolist(),
            shared_projection=self.shared_projection.tolist(),
            ranking_weights=self.ranking_weights.tolist(),
            auxiliary_weights=self.auxiliary_weights.tolist(),
            auxiliary_bias=self.auxiliary_bias.tolist(), residual_cap=self.residual_cap,
            training_metadata=dict(self.training_metadata),
            training_curve=list(self.training_curve),
            architecture="shared_tanh8_masked_auxiliary_and_pairwise_rank",
            role="development_recovery_model_not_independent_benefit_evidence",
        )
        record["model_sha256"] = _digest(record)
        return record

    @classmethod
    def from_record(cls, record):
        payload = dict(record)
        expected_hash = payload.pop("model_sha256", None)
        if expected_hash is None or expected_hash != _digest(payload):
            raise ValueError("recovery residual serialized model hash mismatch")
        _feature_contract(record["feature_names"], record.get("feature_schema"))
        return cls(
            tuple(record["feature_names"]), tuple(record["target_names"]),
            *(np.asarray(record[name], dtype=np.float64) for name in (
                "mean", "scale", "support_min", "support_max", "shared_projection",
                "ranking_weights", "auxiliary_weights", "auxiliary_bias")),
            dict(record["training_metadata"]), tuple(record["training_curve"]),
            float(record["residual_cap"]), str(record["feature_schema"]), str(record["schema"]),
        )


def fit_recovery_residual(*, rank_rows: Sequence[Mapping],
                          auxiliary_rows: Sequence[Mapping],
                          feature_names, target_names=PREDICATES,
                          seed=0, epochs=300) -> RecoveryResidualModel:
    """Deterministic train-only pairwise + masked auxiliary multitask fit.

    ``rank_rows`` require pool_id, split='train', features, backbone_score,
    success (bool/0/1).  Optional candidate_id is retained for identity audit.
    ``auxiliary_rows`` also require split='train' explicitly, features, targets,
    masks; unknown/masked targets never participate as positive or negative.
    Labels/outcomes remain in training fingerprints, never deployment vectors.
    Both losses backpropagate through shared_projection.  No mixed candidate
    pool means no ranking supervision: the rank head remains zero.
    """
    names = _feature_contract(feature_names)
    targets = tuple(str(name) for name in target_names)
    if not targets or len(set(targets)) != len(targets) or any(not name for name in targets):
        raise ValueError("invalid auxiliary target contract")
    if not isinstance(seed, (int, np.integer)) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("training seed/epochs must be integers and epochs positive")
    if not rank_rows:
        raise ValueError("at least one formally identified train candidate row is required")
    width = len(names)
    rank_x, backbone, successes, identities, pools = [], [], [], [], {}
    normalized_rank_records = []
    for index, row in enumerate(rank_rows):
        _require_train_role(row, "rank")
        pool = row.get("pool_id")
        if not isinstance(pool, str) or not pool:
            raise ValueError("ranking row requires a source-scoped pool_id")
        features = _vector(row["features"], width, "rank")
        frozen_score = float(row["backbone_score"])
        success = row["success"]
        if not np.isfinite(frozen_score) or not isinstance(success, (bool, int, np.integer, np.bool_)) or success not in (0, 1):
            raise ValueError("invalid frozen score or success supervision")
        candidate_id = str(row.get("candidate_id", index))
        identity = (pool, candidate_id)
        if identity in identities:
            raise ValueError("duplicate ranking candidate identity")
        identities.append(identity)
        pools.setdefault(pool, []).append(index)
        rank_x.append(features); backbone.append(frozen_score); successes.append(bool(success))
        normalized_rank_records.append(dict(pool_id=pool, candidate_id=candidate_id,
            split="train", features=features.tolist(), backbone_score=frozen_score,
            success=bool(success)))
    rank_x = np.stack(rank_x)
    backbone = np.asarray(backbone, dtype=np.float64)
    positives, negatives = [], []
    for indices in pools.values():
        good = [index for index in indices if successes[index]]
        bad = [index for index in indices if not successes[index]]
        for positive in good:
            for negative in bad:
                positives.append(positive); negatives.append(negative)
    positives, negatives = np.asarray(positives, dtype=int), np.asarray(negatives, dtype=int)
    pair_count = len(positives)
    auxiliary_x, labels, masks, normalized_auxiliary_records = [], [], [], []
    auxiliary_ids = set()
    for index, row in enumerate(auxiliary_rows):
        _require_train_role(row, "auxiliary")
        if "row_id" in row:
            if not isinstance(row["row_id"], str) or not row["row_id"] or row["row_id"] in auxiliary_ids:
                raise ValueError("duplicate/invalid source-scoped auxiliary identity")
            auxiliary_ids.add(row["row_id"])
        features = _vector(row["features"], width, "auxiliary")
        values = np.asarray(row["targets"], dtype=np.float64)
        raw_mask = np.asarray(row["masks"])
        if values.shape != (len(targets),) or raw_mask.shape != values.shape:
            raise ValueError("auxiliary labels and masks must match target_names")
        if not np.isin(raw_mask, (False, True)).all():
            raise ValueError("auxiliary masks must be binary")
        mask = raw_mask.astype(bool)
        if not np.isfinite(values[mask]).all() or np.any(values[mask] < 0) or np.any(values[mask] > 1):
            raise ValueError("unmasked recovery targets must be observable probabilities in [0,1]")
        values = np.where(mask, values, 0.)
        auxiliary_x.append(features); labels.append(values); masks.append(mask)
        normalized_auxiliary_records.append(dict(
            row_id=str(row.get("row_id", f"train_auxiliary:{index}")), split="train",
            features=features.tolist(), targets=values.tolist(), masks=mask.tolist()))
    auxiliary_x = np.stack(auxiliary_x) if auxiliary_x else np.zeros((0, width))
    labels = np.stack(labels) if labels else np.zeros((0, len(targets)))
    masks = np.stack(masks) if masks else np.zeros((0, len(targets)), dtype=bool)
    mask_count = int(masks.sum())
    support = np.concatenate([rank_x, auxiliary_x], axis=0)
    mean, scale = support.mean(axis=0), support.std(axis=0)
    scale[scale < 1e-8] = 1.
    normalized_rank = (rank_x - mean) / scale
    normalized_auxiliary = (auxiliary_x - mean) / scale
    generator = np.random.default_rng(int(seed))
    initial_projection = generator.normal(0., .03, (width, LATENT_WIDTH))
    parameters = dict(
        shared_projection=initial_projection.copy(), ranking_weights=np.zeros(LATENT_WIDTH),
        # Never emit random apparent certainty for a completely masked head.
        auxiliary_weights=generator.normal(0., .03, (LATENT_WIDTH, len(targets)))
            * (masks.sum(axis=0) > 0)[None, :],
        auxiliary_bias=np.zeros(len(targets)),
    )
    adam_m = {name: np.zeros_like(value) for name, value in parameters.items()}
    adam_v = {name: np.zeros_like(value) for name, value in parameters.items()}
    curves = []

    def objective_and_gradients():
        projection = parameters["shared_projection"]
        rank_weight = parameters["ranking_weights"]
        aux_weight, aux_bias = parameters["auxiliary_weights"], parameters["auxiliary_bias"]
        rank_latent = np.tanh(normalized_rank @ projection)
        rank_logit = rank_latent @ rank_weight
        squash = np.tanh(rank_logit)
        corrections = RESIDUAL_CAP * squash
        score_gradient = np.zeros(len(rank_rows))
        rank_loss = 0.
        if pair_count:
            margins = backbone[positives] - backbone[negatives] + corrections[positives] - corrections[negatives]
            rank_loss = float(np.logaddexp(0., -margins).mean())
            pair_gradient = -_sigmoid(-margins) * RANK_LOSS_WEIGHT / pair_count
            np.add.at(score_gradient, positives, pair_gradient)
            np.add.at(score_gradient, negatives, -pair_gradient)
        logit_gradient = score_gradient * RESIDUAL_CAP * (1. - squash ** 2)
        rank_weight_gradient = rank_latent.T @ logit_gradient
        latent_gradient = logit_gradient[:, None] * rank_weight[None, :]
        projection_gradient = normalized_rank.T @ (latent_gradient * (1. - rank_latent ** 2))
        aux_weight_gradient, aux_bias_gradient = np.zeros_like(aux_weight), np.zeros_like(aux_bias)
        auxiliary_loss = 0.
        if mask_count:
            aux_latent = np.tanh(normalized_auxiliary @ projection)
            aux_logits = aux_latent @ aux_weight + aux_bias
            auxiliary_loss = float(((np.logaddexp(0., aux_logits) - labels * aux_logits) * masks).sum() / mask_count)
            aux_gradient = AUXILIARY_LOSS_WEIGHT * (_sigmoid(aux_logits) - labels) * masks / mask_count
            aux_weight_gradient = aux_latent.T @ aux_gradient
            aux_bias_gradient = aux_gradient.sum(axis=0)
            projection_gradient += normalized_auxiliary.T @ ((aux_gradient @ aux_weight.T) * (1. - aux_latent ** 2))
        gradients = dict(shared_projection=projection_gradient,
            ranking_weights=rank_weight_gradient, auxiliary_weights=aux_weight_gradient,
            auxiliary_bias=aux_bias_gradient)
        regularization = .5 * L2_WEIGHT * sum(float((value ** 2).sum()) for value in parameters.values())
        for name in gradients:
            gradients[name] += L2_WEIGHT * parameters[name]
        total = RANK_LOSS_WEIGHT * rank_loss + AUXILIARY_LOSS_WEIGHT * auxiliary_loss + regularization
        if not np.isfinite(total) or any(not np.isfinite(value).all() for value in gradients.values()):
            raise RuntimeError("nonfinite recovery residual optimization; no model emitted")
        return gradients, dict(rank_loss=rank_loss, masked_auxiliary_loss=auxiliary_loss,
            l2_loss=regularization, total_loss=total,
            max_abs_train_correction=float(np.max(np.abs(corrections))))

    _, initial_curve = objective_and_gradients()
    curves.append(dict(epoch=0, **initial_curve))
    for epoch in range(1, int(epochs) + 1):
        gradients, _ = objective_and_gradients()
        for name, value in parameters.items():
            adam_m[name] = .9 * adam_m[name] + .1 * gradients[name]
            adam_v[name] = .999 * adam_v[name] + .001 * gradients[name] ** 2
            value -= LEARNING_RATE * (adam_m[name] / (1. - .9 ** epoch)) / (
                np.sqrt(adam_v[name] / (1. - .999 ** epoch)) + 1e-8)
        if epoch % 10 == 0 or epoch == epochs:
            _, curve = objective_and_gradients()
            curves.append(dict(epoch=epoch, **curve))
    fingerprint = dict(rank_input_sha256=_digest(normalized_rank_records),
        auxiliary_input_sha256=_digest(normalized_auxiliary_records),
        feature_names_sha256=_digest(list(names)), target_names_sha256=_digest(list(targets)))
    metadata = dict(
        input_roles=["train"], seed=int(seed), epochs=int(epochs),
        rank_rows=len(rank_rows), auxiliary_rows=len(auxiliary_rows),
        pool_count=len(pools), success_first_pair_count=pair_count,
        observed_auxiliary_targets=mask_count,
        masks_by_target=dict(zip(targets, masks.sum(axis=0).astype(int).tolist())),
        ranking_supervision_available=bool(pair_count),
        shared_projection_receives_both_losses=bool(pair_count and mask_count),
        rank_head_norm=float(np.linalg.norm(parameters["ranking_weights"])),
        shared_projection_change_norm=float(np.linalg.norm(parameters["shared_projection"] - initial_projection)),
        initialization_rank_head_zero=True, forced_nonzero_residual=False,
        rank_loss_weight=RANK_LOSS_WEIGHT, auxiliary_loss_weight=AUXILIARY_LOSS_WEIGHT,
        l2_weight=L2_WEIGHT, residual_cap=RESIDUAL_CAP, learning_rate=LEARNING_RATE,
        optimizer="deterministic_full_batch_adam_0.9_0.999_1e-8",
        training_fingerprints=fingerprint,
        success_used_only_as_training_supervision=True,
        auxiliary_predictions_do_not_restore_belief=True,
        no_hard_gate_override=True,
        independent_coupling_benefit_established=False,
    )
    return RecoveryResidualModel(names, targets, mean, scale,
        support.min(axis=0), support.max(axis=0), parameters["shared_projection"],
        parameters["ranking_weights"], parameters["auxiliary_weights"],
        parameters["auxiliary_bias"], metadata, tuple(curves))


# Compact API requested by the versioned V9 training launcher.
fit = fit_recovery_residual
