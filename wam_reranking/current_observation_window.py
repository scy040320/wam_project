"""Source-bound actual history at a PRE decision, never a physical witness.

This additive interface makes measured temporal inputs available for a later,
independently validated observer. It does not load a detector, derive grasp or
lift, admit a fact, change a gate, or treat past commands as measured motion.
All frames and executed commands belong to one recipient's actual history.
Candidate predictions and offline supervision windows are not accepted.

Matching content hashes proves content/identity binding, not that an upstream
capture producer is trustworthy or that an object is held. Those remain
separate audits; the existing fact registry and selector are unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np

from .current_observation_evidence import (
    ActualObservationInputs, HASH_SCHEME, SOURCE_ROLE, SourceSnapshot,
    array_sha256,
)


SCHEMA = "actual_PRE_observation_window_v1"


def _sha256(value: str, name: str) -> None:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(name + " must be lowercase SHA256")


def _step(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be an exact nonnegative policy step")


def _commands(value: np.ndarray, expected_shape: tuple[int, int], name: str,
              expected_sha256: str) -> np.ndarray:
    _sha256(expected_sha256, name + "_sha256")
    array = np.asarray(value)
    if (array.shape != expected_shape or array.dtype.kind not in "fiu" or
            not np.isfinite(array).all()):
        raise ValueError(name + " must contain finite measured N x 7 commands")
    if array_sha256(array) != expected_sha256:
        raise ValueError(name + " content differs from declared SHA256")
    return array


@dataclass(frozen=True)
class ActualObservationWindow:
    """Literal actual frames and their already-executed transitions.

    ``command_policy_steps[i]`` is the state at which requested/applied row i
    was executed, taking frame i to frame i+1. It is not a future candidate
    command or an estimated robot trajectory. Every frame includes measured
    dual-view RGB and a nine-component proprio vector; missing history is not
    padded or synthesized from commands.

    The final frame equals the independently supplied ``recipient_snapshot``.
    Its policy step cannot exceed the prerequisite's requested use step. No
    image/proprio snapshot from another pool/entity, offline teacher window,
    earlier cached endpoint or later candidate execution can be substituted.
    """
    recipient_snapshot: SourceSnapshot
    use_policy_step: int
    frames: tuple[ActualObservationInputs, ...]
    command_policy_steps: tuple[int, ...]
    requested: np.ndarray
    applied: np.ndarray
    requested_sha256: str
    applied_sha256: str
    source_role: str = SOURCE_ROLE
    predecision: bool = True
    hash_scheme: str = HASH_SCHEME

    def __post_init__(self):
        if not isinstance(self.recipient_snapshot, SourceSnapshot):
            raise ValueError("An independently pinned recipient SourceSnapshot is required")
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "command_policy_steps", tuple(self.command_policy_steps))
        self.validate()
        for name in ("requested", "applied"):
            array = np.asarray(getattr(self, name)).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def validate(self) -> None:
        """Recheck every actual byte, identity and timestamp at point of use."""
        if (self.source_role != SOURCE_ROLE or self.predecision is not True or
                self.hash_scheme != HASH_SCHEME):
            raise ValueError("Only runtime actual PRE windows are admissible; not offline Y")
        _step(self.use_policy_step, "use_policy_step")
        if len(self.frames) < 2:
            raise ValueError("At least two literal actual frames are required; no padding")
        if any(not isinstance(frame, ActualObservationInputs) for frame in self.frames):
            raise ValueError("Every frame must be typed ActualObservationInputs")
        recipient = self.recipient_snapshot
        if self.frames[-1].snapshot != recipient:
            raise ValueError("Final actual frame differs from independently pinned recipient snapshot")
        if recipient.policy_step > self.use_policy_step:
            raise ValueError("Actual history ends after the prerequisite use time")
        steps = tuple(frame.snapshot.policy_step for frame in self.frames)
        if steps != tuple(range(steps[0], steps[0] + len(steps))):
            raise ValueError("Actual frame policy steps must be literal and continuous")
        if self.command_policy_steps != steps[:-1]:
            raise ValueError("Executed command times must match their actual frame transitions")
        for value in self.command_policy_steps:
            _step(value, "command_policy_step")
        expected_shape = (len(self.frames) - 1, 7)
        _commands(self.requested, expected_shape, "requested", self.requested_sha256)
        _commands(self.applied, expected_shape, "applied", self.applied_sha256)
        primary_shape = self.frames[0].primary.shape
        wrist_shape = None if self.frames[0].wrist is None else self.frames[0].wrist.shape
        previous_block = None
        for frame in self.frames:
            snapshot = frame.snapshot
            if ((snapshot.pool_id, snapshot.entity_key, snapshot.anchor_key,
                 snapshot.extractor_version, snapshot.source_role, snapshot.hash_scheme) !=
                    (recipient.pool_id, recipient.entity_key, recipient.anchor_key,
                     recipient.extractor_version, SOURCE_ROLE, HASH_SCHEME)):
                raise ValueError("Window pool/entity/anchor/extractor/source identity mismatch")
            if snapshot.predecision is not True or snapshot.policy_step > self.use_policy_step:
                raise ValueError("A frame is not actual PRE evidence at the requested use time")
            if (snapshot.block_index > recipient.block_index or
                    (previous_block is not None and snapshot.block_index < previous_block)):
                raise ValueError("Actual block identities cannot move backwards or exceed recipient")
            previous_block = snapshot.block_index
            if frame.wrist is None or frame.proprio is None:
                raise ValueError("Measured dual RGB and proprio history required; commands cannot fill missing state")
            if (frame.primary.shape != primary_shape or frame.wrist.shape != wrist_shape or
                    frame.proprio.shape != (9,)):
                raise ValueError("Within-view shapes and measured nine-component proprio must align")
            for name in ("primary", "wrist", "proprio"):
                array = getattr(frame, name)
                expected = getattr(snapshot, name + "_sha256")
                if array_sha256(array) != expected:
                    raise ValueError("Actual frame content changed after snapshot admission: " + name)

    def validate_for(self, recipient_snapshot: SourceSnapshot, use_policy_step: int) -> None:
        """Reject borrowing a valid history window for a different PRE recipient."""
        self.validate()
        _step(use_policy_step, "use_policy_step")
        if recipient_snapshot != self.recipient_snapshot or use_policy_step != self.use_policy_step:
            raise ValueError("Window is not bound to this exact recipient and use time")

    @property
    def identity_sha256(self) -> str:
        """Hash the explicit identities and canonical-array manifests, not labels."""
        self.validate()
        payload = self._manifest()
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _manifest(self) -> dict:
        return dict(schema=SCHEMA, recipient_snapshot=asdict(self.recipient_snapshot),
            use_policy_step=self.use_policy_step,
            frames=[asdict(frame.snapshot) for frame in self.frames],
            command_policy_steps=list(self.command_policy_steps),
            requested_sha256=self.requested_sha256, applied_sha256=self.applied_sha256,
            source_role=self.source_role, predecision=self.predecision, hash_scheme=self.hash_scheme)

    def audit_record(self) -> dict:
        """Structural evidence only; zero physical facts or detector claims."""
        self.validate()
        return self._manifest() | dict(
            window_sha256=self.identity_sha256, actual_frames=len(self.frames),
            executed_transitions=len(self.command_policy_steps),
            candidate_invariant=True, physical_state_certified=False,
            predicate_facts=[], confidence_calibrated=False,
            binding_is_physical_certificate=False,
            upstream_capture_provenance_independently_certified=False,
            command_values_are_not_measured_robot_motion=True,
            registry_or_selector_modified=False,
        )


def make_actual_observation_window(*, recipient_snapshot: SourceSnapshot,
                                  use_policy_step: int,
                                  actual_frames: tuple[ActualObservationInputs, ...],
                                  actual_requested: np.ndarray,
                                  actual_applied: np.ndarray) -> ActualObservationWindow:
    """Build an actual-only window; no forecast, Y, stage or outcome arguments."""
    frames = tuple(actual_frames)
    if not frames or any(not isinstance(frame, ActualObservationInputs) for frame in frames):
        raise ValueError("Typed literal actual frames are required")
    return ActualObservationWindow(
        recipient_snapshot=recipient_snapshot, use_policy_step=use_policy_step,
        frames=frames,
        command_policy_steps=tuple(frame.snapshot.policy_step for frame in frames[:-1]),
        requested=actual_requested, applied=actual_applied,
        requested_sha256=array_sha256(actual_requested),
        applied_sha256=array_sha256(actual_applied),
    )
