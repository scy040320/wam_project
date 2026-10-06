"""Read-only full-hand runtime metadata for private offline supervision.

This module never constructs an environment, calls step/render/reset, creates X,
or certifies held=False. Complete simulator no-contact is a MEASUREMENT, not a
visual certificate. Unknown constraints and missing fields remain unknown.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

SCHEMA = "full_hand_contact_private_metadata_v1"
EQ_TYPES = {0: "connect", 1: "weld", 2: "joint", 3: "tendon", 4: "flex", 5: "distance"}


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer ID")
    return int(value)


def _array(obj, name, shape=None, integer=False):
    value = getattr(obj, name, None)
    if value is None:
        return None
    result = np.asarray(value)
    if shape is not None and result.shape != shape:
        raise ValueError(name + " shape mismatch")
    if not (np.issubdtype(result.dtype, np.number) or np.issubdtype(result.dtype, np.bool_)) or not np.isfinite(result).all():
        raise ValueError(name + " must be finite numeric metadata")
    if integer and not np.equal(result, np.floor(result)).all():
        raise ValueError(name + " must contain integer IDs")
    return result.astype(int) if integer else result


def _name(model, kind, index):
    try:
        value = getattr(model, kind + "_id2name")(index)
    except (AttributeError, ValueError, KeyError):
        value = None
    return value if isinstance(value, str) else None


def _body_id(model, value, nbody):
    if isinstance(value, str):
        value = model.body_name2id(value)
    result = _integer(value, "body")
    if not 0 <= result < nbody:
        raise ValueError("body ID is outside the exact model")
    return result


def _ancestors(parents, body):
    result, seen = [], set()
    while True:
        if body in seen:
            raise ValueError("cyclic body ancestry")
        seen.add(body)
        result.append(body)
        if body == 0:
            return result
        body = int(parents[body])


def _subtree(parents, root):
    return [body for body in range(len(parents)) if root in _ancestors(parents, body)]


def _select_arm(value, arm, field):
    if not isinstance(value, Mapping):
        return value
    if arm is None:
        if len(value) != 1:
            raise ValueError(field + " requires explicit gripper_arm for a multi-arm robot")
        return next(iter(value.values()))
    if arm not in value:
        raise ValueError("unknown arm in " + field)
    return value[arm]


def capture_hand_contact_metadata(env, target_body, *, robot_index=0,
                                  gripper_arm=None, source_identity=None):
    """Read an ALREADY EXISTING env; output private-Y metadata only.

    ``target_body`` is an exact runtime body ID/name, not a task ID or inferred
    target. Caller hashes/binds this output to the source frame. Names are
    optional display metadata: contact/body membership always uses integer IDs.
    Missing current eq_active never falls back to model.eq_active0.
    Full hand = gripper subtree + explicitly bound robot mount body's own geoms;
    all-robot contact is also reported, preventing a forearm contact exclusion
    from being silently passed as full manipulator separation.
    """
    base = env
    for _ in range(8):
        if hasattr(base, "sim") and hasattr(base, "robots"):
            break
        inner = getattr(base, "env", None)
        if inner is None or inner is base:
            raise ValueError("existing env with sim and robots is required")
        base = inner
    else:
        raise ValueError("environment wrapper depth exceeded")
    model, data = base.sim.model, base.sim.data
    nbody, ngeom = _integer(model.nbody, "nbody"), _integer(model.ngeom, "ngeom")
    parents = _array(model, "body_parentid", (nbody,), integer=True)
    geom_bodies = _array(model, "geom_bodyid", (ngeom,), integer=True)
    if parents is None or geom_bodies is None:
        raise ValueError("exact body and geom ID tables are mandatory")
    if nbody < 1 or np.any(parents < 0) or np.any(parents >= nbody) or parents[0] != 0:
        raise ValueError("invalid body parent table")
    if np.any(geom_bodies < 0) or np.any(geom_bodies >= nbody):
        raise ValueError("invalid geom body IDs")
    for body in range(nbody):
        _ancestors(parents, body)
    target_id = _body_id(model, target_body, nbody)
    if target_id == 0:
        raise ValueError("world is not a uniquely bound target instance")
    index = _integer(robot_index, "robot_index")
    if not 0 <= index < len(base.robots):
        raise ValueError("robot_index outside existing environment")
    robot = base.robots[index]
    gripper = _select_arm(robot.gripper, gripper_arm, "gripper")
    hand_root = _body_id(model, gripper.root_body, nbody)
    robot_model = robot.robot_model
    robot_root = _body_id(model, robot_model.root_body, nbody)
    robot_ids = set(_subtree(parents, robot_root))
    all_robot_ids = set()
    for existing_robot in base.robots:
        all_robot_ids.update(_subtree(parents, _body_id(model, existing_robot.robot_model.root_body, nbody)))
    hand_ids = set(_subtree(parents, hand_root))
    missing, mount_id = [], None
    mount_name = _select_arm(getattr(robot_model, "eef_name", None), gripper_arm, "eef_name")
    if mount_name is None:
        missing.append("explicit_robot_hand_mount_not_recorded")
    else:
        mount_id = _body_id(model, mount_name, nbody)
        if mount_id not in _ancestors(parents, hand_root):
            raise ValueError("gripper root is not under the explicit robot hand mount")
        hand_ids.add(mount_id)
    if hand_root not in robot_ids:
        raise ValueError("gripper root is outside selected robot subtree")
    target_ids = set(_subtree(parents, target_id))
    contype = _array(model, "geom_contype", (ngeom,), integer=True)
    conaffinity = _array(model, "geom_conaffinity", (ngeom,), integer=True)
    group = _array(model, "geom_group", (ngeom,), integer=True)
    if contype is None or conaffinity is None:
        missing.append("collision_affinity_metadata_missing")
    if group is None:
        missing.append("render_geom_group_missing")
    declared_contact = set(getattr(gripper, "contact_geoms", ()) or ())
    declared_visual = set(getattr(gripper, "visual_geoms", ()) or ())
    geom_table = []
    for gid in range(ngeom):
        name = _name(model, "geom", gid)
        enabled = None if contype is None or conaffinity is None else bool(contype[gid] or conaffinity[gid])
        geom_table.append(dict(geom_id=gid, geom_name=name, body_id=int(geom_bodies[gid]),
            body_name=_name(model, "body", int(geom_bodies[gid])),
            contype=None if contype is None else int(contype[gid]),
            conaffinity=None if conaffinity is None else int(conaffinity[gid]),
            group=None if group is None else int(group[gid]), collision_enabled=enabled,
            gripper_declared_contact=name in declared_contact,
            gripper_declared_visual=name in declared_visual))
    hand_geoms = [row["geom_id"] for row in geom_table if row["body_id"] in hand_ids]
    target_geoms = [row["geom_id"] for row in geom_table if row["body_id"] in target_ids]
    if not hand_geoms or not target_geoms:
        missing.append("empty_fullhand_or_target_geometry")
    ncon = _integer(data.ncon, "ncon")
    if ncon < 0 or len(data.contact) < ncon:
        raise ValueError("incomplete runtime contact array")
    contacts = []
    for i in range(ncon):
        c = data.contact[i]
        ids = [_integer(c.geom1, "contact.geom1"), _integer(c.geom2, "contact.geom2")]
        if any(not 0 <= gid < ngeom for gid in ids):
            raise ValueError("contact references invalid or unresolvable geom ID")
        bodies = [int(geom_bodies[gid]) for gid in ids]
        is_target = [body in target_ids for body in bodies]
        touches_hand = any(is_target[j] and bodies[1-j] in hand_ids for j in (0, 1))
        touches_robot = any(is_target[j] and bodies[1-j] in all_robot_ids for j in (0, 1))
        contacts.append(dict(contact_index=i, geom_ids=ids, body_ids=bodies,
            geom_names=[_name(model, "geom", gid) for gid in ids],
            target_fullhand_contact=touches_hand, target_any_robot_contact=touches_robot))
    neq = getattr(model, "neq", None)
    eq_rows, unsupported, target_constraints = [], [], []
    if neq is None:
        missing.append("equality_count_missing")
    else:
        neq = _integer(neq, "neq")
        if neq < 0:
            raise ValueError("negative equality count")
        fields = {key: _array(model, key, (neq,), integer=True)
                  for key in ("eq_type", "eq_objtype", "eq_obj1id", "eq_obj2id")}
        active = _array(data, "eq_active", (neq,), integer=True)
        eq_data = _array(model, "eq_data")
        if neq and (any(value is None for value in fields.values()) or active is None or eq_data is None):
            missing.append("current_equality_fields_incomplete")
        elif neq:
            if eq_data.ndim != 2 or len(eq_data) != neq:
                raise ValueError("eq_data shape mismatch")
            if not np.isin(active, (0, 1)).all():
                raise ValueError("eq_active is not a current boolean activation array")
            joint_body = _array(model, "jnt_bodyid", integer=True)
            site_body = _array(model, "site_bodyid", integer=True)
            for eid in range(neq):
                typ = int(fields["eq_type"][eid])
                objtype = int(fields["eq_objtype"][eid])
                objs = [int(fields[key][eid]) for key in ("eq_obj1id", "eq_obj2id")]
                bodies, reason = None, None
                if typ in (0, 1) and objtype == 1 and all(0 <= obj < nbody for obj in objs):
                    bodies = objs
                elif typ in (0, 1) and objtype == 6 and site_body is not None and all(0 <= obj < len(site_body) for obj in objs):
                    bodies = [int(site_body[obj]) for obj in objs]
                elif typ == 2 and objtype == 3 and joint_body is not None and 0 <= objs[0] < len(joint_body) and (objs[1] == -1 or 0 <= objs[1] < len(joint_body)):
                    bodies = [int(joint_body[obj]) for obj in objs if obj != -1]
                else:
                    reason = "constraint_type_or_object_binding_not_interpreted"
                is_active = bool(active[eid])
                if is_active and bodies is None:
                    unsupported.append(eid)
                target_bound = None if bodies is None else any(body in target_ids for body in bodies)
                if is_active and target_bound:
                    target_constraints.append(eid)
                eq_rows.append(dict(eq_id=eid, eq_type=typ, eq_type_name=EQ_TYPES.get(typ, "unrecognized"), eq_object_type=objtype,
                    obj_ids=objs, body_ids=bodies, active=is_active, target_bound=target_bound,
                    interpretation_reason=reason, parameters=eq_data[eid].tolist()))
    if unsupported:
        missing.append("active_constraint_interpretation_unknown")
    mocap = _array(model, "body_mocapid", (nbody,), integer=True)
    if mocap is None:
        missing.append("target_mocap_exclusion_unknown")
    target_mocap = None if mocap is None else [body for body in sorted(target_ids) if mocap[body] >= 0]
    target_in_robot = bool(target_ids & all_robot_ids)
    attached = target_in_robot or bool(target_mocap) or bool(target_constraints)
    attachment_status = "present" if attached else ("unknown" if any(x.startswith("equality") or "equality" in x or "constraint" in x or "mocap" in x for x in missing) else "verified_absent")
    # Absence of equality is not proof against actuators/plugins/external forces.
    # Capture exact passive-entity evidence when available; otherwise unknown.
    passive_unknown, target_actuators = [], []
    nu = getattr(model, "nu", None)
    jbody = _array(model, "jnt_bodyid", integer=True)
    jtype = _array(model, "jnt_type", integer=True)
    jadr = _array(model, "body_jntadr", (nbody,), integer=True)
    jnum = _array(model, "body_jntnum", (nbody,), integer=True)
    target_joints = None
    if any(field is None for field in (jbody, jtype, jadr, jnum)):
        passive_unknown.append("target_joint_kind_binding_unknown")
    else:
        if jbody.shape != jtype.shape or jbody.ndim != 1 or np.any(jnum < 0):
            raise ValueError("joint kind/body mapping mismatch")
        target_joints = []
        for body in sorted(target_ids):
            if jnum[body] == 0:
                continue
            first, count = int(jadr[body]), int(jnum[body])
            if first < 0 or first + count > len(jbody):
                raise ValueError("target joint address outside model")
            for jid in range(first, first + count):
                if jbody[jid] != body:
                    raise ValueError("target joint body/address mismatch")
                target_joints.append(dict(joint_id=jid, body_id=body, joint_type=int(jtype[jid])))
    if nu is None:
        passive_unknown.append("actuator_count_missing")
    else:
        nu = _integer(nu, "nu")
        if nu < 0:
            raise ValueError("negative actuator count")
        trntype, trnid = _array(model, "actuator_trntype", (nu,), integer=True), _array(model, "actuator_trnid", (nu, 2), integer=True)
        if nu and (trntype is None or trnid is None):
            passive_unknown.append("actuator_transmission_missing")
        elif nu:
            for aid in range(nu):
                typ, oid = int(trntype[aid]), int(trnid[aid, 0])
                if typ in (0, 1) and jbody is not None and 0 <= oid < len(jbody):
                    if int(jbody[oid]) in target_ids:
                        target_actuators.append(aid)
                else:
                    passive_unknown.append("actuator_transmission_not_interpreted:" + str(aid))
    nplugin = getattr(model, "nplugin", None)
    if nplugin is None or nplugin != 0:
        passive_unknown.append("plugin_effects_not_excluded")
    external = _array(data, "xfrc_applied", (nbody, 6))
    qforce = _array(data, "qfrc_applied")
    dof_body = _array(model, "dof_bodyid", integer=True)
    if external is None or qforce is None or dof_body is None:
        passive_unknown.append("external_force_exclusion_incomplete")
        external_target = None
    else:
        if qforce.shape != dof_body.shape:
            raise ValueError("qfrc_applied/dof_bodyid mismatch")
        external_target = bool(np.any(external[sorted(target_ids)] != 0) or np.any(qforce[np.isin(dof_body, list(target_ids))] != 0))
    passive_status = "present" if target_actuators or external_target else ("unknown" if passive_unknown else "verified_absent")
    return dict(schema=SCHEMA, role="offline_supervision_only", deployment_allowed=False,
        source_identity=dict(source_identity or {}), source_binding_requires_caller_hash_audit=True,
        target=dict(root_body_id=target_id, root_body_name=_name(model, "body", target_id),
                    body_ids=sorted(target_ids), geom_ids=target_geoms, joint_bindings=target_joints,
                    exactly_one_root_freejoint=None if target_joints is None else
                    (len(target_joints) == 1 and target_joints[0]["body_id"] == target_id and target_joints[0]["joint_type"] == 0)),
        full_hand=dict(gripper_root_body_id=hand_root, gripper_root_body_name=_name(model, "body", hand_root),
            mount_body_id=mount_id, body_ids=sorted(hand_ids), geom_ids=hand_geoms,
            collision_geom_ids=[gid for gid in hand_geoms if geom_table[gid]["collision_enabled"] is True],
            visual_geom_ids=[gid for gid in hand_geoms if geom_table[gid]["gripper_declared_visual"] or geom_table[gid]["group"] == 1],
            declared_visual_geom_ids=[gid for gid in hand_geoms if geom_table[gid]["gripper_declared_visual"]],
            all_subtree_geoms_retained_for_visual_adapter=True,
            important_geoms_is_not_a_full_hand_definition=True),
        robot=dict(root_body_id=robot_root, body_ids=sorted(robot_ids), all_robot_body_ids=sorted(all_robot_ids)),
        body_parent_ids=parents.tolist(), geom_id_table=geom_table,
        contacts=dict(ncon=ncon, pairs=contacts, geom_ids_complete=True,
            target_fullhand_contact=any(row["target_fullhand_contact"] for row in contacts),
            target_any_robot_contact=any(row["target_any_robot_contact"] for row in contacts)),
        equality=dict(neq=neq, rows=eq_rows, current_state_not_initial_eq_active0=True,
            unsupported_active_constraint_ids=unsupported, active_target_constraint_ids=target_constraints),
        attachment=dict(status=attachment_status, target_in_robot_subtree=target_in_robot,
            target_robot_ancestor_ids=sorted(set(_ancestors(parents, target_id)) & all_robot_ids),
            target_mocap_body_ids=target_mocap, body_mocap_ids=None if mocap is None else mocap.tolist(),
            target_active_constraint_ids=target_constraints),
        other_drives=dict(status=passive_status, target_actuator_ids=target_actuators,
            target_external_force_nonzero=external_target,
            target_xfrc_applied=None if external is None else
                [dict(body_id=body, values=external[body].tolist()) for body in sorted(target_ids)],
            unknown_reasons=passive_unknown),
        metadata_complete=not missing and not passive_unknown, unknown_reasons=missing + passive_unknown,
        supervision_mask=False, held_negative_certificate_issued=False,
        release_certificate_issued=False, whole_training_gate_pass=False,
        limitations=["Contact samples are control-boundary measurements, not substep continuity.",
            "No-contact and attachment metadata alone are not an actual-RGB full-hand separation certificate.",
            "Visual mappings require the separate frozen actual-RGB role/identity/noise adapter.",
            "Old carried True/None atoms and all old masks are unchanged."])
