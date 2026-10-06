"""Synthetic certificate fixtures are tests, never real recovered labels."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wam_reranking.historical_replay_observation_contract import (
    SOURCE_SCHEMA, _certify_verified, _interval, _near_interface_rgb,
    certify_historical_chain, sha256,
)
from wam_reranking.observation_geometry import local_rgb_registration


def good_rgb():
    return dict(passed=True, mean_abs=0., p95=0., fraction_gt5=0., psnr=99.)


def good_qc():
    return dict(passed=True, sim_max_abs=0., sim_p95=0., fraction_gt_1e3=0.,
                contact_atoms_match=True, terminal_sequence_match=True)


def fixture(n=8):
    yy, xx = np.indices((64, 64))
    rgb = np.zeros((64, 64, 3), np.uint8)
    masks = {}
    for role, bounds in {"target": (4, 24), "eef": (4, 24), "anchor": (4, 24),
                         "left": (30, 44), "right": (48, 62)}.items():
        m = np.zeros((64, 64), bool); m[4:24, bounds[0]:bounds[1]] = True
        masks[role] = m
        rgb[m] = (70+80*((yy+xx) % 2))[m, None]
    arrays = {"proprio": np.zeros((n, 9)), "qpos": np.zeros((n, 2)), "qvel": np.zeros((n, 2))}
    for view in ("primary", "wrist"):
        for kind in ("actual", "rendered"): arrays[f"{view}_{kind}"] = np.stack([rgb.copy() for _ in range(n)])
        for role, mask in masks.items(): arrays[f"{view}_role_{role}"] = np.stack([mask.copy() for _ in range(n)])
    frames = []
    for s in range(n):
        t = max(0., min(s-1, 3)*.003); e = t if s <= 4 else t-(s-4)*.003
        aperture = 0. if s <= 4 else .006
        arrays["proprio"][s, :2] = aperture/2
        arrays["proprio"][s, 2:5] = [e, 0., 0.]
        identity = dict(dataset="test_synthetic_only", suite="libero90", task=46, state=10,
            candidate_id=3 if s <= 4 else 1, observed_block_id="prefix" if s <= 4 else "tail", source_step=s if s <= 4 else s-4)
        ev = dict(chain_index=s, source_identity=deepcopy(identity), views={}, actual_rgb_temporal_intervals=[])
        for view in ("primary", "wrist"):
            ev["views"][view] = dict(fresh_geometry_registered_to_actual_input=s>0, global_rgb=good_rgb(),
                roles={r:dict(rendered_pixels=int(m.sum()), local_rgb=local_rgb_registration(rgb,rgb,m)) for r,m in masks.items()},
                repeat_noise={r:dict(centroid_noise_px=.1, rgb_repeat=good_rgb()) for r in masks})
        frames.append(dict(chain_index=s, source_identity=identity, observation_evidence=ev, repeat_qc=good_qc(),
            physics=dict(target_pos_m=[t,0.,0.], eef_pos_m=[e,0.,0.], gripper_aperture_m=aperture,
                left_finger_target_contact=s<=4, right_finger_target_contact=s<=4, target_support_contact=False)))
    for end in range(2,n):
        for start in sorted({max(1,end-2),max(1,end-1)}):
            for view in ("primary", "wrist"):
                tracks = {}
                for role in masks:
                    delta = 3. if end <= 4 or role != "eef" else 6.
                    x = 7 if role not in ("left","right") else 32 if role=="left" else 50
                    points = []
                    for y in (8,10,12):
                        points.append(dict(start_pixel_xy=[float(x),float(y)], end_pixel_xy=[x+delta,float(y)],
                            projected_end_pixel_xy=[x+delta,float(y)], camera_only_end_pixel_xy=[float(x),float(y)],
                            forward_backward_error_px=0., projected_endpoint_error_px=0.))
                    tracks[role] = dict(verified=True, geometry_motion_support=True,
                        evidence_kind="actual_RGB_geometry_correspondence",count=3,tracks=points)
                frames[end]["observation_evidence"]["actual_rgb_temporal_intervals"].append(dict(start_step=start,end_step=end,view=view,
                    tracks=tracks,same_candidate_repeat_noise={r:.1 for r in masks},
                    source_start_identity=deepcopy(frames[start]["source_identity"]),
                    source_end_identity=deepcopy(frames[end]["source_identity"]),
                    no_future_identity_backfill=True,cross_block_context_preserves_distinct_candidate_ids=True))
    teacher=dict(schema=SOURCE_SCHEMA,split="train",frames=frames)
    return teacher,arrays


class HistoricalLogicTests(unittest.TestCase):
    def certify(self,t,a): return _certify_verified(t,a,{"synthetic_test_fixture_only":True})

    def test_positive_carry_and_cross_candidate_release_preserve_sources(self):
        t,a=fixture(); result=self.certify(t,a)
        self.assertTrue(result["frames"][4]["carried_sufficient_evidence"]["supervision_mask"])
        self.assertTrue(result["frames"][6]["released_sufficient_evidence"]["supervision_mask"])
        self.assertEqual(result["frames"][6]["source_identity"]["candidate_id"],1)
        self.assertEqual(result["frames"][6]["released_sufficient_evidence"]["prior_certified_carry_source"]["candidate_id"],3)
        self.assertFalse(result["training_ready"])

    def test_no_prior_observed_carry_never_certifies_release(self):
        t,a=fixture()
        for f in t["frames"][:5]: f["physics"]["target_support_contact"]=True
        result=self.certify(t,a)
        self.assertFalse(any(f["released_sufficient_evidence"]["supervision_mask"] for f in result["frames"]))

    def test_physical_carry_without_actual_motion_is_masked_not_negative(self):
        t,a=fixture()
        for f in t["frames"]:
            for interval in f["observation_evidence"]["actual_rgb_temporal_intervals"]:
                for report in interval["tracks"].values():
                    for point in report["tracks"]:
                        point["end_pixel_xy"]=point["start_pixel_xy"].copy()
                        point["projected_end_pixel_xy"]=point["start_pixel_xy"].copy()
        result=self.certify(t,a)
        atom=result["frames"][4]["carried_sufficient_evidence"]
        self.assertTrue(atom["physical_sufficient"]);self.assertIsNone(atom["value"]);self.assertFalse(atom["supervision_mask"])

    def test_perspective_different_pixel_deltas_do_not_reject_real_3d_carry(self):
        t,a=fixture()
        for interval in t["frames"][4]["observation_evidence"]["actual_rgb_temporal_intervals"]:
            for p in interval["tracks"]["eef"]["tracks"]:
                p["end_pixel_xy"][0]+=4.;p["projected_end_pixel_xy"][0]+=4.
        self.assertTrue(self.certify(t,a)["frames"][4]["carried_sufficient_evidence"]["supervision_mask"])

    def test_reported_perfect_fit_cannot_hide_actual_projection_mismatch(self):
        t,a=fixture()
        for interval in t["frames"][4]["observation_evidence"]["actual_rgb_temporal_intervals"]:
            for p in interval["tracks"]["target"]["tracks"]:p["projected_end_pixel_xy"][0]+=1.01
        self.assertFalse(self.certify(t,a)["frames"][4]["carried_sufficient_evidence"]["supervision_mask"])

    def test_physical_atom_preserved_even_without_certified_prior(self):
        t,a=fixture()
        for f in t["frames"]:
            for v in f["observation_evidence"]["views"].values():v["fresh_geometry_registered_to_actual_input"]=False
        physical={6:dict(value=True,measurement_valid=True,original_supervision_mask=False)}
        result=_certify_verified(t,a,{"test_only":True},physical)
        atom=result["frames"][6]["released_sufficient_evidence"]
        self.assertTrue(atom["raw_physical_release"]["value"]);self.assertFalse(atom["supervision_mask"])
        self.assertFalse(atom["certified_prior_required_release_eval"])

    def test_missing_repeat_noise_cannot_default_to_zero(self):
        t,a=fixture()
        for f in t["frames"]:
            for v in f["observation_evidence"]["views"].values():v["repeat_noise"]["target"]["centroid_noise_px"]=None
        self.assertFalse(any(f["carried_sufficient_evidence"]["supervision_mask"] for f in self.certify(t,a)["frames"]))

    def test_high_repeat_noise_abstains(self):
        t,a=fixture()
        for f in t["frames"]:
            for v in f["observation_evidence"]["views"].values():v["repeat_noise"]["target"]["centroid_noise_px"]=2.
            for i in f["observation_evidence"]["actual_rgb_temporal_intervals"]:i["same_candidate_repeat_noise"]["target"]=2.
        self.assertFalse(any(f["carried_sufficient_evidence"]["supervision_mask"] for f in self.certify(t,a)["frames"]))

    def test_failed_geometry_registration_is_not_strong_certification(self):
        t,a=fixture()
        for f in t["frames"]:
            for v in f["observation_evidence"]["views"].values():v["fresh_geometry_registered_to_actual_input"]=False
        self.assertFalse(any(f["carried_sufficient_evidence"]["supervision_mask"] for f in self.certify(t,a)["frames"]))

    def test_three_frame_identities_cannot_splice_unverified_views(self):
        t,a=fixture()
        t["frames"][2]["observation_evidence"]["views"]["primary"]["fresh_geometry_registered_to_actual_input"]=False
        t["frames"][4]["observation_evidence"]["views"]["wrist"]["fresh_geometry_registered_to_actual_input"]=False
        self.assertFalse(self.certify(t,a)["frames"][4]["carried_sufficient_evidence"]["supervision_mask"])

    def test_registered_cache_and_direct_interval_paths_are_identical(self):
        from wam_reranking.historical_replay_observation_contract import _view_ok
        t,a=fixture();ready={(s,v):_view_ok(t["frames"],a,s,v) for s in range(len(t["frames"])) for v in ("primary","wrist")}
        cache={}
        for interval in t["frames"][4]["observation_evidence"]["actual_rgb_temporal_intervals"]:
            plain=_interval(t["frames"],a,interval,4)
            cached=_interval(t["frames"],a,interval,4,ready,cache)
            self.assertEqual(plain["local"],cached["local"]);self.assertEqual(plain["noise"],cached["noise"])
            for role in plain["tracks"]:
                for key in ("actual","projected"):np.testing.assert_array_equal(plain["tracks"][role][key],cached["tracks"][role][key])

    def test_cached_zero_is_not_future_identity_backfilled(self):
        t,a=fixture(); self.assertFalse(self.certify(t,a)["frames"][0]["carried_sufficient_evidence"]["supervision_mask"])

    def test_changed_interval_candidate_identity_is_rejected(self):
        t,a=fixture();i=t["frames"][6]["observation_evidence"]["actual_rgb_temporal_intervals"][0]
        i["source_start_identity"]["candidate_id"]=1
        with self.assertRaises(ValueError):self.certify(t,a)

    def test_future_interval_does_not_certify_earlier_step(self):
        t,a=fixture();i=t["frames"][4]["observation_evidence"]["actual_rgb_temporal_intervals"][0]
        i["end_step"]=6
        self.assertIsNone(_interval(t["frames"],a,i,4))

    def test_one_hidden_own_finger_cannot_borrow_eef_or_other_side(self):
        t,a=fixture();a["primary_role_right"][5:]=False;a["wrist_role_right"][5:]=False
        self.assertFalse(self.certify(t,a)["frames"][6]["released_sufficient_evidence"]["supervision_mask"])

    def test_open_command_is_not_actual_opening(self):
        t,a=fixture()
        for f in t["frames"][5:]:f["physics"]["gripper_aperture_m"]=0.
        self.assertFalse(self.certify(t,a)["frames"][6]["released_sufficient_evidence"]["supervision_mask"])

    def test_release_requires_two_actual_no_contact_frames(self):
        t,a=fixture();t["frames"][5]["physics"]["left_finger_target_contact"]=True
        self.assertFalse(self.certify(t,a)["frames"][6]["released_sufficient_evidence"]["supervision_mask"])

    def test_three_carry_frames_and_frozen_physical_thresholds(self):
        for key,value in (("target_support_contact",True),("right_finger_target_contact",False)):
            t,a=fixture();t["frames"][3]["physics"][key]=value
            self.assertFalse(self.certify(t,a)["frames"][4]["carried_sufficient_evidence"]["supervision_mask"])

    def test_untextured_nearest_interface_cannot_use_far_textured_edges(self):
        w=dict(actual_boundary_coordinates=dict(target=[[0,0],[0,1]],finger=[[0,10],[0,11]]),actual_surface_gap_px=2.)
        self.assertFalse(_near_interface_rgb(w));w["actual_surface_gap_px"]=9.
        self.assertTrue(_near_interface_rgb(w))

    def test_evaluator_does_not_change_old_teacher_or_arrays(self):
        t,a=fixture();old=deepcopy(t);self.certify(t,a);self.assertEqual(t,old)

    def test_certificate_is_finite_json_serializable_no_numpy_boolean(self):
        t,a=fixture();json.dumps(self.certify(t,a),allow_nan=False)


class HistoricalLineageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.out=self.root/"observation"
        self.scenario="test_chain";self.folder=self.out/"chains"/self.scenario;self.folder.mkdir(parents=True)
        t,a=fixture(19);self.t=t;self.a=a
        blocks=[dict(block=3,candidate_id=3,directory="original/prefix"),dict(block=4,candidate_id=1,directory="original/tail",valid_length=2)]
        hashes={}
        for b in blocks:
            dest=self.root/b["directory"];(dest/"evidence/block").mkdir(parents=True)
            length=b.get("valid_length",16);start=0 if b["block"]==3 else 16
            np.savez(dest/"trajectory.npz",primary=a["primary_actual"][start+1:start+length+1],wrist=a["wrist_actual"][start+1:start+length+1])
            for v in ("primary","wrist"):np.save(dest/"evidence/block"/f"O_t_{v}.npy",a[f"{v}_actual"][start])
            if b["block"]==3:np.save(dest/"evidence/block/actual_observed_proprio.npy",a["proprio"][16])
        for p in (self.root/"original").rglob("*"):
            if p.is_file():hashes[p.relative_to(self.root).as_posix()]=sha256(p)
        self.manifest=dict(source="test_synthetic_only",complete_blocks_count_unchanged=797,same_arm_repeat_not_surrogate=True,
            source_sha256=hashes,chains=[dict(scenario=self.scenario,task=46,state=10,split="train",chain=blocks[:1],tail=blocks[-1])])
        for i,f in enumerate(t["frames"]):
            b=blocks[0] if i<=16 else blocks[1];step=i if i<=16 else i-16
            identity=dict(dataset="test_synthetic_only",suite="libero90",task=46,state=10,candidate_id=b["candidate_id"],observed_block_id=b["directory"],source_step=step)
            f["source_identity"]=identity;f["observation_evidence"]["source_identity"]=deepcopy(identity)
            f["original_replay_checks"]=dict(primary=good_rgb(),wrist=good_rgb())
            if i==16:f["original_replay_checks"]["endpoint_proprio_max_abs"]=0.
            if i>16:f["original_replay_checks"].update(endpoint_proprio_max_abs=None,original_endpoint_proprio_available=False)
            f["original_image_sources"]={}
            for v in ("primary","wrist"):
                name=f'{b["directory"]}/trajectory.npz' if step else f'{b["directory"]}/evidence/block/O_t_{v}.npy'
                f["original_image_sources"][v]=dict(path=name,sha256=hashes[name],array_name=v if step else None,array_index=step-1 if step else None,cached_reference_only=not bool(step))
        for f in t["frames"]:
            for interval in f["observation_evidence"]["actual_rgb_temporal_intervals"]:
                interval["source_start_identity"]=deepcopy(t["frames"][interval["start_step"]]["source_identity"])
                interval["source_end_identity"]=deepcopy(t["frames"][interval["end_step"]]["source_identity"])
        t.update(role="offline_supervision_only",training_ready=False,original_short_endpoint_proprio_available=False,
            original_per_step_proprio_not_fabricated=True,original_complete_blocks_count_unchanged=797,real_short_length=2)
        self.audit=dict(passed=True,all_original_RGB_and_terminal_lengths_match=True,new_mask_not_created=True,
            original_short_proprio_check=None,combined_repeat_qc=good_qc(),short_repeat_qc=good_qc(),prefix_repeat_qc=[good_qc()])
        self.write()

    def tearDown(self):self.tmp.cleanup()

    def write(self):
        np.savez(self.folder/"replay_arrays.npz",**self.a);digest=sha256(self.folder/"replay_arrays.npz")
        self.t["replay_arrays_sha256"]=digest
        for i,f in enumerate(self.t["frames"]):
            f["measured_proprio"]=dict(source="newly_measured_verified_historical_replay",array_index=i,replay_lineage_sha256=digest,
                gripper_qpos_m=self.a["proprio"][i,:2].tolist(),eef_pos_m=self.a["proprio"][i,2:5].tolist())
        for name,value in (("temporal_teacher.json",self.t),("temporal_measurements.json",{"test_only":True}),("replay_audit.json",self.audit)):
            (self.folder/name).write_text(json.dumps(value),encoding="utf-8")
        (self.out/"frozen_membership.json").write_text(json.dumps(self.manifest),encoding="utf-8")
        entries=[f'{sha256(p)}  {p.relative_to(self.out).as_posix()}\n' for p in sorted(self.out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.txt"]
        (self.out/"SHA256SUMS.txt").write_text("".join(entries),encoding="utf-8")

    def certify(self):return certify_historical_chain(self.root,self.out,self.scenario)

    def test_full_disk_lineage_verifies_new_vs_original_proprio(self):
        result=self.certify();self.assertEqual(result["lineage_audit"]["available_original_endpoint_proprio_verified"],1)
        self.assertFalse(result["lineage_audit"]["original_short_endpoint_proprio_available"])

    def test_original_file_hash_mutation_is_rejected(self):
        p=self.root/"original/prefix/trajectory.npz";p.write_bytes(b"changed")
        with self.assertRaises(ValueError):self.certify()

    def test_output_file_hash_mutation_is_rejected(self):
        (self.folder/"temporal_teacher.json").write_text("{}",encoding="utf-8")
        with self.assertRaises(ValueError):self.certify()

    def test_candidate_id_backfill_rejected_even_if_new_manifest_matches_sha(self):
        self.t["frames"][16]["source_identity"]["candidate_id"]=1;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_new_proprio_cannot_claim_original(self):
        self.write();f=self.folder/"temporal_teacher.json";t=json.loads(f.read_text());t["frames"][18]["measured_proprio"]["source"]="original_hashed_proprio"
        f.write_text(json.dumps(t),encoding="utf-8");self.rehash()
        with self.assertRaises(ValueError):self.certify()

    def rehash(self):
        (self.out/"SHA256SUMS.txt").write_text("".join(f'{sha256(p)}  {p.relative_to(self.out).as_posix()}\n' for p in sorted(self.out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.txt"),encoding="utf-8")

    def test_short_endpoint_proprio_cannot_be_fabricated(self):
        self.t["frames"][18]["original_replay_checks"]["endpoint_proprio_max_abs"]=0.;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_original_available_endpoint_proprio_threshold_not_relaxed(self):
        self.a["proprio"][16,8]=.002;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_actual_rgb_must_equal_own_hashed_original_array(self):
        self.a["primary_actual"][18,2,2]=255;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_source_step_not_replaced_by_chain_index(self):
        self.t["frames"][18]["source_identity"]["source_step"]=18;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_nonfinite_new_measurement_rejected(self):
        self.a["qvel"][18,0]=np.nan;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_failed_repeat_hard_threshold_rejected(self):
        self.audit["combined_repeat_qc"]["sim_max_abs"]=.0011;self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_missing_prefix_repeat_list_is_hard_failure(self):
        self.audit["prefix_repeat_qc"]=[];self.write()
        with self.assertRaises(ValueError):self.certify()

    def test_original_path_escape_rejected(self):
        self.manifest["source_sha256"]["../outside"]= "a"*64;self.write()
        with self.assertRaises(ValueError):self.certify()


if __name__=="__main__":unittest.main()
