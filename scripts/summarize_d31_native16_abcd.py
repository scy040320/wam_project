#!/usr/bin/env python3
"""Summarize D31 without changing any frozen decision rule."""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

def mean(xs): return statistics.fmean(xs) if xs else 0.0
def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--root", required=True, type=Path); root=ap.parse_args().root
    with (root/"manifest.csv").open(newline="",encoding="utf-8") as f: manifest=list(csv.DictReader(f))
    rows=[]; missing=[]
    for m in manifest:
        out=root/"runs"/f'{m["cell_id"]}_{m["approach"]}'; p=out/"online_decision_log.json"
        try:
            d=json.loads(p.read_text(encoding="utf-8")); decisions=d.get("decisions",[])
            intervention=next((x for x in decisions if x.get("intervention_injected")),{})
            rows.append({**m,"success":int(bool(d.get("success"))),"safety_stopped":int(bool(d.get("safety_stopped"))),
                "wam_calls":d.get("wam_calls",0),"global_refreshes":d.get("global_refreshes",0),
                "guarded_reobserves":d.get("guarded_reobserves",0),"invalidated_node_total":d.get("invalidated_node_total",0),
                "environment_steps":d.get("environment_steps",0),"mean_wam_latency_ms":d.get("mean_wam_latency_ms",0.0),
                "p95_wam_latency_ms":d.get("p95_wam_latency_ms",0.0),"intervention_cause":intervention.get("cause",""),
                "intervention_recovery":intervention.get("recovery_action",""),"intervention_guard_level":intervention.get("guard_level",""),
                "decision_count":len(decisions)})
        except (OSError,ValueError) as e: missing.append({"job":m,"error":repr(e)})
    fields=list(rows[0]) if rows else []
    with (root/"episode_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    groups=defaultdict(list)
    for r in rows: groups[(r["analysis_set"],r["approach"],r["condition"])].append(r)
    aggregates=[]
    for (aset,approach,condition),rs in sorted(groups.items()):
        aggregates.append({"analysis_set":aset,"approach":approach,"condition":condition,"n":len(rs),
            "success_rate":mean([r["success"] for r in rs]),"safety_stop_rate":mean([r["safety_stopped"] for r in rs]),
            "mean_wam_calls":mean([float(r["wam_calls"]) for r in rs]),"mean_global_refreshes":mean([float(r["global_refreshes"]) for r in rs]),
            "mean_guarded_reobserves":mean([float(r["guarded_reobserves"]) for r in rs]),
            "mean_invalidated_nodes":mean([float(r["invalidated_node_total"]) for r in rs]),
            "mean_environment_steps":mean([float(r["environment_steps"]) for r in rs]),
            "mean_wam_latency_ms":mean([float(r["mean_wam_latency_ms"]) for r in rs])})
    afields=list(aggregates[0]) if aggregates else []
    with (root/"aggregate_by_method_condition.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=afields); w.writeheader(); w.writerows(aggregates)
    summary={"integrity":{"expected":len(manifest),"loaded":len(rows),"missing_or_invalid":missing},
        "primary_jobs":sum(r["analysis_set"]=="primary_heldout" for r in rows),
        "supplemental_jobs":sum(r["analysis_set"]!="primary_heldout" for r in rows),"aggregate":aggregates,
        "warning":"Seeds 30-31 overlap validation and are excluded from primary inference. D31 is preliminary and must not retune thresholds."}
    (root/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# D31 native-16 paired A/B/C/D summary","",f"- Integrity: {len(rows)}/{len(manifest)} readable episodes",
        f"- Primary held-out jobs: {summary['primary_jobs']} (seeds 32-49)",f"- Supplemental jobs: {summary['supplemental_jobs']} (seeds 30-31; excluded)",
        "- Frozen D30 V6 rules were not adapted during D31.","","| set | method | condition | n | success | safety stop | global refresh | WAM calls | invalidated nodes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for a in aggregates:
        lines.append(f"| {a['analysis_set']} | {a['approach']} | {a['condition']} | {a['n']} | {a['success_rate']:.3f} | {a['safety_stop_rate']:.3f} | {a['mean_global_refreshes']:.2f} | {a['mean_wam_calls']:.2f} | {a['mean_invalidated_nodes']:.2f} |")
    (root/"D31_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0 if not missing else 2
if __name__=="__main__": raise SystemExit(main())
