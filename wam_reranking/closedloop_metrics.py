"""Outcome-only metrics; never imported by the deployable selection path."""
import statistics


def describe(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def total_calls(outcome):
    return outcome['common_prefix_wam_calls'] + outcome['wam_calls_after_fork']


def execution_cost_summary(rows, methods, *, full_method='full_repaired'):
    """Report conditional costs AND paired costs on shared successes.

    Logical calls bill K candidate queries even for cached payloads. Physical
    suffix generations are diagnostic only: cache reuse is not a policy gain.
    Failed/unobserved outcomes are retained in the all-scenario denominator.
    """
    report = {'scenario_count': len(rows), 'methods': {}, 'paired_both_success': {},
              'primary_steps': 'total_policy_steps_including_common_prefix',
              'primary_calls': 'logical_candidate_queries_including_common_prefix',
              'k_candidates_are_k_calls_not_one': True,
              'cache_savings_not_a_policy_efficiency_claim': True}
    for method in methods:
        all_outcomes = [r['methods'][method] for r in rows]
        successes = [o for o in all_outcomes if o['success'] and o.get('outcome_observed', False)]
        report['methods'][method] = {
            'scenarios': len(all_outcomes), 'success': len(successes),
            'success_rate': len(successes) / len(rows) if rows else None,
            'unobserved': sum(not o.get('outcome_observed', False) for o in all_outcomes),
            'successful_total_steps': describe(o['total_policy_steps'] for o in successes),
            'successful_postfork_steps': describe(o['executed_steps_after_fork'] for o in successes),
            'successful_total_logical_wam_calls': describe(total_calls(o) for o in successes),
            'successful_postfork_logical_wam_calls': describe(o['wam_calls_after_fork'] for o in successes),
            'successful_postfork_query_rounds': describe(o['wam_calls_after_fork']/4 for o in successes),
            'all_scenario_total_steps': describe(o['total_policy_steps'] for o in all_outcomes),
            'all_scenario_total_logical_wam_calls': describe(total_calls(o) for o in all_outcomes),
            'fallback_episodes': sum(bool(o.get('fallback_events')) for o in all_outcomes),
        }
    for method in methods:
        if method == full_method: continue
        paired = [r for r in rows if all(r['methods'][m]['success'] and r['methods'][m].get('outcome_observed', False)
                                        for m in [full_method, method])]
        entries = [{'task': r['task'], 'state': r['state'], 'condition': r['condition'],
                    'full_steps': r['methods'][full_method]['total_policy_steps'],
                    'control_steps': r['methods'][method]['total_policy_steps'],
                    'full_logical_calls': total_calls(r['methods'][full_method]),
                    'control_logical_calls': total_calls(r['methods'][method])} for r in paired]
        report['paired_both_success'][method] = {
            'n': len(paired), 'delta_sign': 'full_minus_control_negative_is_lower_cost',
            'steps_delta': describe(e['full_steps']-e['control_steps'] for e in entries),
            'logical_wam_calls_delta': describe(e['full_logical_calls']-e['control_logical_calls'] for e in entries),
            'entries': entries}
    return report
