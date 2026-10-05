import unittest
from wam_reranking.closedloop_metrics import execution_cost_summary


def outcome(success, steps, calls):
    return dict(success=success, outcome_observed=True, total_policy_steps=48+steps,
                executed_steps_after_fork=steps, common_prefix_wam_calls=3,
                wam_calls_after_fork=calls, fallback_events=[])


class CostMetricsTests(unittest.TestCase):
    def rows(self):
        return [dict(task=0,state=0,condition='clean',methods={
            'value_only':outcome(True,100,28),'full_repaired':outcome(True,80,24)}),
            dict(task=0,state=1,condition='clean',methods={
            'value_only':outcome(False,352,88),'full_repaired':outcome(True,40,12)})]

    def test_prefix_and_k_calls_not_omitted(self):
        report=execution_cost_summary(self.rows(),['value_only','full_repaired'])
        self.assertEqual(report['methods']['value_only']['successful_total_steps']['mean'],148)
        self.assertEqual(report['methods']['value_only']['successful_total_logical_wam_calls']['mean'],31)
        self.assertEqual(report['methods']['value_only']['successful_postfork_query_rounds']['mean'],7)

    def test_paired_denominator_is_shared_success_only(self):
        report=execution_cost_summary(self.rows(),['value_only','full_repaired'])
        pair=report['paired_both_success']['value_only']
        self.assertEqual(pair['n'],1)
        self.assertEqual(pair['steps_delta']['mean'],-20)
        self.assertEqual(pair['logical_wam_calls_delta']['mean'],-4)

    def test_failure_stays_in_success_rate_denominator(self):
        report=execution_cost_summary(self.rows(),['value_only','full_repaired'])
        self.assertEqual(report['methods']['value_only']['success_rate'],.5)
        self.assertEqual(report['methods']['value_only']['all_scenario_total_steps']['n'],2)

    def test_no_success_returns_null_not_zero_cost(self):
        rows=self.rows()
        for row in rows:row['methods']['full_repaired']['success']=False
        report=execution_cost_summary(rows,['value_only','full_repaired'])
        self.assertIsNone(report['methods']['full_repaired']['successful_total_steps']['mean'])
        self.assertEqual(report['paired_both_success']['value_only']['n'],0)

    def test_explicit_full_method_for_four_arm_ablation(self):
        rows = self.rows()
        for row in rows: row['methods']['full'] = row['methods'].pop('full_repaired')
        report = execution_cost_summary(rows, ['value_only', 'full'], full_method='full')
        self.assertEqual(report['paired_both_success']['value_only']['n'], 1)
        self.assertEqual(report['paired_both_success']['value_only']['steps_delta']['mean'], -20)


if __name__=='__main__':unittest.main()
