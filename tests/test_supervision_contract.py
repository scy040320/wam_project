import unittest
import numpy as np

from wam_reranking.supervision_contract import (
    contrastive_supervised_rows, paired_object_indices, supervised_factor_recall,
    supported_normal, task_development_gate,
)


def test_masked_positive_cannot_fail_recall_gate():
    recall, support = supervised_factor_recall([[.8],[.1]], [[1],[1]], [[1],[0]])
    assert recall == [1.0] and support == [1]


def test_contrastive_rows_include_normal_with_complete_supervision():
    eligible = contrastive_supervised_rows([1,1,0], [[1,1,1,1],[1,0,1,1],[1,1,1,1]])
    np.testing.assert_array_equal(eligible, [True,False,False])


def test_pairing_stays_in_source_state_moment_and_train():
    def part(states, split):
        return {'split':np.asarray(split), 'tasks':np.asarray([16]*4), 'states':np.asarray(states),
                'moments':np.zeros(4,int), 'y':np.asarray([0,2,0,2]),
                'X':np.ones((4,2)), 'f':np.asarray([[0,0,0,0,0,0],[0,1,0,0,0,0]]*2)}
    first = part([0,0,1,1], ['train','train','val','val'])
    second = part([0,2,3,3], ['train']*4)
    pos,neg = paired_object_indices([first,second])
    np.testing.assert_array_equal(pos,[1,5]); np.testing.assert_array_equal(neg,[0,4])


def test_normal_head_cannot_override_unknown_observability_or_positive_factor():
    result = supported_normal([0,0,0,1], [.9]*4, [[0,0,0,0],[0,0,0,0],[0,.6,0,0],[0,0,0,0]],
                              [1,0,1,0], [0]*4,[0]*4)
    np.testing.assert_array_equal(result,[True,False,False,True])


def test_known_action_deviation_and_unreliable_records_block_normal():
    result = supported_normal([1,1],[.9,.9],np.zeros((2,4)),[1,1],[1,0],[0,1])
    np.testing.assert_array_equal(result,[False,False])


def test_original_point_five_gate_remains_identical_across_tasks():
    tasks = (14,16,17,23,34,39,45,58,64,73)
    metrics = {str(t):{'per_class':{c:{'recall':.6} for c in ('normal','object_shift','unknown','execution_contact_deviation')}} for t in tasks}
    assert all(task_development_gate(metrics).values())
    metrics['16']['per_class']['object_shift']['recall'] = .4
    assert not task_development_gate(metrics)['predeclared_observable_task_object_shift_recall_each_at_least_0_50']


class SupervisionContractTests(unittest.TestCase):
    """Expose the contract checks to the repository's dependency-light runner."""
    test_masked_positive = staticmethod(test_masked_positive_cannot_fail_recall_gate)
    test_contrastive_rows = staticmethod(test_contrastive_rows_include_normal_with_complete_supervision)
    test_source_scoped_pairing = staticmethod(test_pairing_stays_in_source_state_moment_and_train)
    test_normal_observability = staticmethod(test_normal_head_cannot_override_unknown_observability_or_positive_factor)
    test_action_evidence = staticmethod(test_known_action_deviation_and_unreliable_records_block_normal)
    test_unchanged_task_gate = staticmethod(test_original_point_five_gate_remains_identical_across_tasks)
