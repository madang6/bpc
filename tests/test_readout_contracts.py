"""Extending policy metadata preserves acceleration; changing its math opts out."""
import types
import unittest
from unittest.mock import patch

from dpc_nldl.bpc import Policy
from dpc_nldl.lda_selector import Selector
from dpc_nldl.readout_contracts import supports_readout
from dpc_nldl.retrieval import FisherLookup


def contract_policy(policy_type=Policy):
    """Only method ownership matters to these predicates; no fitted bank is needed."""
    policy = policy_type.__new__(policy_type)
    policy.g_lookup = 'lda'
    policy.retrieval = FisherLookup()
    policy.support_selector = Selector.__new__(Selector)
    return policy


class ReadoutContractTests(unittest.TestCase):
    def test_inherited_policy_selector_and_retrieval_keep_acceleration(self):
        class AnnotatedPolicy(Policy):
            pass

        class AnnotatedSelector(Selector):
            pass

        class AnnotatedRetrieval(FisherLookup):
            pass

        p = contract_policy(AnnotatedPolicy)
        p.support_selector.__class__ = AnnotatedSelector
        p.retrieval = AnnotatedRetrieval()
        self.assertTrue(supports_readout(p))
        self.assertTrue(supports_readout(p, online=True, captured=True))

    def test_changed_numerical_methods_are_ineligible(self):
        p = contract_policy()
        overrides = [(p, name, online) for name, online in (
            ('_future_columns', True), ('_interpolate', False),
            ('_float_sources', False), ('_fixed_g', False))]
        overrides += [(p.support_selector, 'features', True), (p.retrieval, 'project', True)]
        for owner, method, online in overrides:
            with self.subTest(owner=type(owner).__name__, method=method), patch.object(
                    owner, method, wraps=getattr(owner, method)):
                self.assertFalse(supports_readout(p, online=online))
                self.assertFalse(supports_readout(p, online=online, captured=True))

    def test_capture_rejects_even_an_instance_alias_of_the_original_method(self):
        p = contract_policy()
        p._rate_trace = types.MethodType(Policy._rate_trace, p)
        self.assertTrue(supports_readout(p, online=True))
        self.assertFalse(supports_readout(p, online=True, captured=True))


if __name__ == '__main__':
    unittest.main()
