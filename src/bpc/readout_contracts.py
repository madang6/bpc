"""Implementation contracts for optional readout accelerators.

Subclasses may add behavior outside these kernels. Overriding a method whose
arithmetic is bypassed or captured opts back into ordinary policy execution.
"""

_COMMON = (
    '_rate_trace', '_g_lookup_projection', '_g_prior', '_reconstruction_columns',
    '_future_columns', '_support_endpoints', '_solve_affine', '_solve_g_normal',
    '_trajectory_stops', '_interpolated_keys', '_past_columns',
    '_retime_action_values', '_interpolate',
)
_ONLINE = ('direct_actions', '_query_features', '_g_kkt')
_OFFLINE = ('_fixed_g', '_float_sources')
_SELECTOR = ('prepare', 'features', '_distance', 'rate_trace')


def _unchanged(instance, base, names, captured):
    for name in names:
        if captured and name in instance.__dict__:
            # Even transparent profiler wrappers have Python side effects
            # which a captured graph would skip on subsequent executions.
            return False
        method = getattr(instance, name)
        if getattr(method, '__func__', method) is not getattr(base, name):
            return False
    return True


def supports_readout(policy, *, online=False, captured=False):
    """Certify inherited numerical methods; route-specific guards stay local."""
    from .bpc import Policy
    from .lda_selector import Selector
    from .retrieval import is_builtin_retrieval
    selector = getattr(policy, "support_selector", None)
    if (not isinstance(policy, Policy) or not isinstance(selector, Selector)
            or not is_builtin_retrieval(policy)):
        return False
    if captured and any(name in policy.retrieval.__dict__ for name in ('fit', 'project')):
        return False
    methods = _COMMON + (_ONLINE if online else _OFFLINE)
    if captured:
        methods += ('_rate_target_overlap',)
    return (_unchanged(policy, Policy, methods, captured)
            and _unchanged(selector, Selector, _SELECTOR, captured))
