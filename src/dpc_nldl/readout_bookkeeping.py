"""Trace assembly shared by ordinary and accelerated policy execution."""

RAGGED_TRACE_KEYS = frozenset({
    'hankel_supports', 'hankel_rates', 'hankel_unwarped_distance', 'g_coefficients',
})


def scatter_readout(output, traces, rows, result):
    """Scatter actions and fixed-width diagnostics back to their query rows."""
    values, trace = result
    output[rows] = values
    for name, value in trace.items():
        if name in RAGGED_TRACE_KEYS:
            continue
        if name not in traces:
            traces[name] = value.new_empty((len(output), *value.shape[1:]))
        traces[name][rows] = value


def recovery_trace(policy, states, status, fired, exhausted):
    return {
        'recovery_fired': fired.to(states.dtype),
        'recovery_active': (status[:, 1] > 0).to(states.dtype),
        'recovery_attempts': status[:, 2].to(states.dtype),
        'recovery_exhausted': exhausted.to(states.dtype),
        'recovery_aperture': policy._raw_aperture(states),
        'recovery_closed_ticks': status[:, 0],
    }
