"""Variable column sets, packed in storage and grouped for the existing solves."""

import torch

from .readout_bookkeeping import scatter_readout


class RaggedColumns:
    """CSR rows; only the requested batch is padded, never the full bank."""

    def __init__(self, values, offsets, *, _max_count=None):
        self.values, self.offsets = values, offsets.to(values.device)
        # Internal constructors already know Python row lengths. Reusing that
        # maximum avoids a device reduction and scalar synchronization.
        maximum = (int(self.counts.max()) if len(offsets) > 1 else 0) if _max_count is None else _max_count
        self.shape = (len(offsets) - 1, maximum,
                      *values.shape[1:])

    @property
    def counts(self):
        return self.offsets.diff()

    @classmethod
    def from_rows(cls, rows):
        values = torch.cat(rows)
        lengths = [len(row) for row in rows]
        counts = torch.tensor(lengths, device=values.device)
        return cls(values, torch.cat([counts.new_zeros(1), counts.cumsum(0)]),
                   _max_count=max(lengths, default=0))

    @classmethod
    def concatenate(cls, parts):
        counts = torch.cat([part.counts for part in parts])
        return cls(torch.cat([part.values for part in parts]),
                   torch.cat([counts.new_zeros(1), counts.cumsum(0)]))

    def __len__(self):
        return self.shape[0]

    def _rows(self, rows):
        return (torch.arange(len(self), device=self.values.device)[rows]
                if isinstance(rows, slice) else torch.as_tensor(
                    rows, device=self.values.device, dtype=torch.long).reshape(-1))

    def __getitem__(self, rows):
        return self._padded(rows)

    def _padding_plan(self, batch_size):
        """Integer metadata for repeated unchanged slices of this object.

        Build locally for each atlas invocation: retaining it on the object
        would require tracking later mutations of values/offsets. Every width
        is the original slice's maximum, not a global or padded batch width.
        """
        counts = self.counts
        lengths = counts.cpu().tolist()
        indices = torch.arange(len(self), device=self.values.device)
        return [(indices[first:first + batch_size], counts[first:first + batch_size],
                 max(lengths[first:first + batch_size]))
                for first in range(0, len(self), batch_size)]

    def _padded(self, rows, counts=None, width=None):
        rows = self._rows(rows)
        count = self.counts[rows] if counts is None else counts
        width = int(count.max()) if width is None else width
        positions = torch.arange(width, device=self.values.device)[None]
        valid = positions < count[:, None]
        indices = (self.offsets[rows, None] + positions).clamp_max(len(self.values) - 1)
        values = self.values[indices]
        return values.masked_fill(~valid.reshape(*valid.shape, *([1] * (values.ndim - 2))), -1)

    def take(self, rows):
        rows = self._rows(rows)
        counts = self.counts[rows]
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        indices = (torch.repeat_interleave(self.offsets[rows] - offsets[:-1], counts)
                   + torch.arange(int(offsets[-1]), device=self.values.device))
        return type(self)(self.values[indices], offsets)

    def align(self, rows, size):
        counts = self.counts.new_zeros(size)
        counts[rows] = self.counts
        return type(self)(self.values, torch.cat([counts.new_zeros(1), counts.cumsum(0)]))

    def to(self, device):
        return type(self)(self.values.to(device), self.offsets.to(device))

    def payload(self):
        return {"values": self.values.cpu(), "offsets": self.offsets.cpu()}


def pooled_mean(values, mask=None):
    if mask is None:
        return values.mean(1)
    return ((values * mask[..., None]).sum(1)
            / mask.sum(1).clamp_min(1)[:, None])


def support_groups(selected, device, *, include_distances=True):
    groups = {}
    for row, (columns, _) in enumerate(selected):
        if not len(columns):
            raise RuntimeError("sparsemax selected an empty support")
        groups.setdefault(len(columns), []).append(row)
    for members in groups.values():
        indices = torch.tensor(members, device=device)
        yield (indices, torch.stack([selected[row][0] for row in members]),
               torch.stack([selected[row][1] for row in members]) if include_distances else None)


def offline_supports_valid(policy, rows, packed, leave_demo_out=False):
    """Certify all ragged exclusion checks with one integer status read.

    False retains the existing per-group checks and their exception ordering.
    In particular, clamp only the certificate's lookup indices: malformed
    indices must reach the original gather rather than become valid supports.
    No floating-point input, group, or solve is changed by this certificate.
    """
    columns = packed.values
    if columns.dtype != torch.long or columns.ndim != 1 or not len(policy.hankel_starts):
        return False
    count = len(policy.hankel_starts)
    invalid = (columns < -count) | (columns >= count)
    safe = columns.clamp(-count, count - 1)
    targets = torch.repeat_interleave(rows, packed.counts, output_size=len(columns))
    starts = policy.hankel_starts[safe]
    invalid |= (targets - policy.history <= starts) & (starts <= targets)
    if leave_demo_out:
        invalid |= policy.hankel_trajectory[safe] == policy.trajectory_of[targets]
    return not bool(invalid.any())


@torch.no_grad()
def direct_actions(policy, actions, states):
    """Reuse the same G/RFF readout at each true cardinality; no padded columns."""
    query = torch.cat([actions.flatten(1), states.flatten(1)], 1)
    selected = policy.support_selector.select(query)
    from .concurrent_readout import direct_actions as concurrent_actions
    concurrent = concurrent_actions(policy, actions, states, selected)
    if concurrent is not None:
        output, traces = concurrent
    else:
        from .ragged_readout import prepare_readout
        output, traces = query.new_empty(len(query), policy.action_dim), {}
        prepared = prepare_readout(policy, actions, states, selected)
        groups = (prepared if prepared is not None else
                  ((rows, actions[rows], states[rows], columns, distances, None)
                   for rows, columns, distances in support_groups(selected, query.device)))
        for rows, u, y, columns, distances, values_ready in groups:
            kwargs = {} if values_ready is None else {"_prepared": values_ready}
            result = policy.direct_actions(u, y, selected=(columns, distances), **kwargs)
            scatter_readout(output, traces, rows, result)
    traces["selector_count"] = query.new_tensor([len(columns) for columns, _ in selected])
    return output, traces
