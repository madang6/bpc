"""Sparsemax simplex projection and uncapped adaptive support selection."""
import math
import torch


def sparsemax_support(topk, width, scale):
    """Full-bank sparsemax support from certified descending-score prefixes.

    ``topk(k, rows)`` returns column indices and their scores (None means all rows).
    A zero final weight
    certifies that every omitted (no larger) score also has zero weight.
    """
    if width < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive sparsemax scale and bank width required')
    count = min(32, width)
    rows = None
    while True:
        columns, scores = topk(count, rows)
        if rows is None:
            rows = torch.arange(len(scores), device=scores.device)
            output = [None] * len(rows)
        probability = sparsemax(scores * scale)
        done = torch.ones_like(rows, dtype=torch.bool) if count == width else probability[:, -1] == 0
        # Pack all completed rows with one nonzero scan. Per-row boolean
        # indexing otherwise performs two device scans/synchronizations for
        # every query. Row-major positions retain the original support order.
        positive = (probability > 0) & done[:, None]
        positions = positive.reshape(-1).nonzero(as_tuple=False).flatten()
        packed_columns = columns.reshape(-1)[positions]
        packed_scores = scores.reshape(-1)[positions]
        metadata = torch.stack((rows, done, positive.sum(1)), 1).tolist()
        offset = 0
        for row, finished, length in metadata:
            if finished:
                output[row] = (packed_columns[offset:offset + length],
                               packed_scores[offset:offset + length])
            offset += length
        rows = rows[~done]
        if not len(rows):
            return output
        count = min(2 * count, width)


def sparsemax(scores):
    """Exact simplex projection; -inf marks excluded columns, never a support cap."""
    if (scores.ndim < 1 or not scores.shape[-1]
            or bool((torch.isnan(scores) | torch.isposinf(scores)).any())
            or not bool(torch.isfinite(scores).any(-1).all())):
        raise ValueError("sparsemax needs at least one finite score per row")
    shifted = scores - scores.max(-1, keepdim=True).values
    width, count = scores.shape[-1], min(32, scores.shape[-1])
    while True:
        values = shifted.topk(count, dim=-1).values
        sums = values.cumsum(-1)
        ranks = torch.arange(1, count + 1, device=scores.device)
        active = (1 + ranks * values > sums).sum(-1, keepdim=True)
        threshold = (sums.gather(-1, active - 1) - 1) / active
        if count == width or bool((values[..., -1:] <= threshold).all()):
            return (shifted - threshold).clamp_min(0)
        count = min(2 * count, width)
