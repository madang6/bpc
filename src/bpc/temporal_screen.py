"""Certified ordered top-k over the minimum distance across temporal rates."""
import math

import torch

from .fast_neighbors import CertifiedTopK, _error_bounds
from .precision import ieee_fp32


class TemporalTopK:
    """FP32 proposes endpoints; bounded FP64 refinement certifies their order.

    A rate is (bank, valid, first, last). The native bank has no rate mask.
    Taking minima is 1-Lipschitz in the infinity norm, so the maximum of
    per-rate error bounds also bounds the minimum-distance metric.
    """

    def __init__(self, rates):
        self.rates = [(CertifiedTopK(bank), valid, first, last)
                      for bank, valid, first, last in rates]

    @staticmethod
    def _mask(valid, first, last, rows, columns=None):
        if columns is not None:
            valid, first, last = valid[columns], first[columns], last[columns]
        return (~valid | ((first <= rows[:, None]) & (rows[:, None] <= last)))

    @torch.no_grad()
    def select(self, queries, count, rows, demos, starts, trajectories, history,
               reference_error=None, prefixes=(128, 256, 512, 1024)):
        output = torch.full((len(rows), count), -1, device=queries.device,
                            dtype=torch.long)
        certified = torch.zeros(len(rows), device=queries.device, dtype=torch.bool)
        excluded = ((rows[:, None] - history <= starts) & (starts <= rows[:, None]))
        if demos is not None:
            excluded |= trajectories == demos[:, None]
        removed = excluded.sum(1)
        with ieee_fp32():
            queries32 = queries.float()
            norm = queries32.square().sum(1)
            coarse = None
            error = torch.zeros(len(rows), device=queries.device, dtype=torch.float64)
            refinement = torch.zeros_like(error)
            finite = torch.ones_like(certified)
            for screen, valid, first, last in self.rates:
                distance = torch.addmm(screen.norm32, queries32, screen.bank32.T,
                                       beta=1, alpha=-2)
                distance.add_(norm[:, None]).clamp_min_(0)
                finite &= torch.isfinite(distance).all(1)
                if valid is not None:
                    distance.masked_fill_(self._mask(valid, first, last, rows), math.inf)
                if coarse is None:
                    coarse = distance
                else:
                    torch.minimum(coarse, distance, out=coarse)
                bound, refined = _error_bounds(queries, queries32, screen.bounds,
                                               reference_error)
                error = torch.maximum(error, bound)
                refinement = torch.maximum(refinement, refined)
            coarse.masked_fill_(excluded, math.inf)
            blocked = ~finite | ~torch.isfinite(error) | ~torch.isfinite(refinement)
            for prefix in prefixes:
                if not count < prefix < len(starts):
                    continue
                active = torch.nonzero(~certified & ~blocked &
                                       (len(starts) - removed > prefix)).flatten()
                if not len(active):
                    break
                values32, candidates = torch.topk(coarse[active], prefix + 1,
                                                  largest=False, sorted=True)
                selected = candidates[:, :prefix]
                distance = None
                for screen, valid, first, last in self.rates:
                    warped = (screen.bank[selected] - queries[active, None]).square().sum(2)
                    if valid is not None:
                        warped.masked_fill_(self._mask(valid, first, last,
                                                       rows[active], selected), math.inf)
                    if distance is None:
                        distance = warped
                    else:
                        torch.minimum(distance, warped, out=distance)
                values, positions = torch.topk(distance, count + 1,
                                               largest=False, sorted=True)
                lower = torch.nextafter(values - refinement[active, None],
                                        torch.full_like(values, -math.inf))
                upper = torch.nextafter(values + refinement[active, None],
                                        torch.full_like(values, math.inf))
                outside = torch.nextafter(values32[:, prefix].double() - error[active],
                                          torch.full_like(error[active], -math.inf))
                accepted = (torch.isfinite(distance).all(1)
                            & (upper[:, :count] < lower[:, 1:count + 1]).all(1)
                            & (upper[:, count - 1] < outside))
                chosen = active[accepted]
                output[chosen] = selected[accepted].gather(1, positions[accepted, :count])
                certified[chosen] = True
            return output, certified, removed
