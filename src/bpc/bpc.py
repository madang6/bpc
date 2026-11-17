"""Manuscript BPC: future-informed retrieval, affine continuation and shared-support RFF correction."""

import math
from collections import deque

import numpy as np
import torch

from . import artifacts, fitting, retrieval
from .artifacts import apply_variant_runtime, fit_policy, load_policy, prepare_lookups
from .data import encoder_spec
from .retrieval import squared_distance as _squared_distance
from .ragged_supports import RaggedColumns, pooled_mean, support_groups
from .pipeline import (
    METHODS, temporal_method,
    target_execution_options,
)

SEARCH_ROWS = 4096
G_PLANE_BYTES = 512 * 1024**2
G_NORMAL_BYTES = 192 * 1024**2
G_SUPPORT_BATCH = 6144


class Policy:
    # Set by the batched evaluator (evaluation.readout_rows): every row of a
    # readout then runs the same shape-fixed ops, so its bits never depend on
    # which other rows share the batch.  Off keeps the parent arithmetic.
    fixed_shape_readout = False
    # Exact-shape singleton CUDA graphs; unsupported cases keep eager execution.
    concurrent_readout = True

    def __init__(self, trajectories, data_config, method_config, fitted=None,
                 fit_residuals=True, load_operators=True):
        method_config = temporal_method(method_config)
        self.config = method_config
        self.target_execution_options = target_execution_options(method_config)
        self.device = torch.device(method_config["device"])
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[
            method_config.get("dtype", "float32")]
        self.history = int(method_config["history"])
        self.hankel_stride = int(method_config.get("hankel_stride", 1))
        self.state_dim = int(method_config["state_dim"])
        self.action_dim = int(method_config["action_dim"])
        keys = data_config["feature_keys"]
        visual = len(keys) * int(encoder_spec(data_config)[1]["feature_dim"]) if keys else 0
        self.proprio_dim = int(method_config["scaling"].get(
            "proprio_dim", self.state_dim - visual))
        if not 0 < self.proprio_dim <= self.state_dim:
            raise ValueError("scaling.proprio_dim is inconsistent with state_dim")
        self._build_bank(trajectories, data_config, fitted)

        self.fit_residuals = fit_residuals
        self.lookback = self.history
        if fit_residuals:
            self.residual_lookup = method_config["residual"].get("lookup", "g_supports")
            neighbors = method_config["temporal_neighbors"]
            self.lookback = int(neighbors["lookback"])
            self.candidates = min(int(neighbors["candidates"]), self.samples - 1)
            self.neighbor_count = max(1, math.floor(
                self.candidates * float(neighbors["keep_ratio"])))
            j = torch.arange(1, self.lookback + 1, device=self.device, dtype=self.dtype)
            self.decay_weights = j.pow(float(neighbors["decay"]))

            self.input_dim = 2 * self.state_dim + self.action_dim
            self.residual_model = method_config["residual"].get("model", "rff")
            rff = method_config["rff"]
            self.rff_dim = int(rff["features"])
            self.rff_ridge = float(rff["ridge"])
            self.rff_scale = math.sqrt(2.0 / self.rff_dim)
            self.rff_dtype = {"float32": torch.float32, "float64": torch.float64}[
                rff.get("dtype", method_config.get("dtype", "float32"))]
            self.rff_atlas = rff.get("dtype") == "float32"
            self.rff_storage = rff.get("storage", "auto")
            if self.rff_storage not in {"auto", "resident", "host_paged"}:
                raise ValueError(f"unknown RFF storage {self.rff_storage}")
            if self.rff_storage == "host_paged" and (
                    not self.rff_atlas or self.device.type != "cuda"):
                raise ValueError("host_paged RFF requires CUDA and rff.dtype=float32")
        g = method_config["g"]
        self.g_state_offset = g.get("state_offset", 0)
        if self.g_state_offset not in (0, 1):
            raise ValueError("g.state_offset must be 0 or 1")
        self.g_state_offset = int(self.g_state_offset)
        self.g_lookup = g.get("lookup", "past")
        self.g_temporal_screen = g.get("temporal_support_screen", False)
        self.g_lookup_rates = tuple(map(float, g.get("lookup_rates", [1])))
        if len(self.g_lookup_rates) > 1 and self.g_lookup == "past":
            raise ValueError("g.lookup_rates requires a learned lookup")
        self._rate_keys = None
        self.g_align_history = bool(g.get("align_history", False))
        self.g_align_future = bool(g.get("align_future", False))
        self.g_rate_scaled_action_dims = tuple(g.get("rate_scaled_action_dims", []))
        if (len(set(self.g_rate_scaled_action_dims)) != len(self.g_rate_scaled_action_dims)
                or any(type(d) is not int or not 0 <= d < self.action_dim
                       for d in self.g_rate_scaled_action_dims)
                or self.g_rate_scaled_action_dims and not self.g_align_future):
            raise ValueError("g.rate_scaled_action_dims requires valid dimensions and align_future")
        if self.g_align_future and not self.g_align_history:
            raise ValueError("g.align_future requires g.align_history")
        if self.g_align_history and self.g_lookup == "past":
            raise ValueError("g.align_history requires a learned lookup")
        self.g_weights = g.get("weights", "optimized")
        self.g_supports = int(g["supports"])
        if fit_residuals:
            count = self.g_supports
            self.feature_batch = max(1, int(rff["feature_pool_chunk_elements"])
                                     // (count * self.rff_dim))
        self.g_l1 = float(g["l1"])
        self.g_constraint = g.get("constraint", "affine")
        self.g_solver = g.get("solver", "pgd")
        if float(g.get("l2", 0)) or self.g_solver not in {"pgd", "exact"} or (
                self.g_solver == "exact" and self.g_l1):
            raise ValueError("exact G solver requires unregularized affine coordinates")
        self.g_iterations = int(g.get("iterations", 200))
        self.g_tolerance = (float(g["tolerance"]) if "tolerance" in g else None)
        self.g_fit_dtype = {"float32": torch.float32, "float64": torch.float64}[
            g.get("fit_dtype", method_config.get("dtype", "float32"))]
        self.g_native = "fit_dtype" in g and self.g_fit_dtype == torch.float32
        self.set_residual_action_source(method_config.get("residual", {}).get(
            "action_source", "native"))
        self._fit_sources = self._support_screen = None

        self.horizon = int(method_config["continuation_horizon"])
        if fit_residuals:
            residual = method_config["residual"]
            self.target_batch = int(residual["target_batch_size"])
            self.residual_variance_floor = float(residual["variance_floor"])
        matrix = (fitted or {}).get("g_lookup_matrix")
        self.g_lookup_matrix = (matrix.to(self.device, dtype=self.dtype)
                                if self.g_lookup != "past" and matrix is not None
                                else None)
        self.g_lookup_scale = float((fitted or {}).get("g_lookup_scale", 1))
        self.g_lookup_stats = (fitted or {}).get("g_lookup")
        self.support_selector = None
        if self.g_lookup == "lda":
            from .lda_selector import Selector
            self.support_selector = Selector(self, (fitted or {}).get("lda"))
        self.retrieval = retrieval.make_retrieval(self.g_lookup)
        self._build_hankel()

        self.omega = self.phase = self.residual_weights = self.auxiliary_residual_weights = None
        self.rff_bank_weight = self.rff_query_weight = self.rff_bias = None
        self.residual_variance = self.auxiliary_residual_variance = None
        self.sigma = None
        self.fit_stats = None
        if fitted is not None and load_operators:
            self.load_fitted_payload(fitted)
        self.method = "bpc" if fit_residuals else "bpc_nnc"
        self.reset()

    def _build_bank(self, trajectories, data_config, fitted):
        state_key = data_config["state_key"]
        action_key = data_config["action_key"]
        feature_keys = data_config["feature_keys"]
        states, actions, starts = [], [], []
        cursor = 0
        for trajectory in trajectories:
            starts.append(cursor)
            state = np.concatenate([
                np.asarray(trajectory[state_key]),
                *(np.asarray(trajectory[key]) for key in feature_keys),
            ], axis=0).T
            action = np.asarray(trajectory[action_key]).T
            if state.shape != (len(action), self.state_dim):
                raise ValueError(f"bank state/action shape mismatch: {state.shape}, {action.shape}")
            states.append(torch.from_numpy(state).to(dtype=self.dtype))
            actions.append(torch.from_numpy(action).to(dtype=self.dtype))
            cursor += len(action)

        raw_state = torch.cat(states).to(self.device)
        raw_action = torch.cat(actions).to(self.device)
        self.samples = len(raw_state)
        self.demo_count = len(trajectories)
        self.trajectory_starts = torch.tensor(starts, device=self.device)
        self.demo_start_state = raw_state[self.trajectory_starts]
        rows = torch.arange(self.samples, device=self.device)
        self.trajectory_of = torch.searchsorted(
            self.trajectory_starts, rows, right=True) - 1
        self.row_in_trajectory = rows - self.trajectory_starts[self.trajectory_of]

        scaling = self.config["scaling"]
        if fitted is not None:
            self.state_mean = fitted["state_mean"].to(self.device, dtype=self.dtype)
            self.state_scale = fitted["state_scale"].to(self.device, dtype=self.dtype)
            self.action_mean = fitted["action_mean"].to(self.device, dtype=self.dtype)
            self.action_scale = fitted["action_scale"].to(self.device, dtype=self.dtype)
        elif scaling["enabled"]:
            correction = scaling["standard_deviation_correction"]
            minimum = float(scaling["minimum_standard_deviation"])
            self.state_mean = raw_state.mean(0)
            self.state_scale = raw_state.std(0, correction=int(correction["state"])).clamp_min(minimum)
            self.action_mean = raw_action.mean(0)
            self.action_scale = raw_action.std(0, correction=int(correction["action"])).clamp_min(minimum)
        else:
            self.state_mean = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
            self.state_scale = torch.ones(self.state_dim, device=self.device, dtype=self.dtype)
            self.action_mean = torch.zeros(self.action_dim, device=self.device, dtype=self.dtype)
            self.action_scale = torch.ones(self.action_dim, device=self.device, dtype=self.dtype)

        if fitted is None and scaling["block_normalize"]:
            proprio = int(scaling["proprio_dim"])
            visual_blocks = len(feature_keys)
            visual_dim = (self.state_dim - proprio) // visual_blocks
            block_scale = torch.ones(self.state_dim, device=self.device, dtype=self.dtype)
            block_scale[:proprio] = math.sqrt(proprio) / float(scaling["proprio_weight"])
            for block in range(visual_blocks):
                start = proprio + block * visual_dim
                block_scale[start:start + visual_dim] = math.sqrt(visual_dim)
            self.state_scale *= block_scale

        self.state = (raw_state - self.state_mean) / self.state_scale
        self.action = (raw_action - self.action_mean) / self.action_scale
        self.state_norm = torch.einsum("ij,ij->i", self.state, self.state)

    def _build_hankel(self):
        """Build the history search plane and demonstrated future-action bank."""
        length = self.history + self.horizon
        if not 1 <= self.hankel_stride <= length:
            raise ValueError("hankel_stride must be between 1 and history + horizon")
        stops = [*self.trajectory_starts.tolist()[1:], self.samples]
        starts, rollout_ids = [], []
        for demo, (start, stop) in enumerate(zip(self.trajectory_starts.tolist(), stops)):
            count = stop - start - length + 1 - self.g_state_offset
            if count <= 0:
                continue
            selected = torch.arange(start, start + count, self.hankel_stride, device=self.device)
            starts.append(selected)
            rollout_ids.append(torch.full_like(selected, demo))
        self.hankel_starts = torch.cat(starts)
        self.hankel_trajectory = torch.cat(rollout_ids)
        columns = len(self.hankel_starts)
        past_width = self.history * (self.action_dim + self.state_dim)
        action_horizon = self.horizon * self.action_dim
        plane_width = (self.g_lookup_matrix.shape[1]
                       if self.g_lookup_matrix is not None else past_width)
        plane_bytes = columns * plane_width * torch.empty((), dtype=self.dtype).element_size()
        # A learned-lookup plane is immediately replaced by its compact projected
        # keys.  Do not materialize the full R3M history plane first.
        compact_pending = (self.g_lookup != "past"
                           and self.g_lookup_matrix is None)
        keep_plane = (not compact_pending and (
            self.g_lookup_matrix is not None or self.device.type != "cuda"
            or plane_bytes
            <= torch.cuda.get_device_properties(self.device).total_memory // 4))
        self.pf = torch.empty(columns, plane_width, dtype=self.dtype) if keep_plane else None
        self.pf_norm = (torch.empty(columns, device=self.device, dtype=self.dtype)
                        if keep_plane else None)
        self.uf = torch.empty(action_horizon, columns, device=self.device, dtype=self.dtype)

        past_offsets = torch.arange(self.history, device=self.device)
        future_offsets = self.history + torch.arange(self.horizon, device=self.device)
        state = self.state[self.g_state_offset:]
        for start in range(0, columns, fitting.FIT_ROWS):
            stop = min(start + fitting.FIT_ROWS, columns)
            bases = self.hankel_starts[start:stop, None]
            if self.pf is not None:
                past_u = self.action[bases + past_offsets]
                past_y = state[bases + past_offsets]
            future_u = self.action[bases + future_offsets]
            if self.pf is not None:
                past = torch.cat([
                    past_u.reshape(stop - start, -1),
                    past_y.reshape(stop - start, -1),
                ], 1)
                if self.g_lookup_matrix is not None:
                    past = self._g_lookup_projection(past)
                self.pf[start:stop].copy_(past.cpu())
                self.pf_norm[start:stop] = torch.einsum("ij,ij->i", past, past)
            self.uf[:, start:stop] = future_u.reshape(stop - start, -1).T
            fitting.progress("Hankel columns", stop, columns)
        if self.pf is not None:
            self.pf = self.pf.to(self.device)
        torch.cuda.empty_cache()

    def scale_state(self, state):
        value = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        return (value - self.state_mean) / self.state_scale

    def _legacy_retrieve(self, windows, lengths, exclude=None, exclude_demo=None):
        """
        BPC temporal retrieval, which uses the last state in the window to
        select candidates, then accumulates a weighted distance over the entire window.
        """
        batch = len(windows)
        distance = _squared_distance(
            windows[:, -1], self.state, self.state_norm)
        distance[distance == 0] = float("inf")
        if exclude is not None:
            distance[torch.arange(batch, device=self.device), exclude] = float("inf")
        if exclude_demo is not None:
            distance.masked_fill_(
                self.trajectory_of[None] == exclude_demo[:, None], float("inf"))
        nearest_distance, candidates = torch.topk(
            distance, self.candidates, largest=False, dim=1)
        accumulated = None
        if (self.g_native and batch > 1 and self.state.dtype == torch.float64
                and self.lookback <= 10):
            from .fast_cuda import call
            accumulated = call(
                "history_accumulators", windows, windows.contiguous(),
                lengths.contiguous(), candidates.contiguous(), self.state.contiguous(),
                self.row_in_trajectory.contiguous(), self.decay_weights.contiguous())
        if accumulated is None:
            candidate_age = self.row_in_trajectory[candidates]
            max_lookback = torch.minimum(
                candidate_age + 1,
                lengths[:, None].expand_as(candidate_age),
            ).clamp(max=self.lookback)
            accumulated = torch.zeros_like(nearest_distance)
            for offset in range(self.lookback):
                valid = offset < max_lookback
                bank_rows = (candidates - offset).clamp(min=0)
                query = windows[:, self.lookback - 1 - offset]
                step_distance = torch.linalg.vector_norm(
                    self.state[bank_rows] - query[:, None], dim=2)
                accumulated += valid * self.decay_weights[offset] * step_distance
        local = torch.topk(
            accumulated, self.neighbor_count, largest=False, dim=1).indices
        selected = torch.gather(candidates, 1, local)
        return selected, nearest_distance[:, 0].sqrt()

    def _factor_rff(self):
        split = self.state_dim + self.action_dim
        query = self.omega[split:].to(self.device) / fitting.SQRT2
        self.rff_bank_weight = torch.cat([
            self.omega[:self.state_dim].to(self.device) + query,
            self.omega[self.state_dim:split].to(self.device),
        ]).to(self.rff_dtype).contiguous()
        self.rff_query_weight = query.to(self.rff_dtype).contiguous()
        self.rff_bias = self.phase.to(self.device, self.rff_dtype).contiguous()
        self.omega, self.phase = self.omega.cpu(), self.phase.cpu()

    def _query_features(self, queries, neighbors, neighbor_actions=None, mask=None):
        from .fused_rff_phase import phase_cosine_
        previous_tf32 = (torch.backends.cuda.matmul.allow_tf32
                         if self.device.type == "cuda" else None)
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
        output = []
        try:
            neighbors = neighbors if mask is None else neighbors.clamp_min(0)
            bank = torch.cat([
                self.state[neighbors], (self.action[neighbors] if neighbor_actions is None
                                        else neighbor_actions),
            ], 2).to(self.rff_dtype)
            queries = queries.to(self.rff_dtype)
            # Keep deployed N=1 projection rounding for large off-bank queries.
            projected = torch.cat([
                row[None, None] @ self.rff_query_weight for row in queries])
            # Bound B*K*D scratch while sharing the projections across live rows.
            tile = self.rff_dim if len(queries) == 1 else min(fitting.FIT_ROWS, self.rff_dim)
            for start in range(0, self.rff_dim, tile):
                stop = min(start + tile, self.rff_dim)
                phase = bank @ self.rff_bank_weight[:, start:stop]
                bias, query_phase = self.rff_bias[start:stop], projected[:, :, start:stop]
                if phase_cosine_(phase, bias, query_phase) is None:
                    phase.add_(bias).sub_(query_phase).cos_()
                output.append(pooled_mean(phase, mask).mul_(self.rff_scale))
        finally:
            if previous_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        return torch.cat(output, dim=1)

    def _queries(self, rows, dtype=None):
        offsets = torch.arange(self.history, device=self.device)
        action_rows = rows[:, None] - self.history + offsets
        state_rows = rows[:, None] - self.history + 1 + offsets
        action, state = (self._float_sources() if dtype == torch.float32 else
                         (self.action, self.state))
        return torch.cat([
            action[action_rows].reshape(len(rows), -1),
            state[state_rows].reshape(len(rows), -1),
        ], 1)

    def _float_sources(self):
        if self._fit_sources is None:
            self._fit_sources = tuple(
                value if value.dtype == torch.float32 else value.float()
                for value in (self.action, self.state))
        return self._fit_sources

    def _fixed_g(self, supports, rows, *, rates=None, query_history=None):
        """Solve fixed supports in bounded history tiles."""
        batch, count = supports.shape
        tile = self._g_microbatch(count)
        grams, rhs, coefficients = [], [], []
        for start in range(0, batch, tile):
            stop = min(start + tile, batch)
            selected = rows[start:stop]
            history = (self._queries(selected) if query_history is None else
                       query_history[start:stop])
            rate_args = {} if rates is None else {"rates": rates[start:stop]}
            design = self._reconstruction_columns(
                supports[start:stop], history, torch.float32, selected,
                **rate_args).transpose(1, 2)
            query = history.float()
            if self.g_solver == "exact":
                coefficients.append(self._solve_affine(design, query))
                continue
            transpose = design.transpose(1, 2)
            grams.append(transpose @ design)
            rhs.append((transpose @ query.unsqueeze(2)).squeeze(2))
        return (torch.cat(coefficients) if coefficients else self._solve_g_normal(
            torch.cat(grams), torch.cat(rhs)))

    def _g_microbatch(self, count):
        width = self.history * (self.action_dim + self.state_dim)
        item_bytes = 4
        return max(1, G_NORMAL_BYTES * (2 if self.g_solver == "exact" else 1)
                   // (count * width * item_bytes))

    def _past_columns(self, columns, dtype=None, rate=1.0):
        shape = columns.shape
        bases = self.hankel_starts[columns.reshape(-1), None]
        offsets = torch.arange(self.history, device=self.device)
        actions, states = (self._float_sources() if dtype == torch.float32 else
                           (self.action, self.state))
        states = states[self.g_state_offset:]
        if rate == 1 and self.g_native and states.is_cuda and dtype in (None, states.dtype):
            from .fast_cuda import call
            packed = call("pack_history", states, states, actions[:len(states)],
                          bases.ravel().contiguous(), self.history)
            if packed is not None:
                return packed.reshape(*shape, -1)
        if rate == 1:
            actions, states = actions[bases + offsets], states[bases + offsets]
        else:
            positions = bases + self.history - 1 + rate * (offsets.double() - self.history + 1)
            lower = positions.floor().long()
            demo_start = self.trajectory_starts[self.hankel_trajectory[columns.reshape(-1)]]
            if bool((lower[:, 0] < demo_start).any()):
                raise ValueError("interpolated history crosses a demonstration boundary")
            actions, states = (self._interpolate(source, positions)
                               for source in (actions, states))
        actions = self._retime_action_values(actions, rate)
        if dtype is not None and dtype != actions.dtype:
            actions, states = actions.to(dtype), states.to(dtype)
        return torch.cat([
            actions.reshape(len(bases), -1), states.reshape(len(bases), -1),
        ], 1).reshape(*shape, -1)

    def _support_endpoints(self, supports):
        return self.hankel_starts[supports] + (self.history - 1 + self.g_state_offset)

    def retimed_neighbor_actions(self, queries, neighbors, target_rows=None):
        """Decode G's first-action columns from cached residual endpoint indices.

        Endpoint states remain native. Offline callers must supply target_rows,
        so rate selection applies the same interpolation leakage exclusions as G.
        """
        starts = neighbors - (self.history - 1 + self.g_state_offset)
        supports = torch.searchsorted(self.hankel_starts, starts)
        if (bool((supports >= len(self.hankel_starts)).any())
                or not torch.equal(self.hankel_starts[supports], starts)):
            raise ValueError("residual endpoints do not identify Hankel columns")
        return self._future_columns(
            supports, queries, target_rows=target_rows)[..., :self.action_dim]

    def _g_lookup_projection(self, queries):
        return self.retrieval.project(self, queries)

    def _fit_action_ridge_lookup(self):
        return retrieval.fit_action_ridge(self, fitting.FIT_ROWS, fitting.progress)

    _metric_topk = retrieval.metric_topk

    _action_neighborhood_features = retrieval.future_action_features

    @torch.no_grad()
    def _project_hankel(self):
        self._support_screen = None
        self._rate_keys = None
        columns = len(self.hankel_starts)
        width = self.g_lookup_matrix.shape[1]
        self.pf = torch.empty(columns, width, dtype=self.dtype)
        for start in range(0, columns, fitting.FIT_ROWS):
            stop = min(start + fitting.FIT_ROWS, columns)
            rows = torch.arange(start, stop, device=self.device)
            projected = self._g_lookup_projection(self._past_columns(rows))
            self.pf[start:stop].copy_(projected.cpu())
            fitting.progress("lookup keys", stop, columns)
        self.pf_norm = torch.einsum("ij,ij->i", self.pf, self.pf).to(self.device)
        self.pf = self.pf.to(self.device)

    @staticmethod
    def _interpolate(source, positions):
        """Linear sampling; callers validate demonstration bounds in FP64."""
        lower, upper = positions.floor().long(), positions.ceil().long()
        weight = (positions - lower).to(source.dtype)[..., None]
        return source[lower] + weight * (source[upper] - source[lower])

    def _rate_target_overlap(self, first, starts, rate, target_rows):
        last = starts + self.history - 1 + (math.ceil(rate) if self.g_align_future else 0)
        return (first <= target_rows[:, None]) & (target_rows[:, None] <= last)

    def _trajectory_stops(self):
        return torch.cat([self.trajectory_starts[1:],
                          self.trajectory_starts.new_tensor([self.samples])])

    def _interpolated_keys(self):
        """Compact rate-specific keys, restricted to valid past/future windows."""
        if self._rate_keys is None:
            self._rate_keys = []
            for rate in self.g_lookup_rates:
                if rate == 1:
                    continue
                first = (self.hankel_starts + self.history - 1
                         - math.ceil(rate * (self.history - 1)))
                valid = first >= self.trajectory_starts[self.hankel_trajectory]
                if self.g_align_future:
                    stops = self._trajectory_stops()
                    last = (self.hankel_starts + self.history - 1
                            + math.ceil(rate * self.horizon) + self.g_state_offset)
                    valid &= last < stops[self.hankel_trajectory]
                columns = torch.nonzero(valid).ravel()
                keys = torch.zeros_like(self.pf)
                for start in range(0, len(columns), fitting.FIT_ROWS):
                    selected = columns[start:start + fitting.FIT_ROWS]
                    keys[selected] = self._g_lookup_projection(
                        self._past_columns(selected, rate=rate))
                self._rate_keys.append((rate, keys, keys.square().sum(1), first, valid))
                print(f"[lookup] rate={rate:g}: {len(columns)} valid endpoints", flush=True)
        return self._rate_keys

    def _rate_trace(self, queries, supports, target_rows=None):
        if getattr(self, "support_selector", None) is not None:
            return self.support_selector.rate_trace(queries, supports, target_rows)
        if len(self.g_lookup_rates) == 1:
            return {}
        keys = self._g_lookup_projection(queries)[:, None]
        ordinary = (self.pf[supports] - keys).square().sum(2)
        distances, rates = [ordinary], [1.0]
        for rate, plane, _, first, valid in self._interpolated_keys():
            excluded = ~valid[supports]
            if target_rows is not None:
                excluded = excluded | self._rate_target_overlap(
                    first[supports], self.hankel_starts[supports], rate, target_rows)
            distances.append((plane[supports] - keys).square().sum(2)
                             .masked_fill(excluded, float("inf")))
            rates.append(rate)
        return {"hankel_rates": queries.new_tensor(rates)[torch.stack(distances).argmin(0)],
                "hankel_unwarped_distance": ordinary.sqrt()}

    def _reconstruction_columns(self, supports, queries, dtype=None, target_rows=None, rates=None):
        if self.g_align_history and len(self.g_lookup_rates) > 1:
            if rates is None:
                rates = self._rate_trace(queries, supports, target_rows)["hankel_rates"]
            if self.g_native and supports.is_cuda:
                from .fused_temporal_history import reconstruction_columns
                plane = reconstruction_columns(self, supports, rates, dtype)
                if plane is not None:
                    return plane
        plane = self._past_columns(supports, dtype)
        if self.g_align_history and len(self.g_lookup_rates) > 1:
            for rate in self.g_lookup_rates:
                if rate != 1:
                    mask = rates == rate
                    if mask.any():
                        plane[mask] = self._past_columns(supports[mask], dtype, rate)
        return plane

    def _retime_action_values(self, values, rate):
        """Scale configured delta commands in raw units, leaving other controls alone."""
        if rate == 1 or not self.g_rate_scaled_action_dims:
            return values
        result = values.clone()
        dims = list(self.g_rate_scaled_action_dims)
        mean = self.action_mean[dims].to(values.dtype)
        scale = self.action_scale[dims].to(values.dtype)
        result[..., dims] = ((values[..., dims] * scale + mean) * rate - mean) / scale
        return result

    def _future_columns(self, supports, queries=None, target_rows=None, rates=None,
                        states=False):
        """Continue the selected past time map through the entire future horizon.

        Action timestamps are e + r*k, k=1..horizon; state timestamps retain
        state_offset. Optional delta scaling applies in raw action units. Rate-one
        action columns remain bitwise identical to the original Hankel readout.
        """
        if not self.g_align_future and not states:
            return self.uf.T[supports]
        if self.g_native and supports.is_cuda and self.g_align_future and len(self.g_lookup_rates) > 1:
            if rates is None:
                rates = self._rate_trace(queries, supports, target_rows)["hankel_rates"]
            from .fused_temporal_future import future_columns
            packed = future_columns(self, supports, rates, states)
            if packed is not None:
                return packed
        offsets = torch.arange(1, self.horizon + 1, device=self.device)
        endpoint = self.hankel_starts[supports] + self.history - 1
        source = self.state[:, :self.proprio_dim] if states else self.action
        shift = self.g_state_offset if states else 0
        result = (source[endpoint[..., None] + offsets + shift] if states else
                  self.uf.T[supports].reshape(*supports.shape, self.horizon, self.action_dim))
        if not self.g_align_future or len(self.g_lookup_rates) == 1:
            return result if states else result.flatten(-2)
        if rates is None:
            rates = self._rate_trace(queries, supports, target_rows)["hankel_rates"]
        result = result.clone()
        stops = self._trajectory_stops()
        for rate in self.g_lookup_rates:
            mask = rates == rate
            if rate == 1 or not mask.any():
                continue
            positions = endpoint[mask][:, None] + rate * offsets.double() + shift
            upper = positions.ceil().long()
            if bool((upper[:, -1] >= stops[self.hankel_trajectory[supports[mask]]]).any()):
                raise ValueError("interpolated future crosses a demonstration boundary")
            values = self._interpolate(source, positions)
            result[mask] = values if states else self._retime_action_values(values, rate)
        return result if states else result.flatten(-2)

    def fit_g_lookup(self):
        return self.retrieval.fit(self)

    def _solve_affine(self, design, query, *, _deferred_errors=None):
        """Solve affine coordinates, retaining the original minimum-norm fallback."""
        from .affine import solve_coordinates
        fixed = self.fixed_shape_readout
        if _deferred_errors is not None and not fixed:
            raise ValueError('deferred affine guards require the untruncated fixed readout')
        if design.shape[-1] == 1:
            return query.new_ones((len(query), 1))
        output, good = solve_coordinates(design, query, fixed=fixed)
        if _deferred_errors is not None:
            # Graph callers check this before exposing output and replay on failure.
            _deferred_errors.append(~good.all())
        elif not good.all():
            rows = slice(None) if fixed else ~good
            fallback = design.to(query.dtype)[rows].double()
            transpose = fallback.transpose(1, 2)
            solved = self._solve_g_normal(
                transpose @ fallback,
                (transpose @ query[rows, :, None].double()).squeeze(2)).to(query.dtype)
            output[~good] = solved[~good] if fixed else solved
        return output

    def _solve_g_normal(self, gram, rhs):
        if self.g_solver == "exact":
            result_dtype = rhs.dtype
            operator = gram.double()
            count = rhs.shape[1]
            tolerance = max(
                count, self.history * (self.action_dim + self.state_dim)
            ) * torch.finfo(torch.float64).eps
            cutoff = operator.abs().sum(-1).amax(-1, keepdim=True) * tolerance
            # Minimum-norm fallback, including singular and repeated supports.
            gram, rhs = operator, rhs.double()
            mean = gram.mean(-1, keepdim=True)
            operator = gram - mean - mean.transpose(1, 2) + mean.mean(1, keepdim=True)
            operator = (operator + operator.transpose(1, 2)) / 2
            target = rhs - mean.squeeze(2)
            target = (target - target.mean(-1, keepdim=True))[:, :, None]
            values, vectors = torch.linalg.eigh(operator)
            inverse = torch.where(values > cutoff, values.reciprocal(), 0)
            correction = vectors @ (
                inverse[:, :, None] * (vectors.transpose(1, 2) @ target))
            correction = correction.squeeze(2)
            return (correction - correction.mean(-1, keepdim=True)
                    + 1 / count).to(result_dtype)
        lipschitz = torch.linalg.eigvalsh(gram)[:, -1].clamp_min(1e-8)
        coefficients = torch.full_like(rhs, 1 / rhs.shape[1])
        threshold = self.g_l1 / lipschitz
        for _ in range(self.g_iterations):
            proposal = coefficients - (
                (gram @ coefficients[:, :, None]).squeeze(2) - rhs
            ) / lipschitz[:, None]
            coefficients = self._affine_l1_prox(proposal, threshold)
        index = coefficients.abs().argmax(1, keepdim=True)
        coefficients.scatter_add_(
            1, index, 1 - coefficients.sum(1, keepdim=True))
        return coefficients

    @staticmethod
    def _affine_l1_prox(proposal, threshold):
        """Soft threshold with the exact sum-to-one multiplier."""
        count = proposal.shape[1]
        points, order = torch.cat([
            proposal - threshold[:, None], proposal + threshold[:, None]], 1
        ).sort(1)
        changes = torch.cat([
            torch.ones_like(proposal), -torch.ones_like(proposal)], 1).gather(1, order)
        slopes = changes.cumsum(1) - count
        first = (proposal - threshold[:, None]).sum(1) - count * points[:, 0]
        values = torch.cat([
            first[:, None], first[:, None] + (
                slopes[:, :-1] * points.diff(1)).cumsum(1)], 1)
        interval = (values >= 1).sum(1).sub_(1).clamp_(0, 2 * count - 2)[:, None]
        value = values.gather(1, interval).squeeze(1)
        slope = slopes.gather(1, interval).squeeze(1)
        offset = points.gather(1, interval).squeeze(1)
        offset += (1 - value) / slope.clamp_max(-1)
        offset = torch.where(first < 1, points[:, 0] + (first - 1) / count, offset)
        centered = proposal - offset[:, None]
        return centered.sign() * (centered.abs() - threshold[:, None]).relu()

    def _g_kkt(self, coefficients, gradient):
        if not self.g_l1:
            return (gradient - gradient.mean(-1, keepdim=True)).abs().amax(-1)
        active = coefficients.abs() > torch.finfo(coefficients.dtype).eps * 10
        stationarity = gradient + self.g_l1 * coefficients.sign()
        multiplier = -(stationarity * active).sum(-1, keepdim=True) / active.sum(
            -1, keepdim=True).clamp_min(1)
        return torch.where(
            active, (stationarity + multiplier).abs(),
            ((gradient + multiplier).abs() - self.g_l1).relu()).amax(-1)

    def _hankel_exclusion(self, starts, target_rows=None, target_demos=None):
        excluded = torch.zeros(
            len(target_rows if target_rows is not None else target_demos),
            len(starts), device=self.device, dtype=torch.bool)
        if target_rows is not None:
            excluded |= ((target_rows[:, None] - self.history <= starts[None])
                         & (starts[None] <= target_rows[:, None]))
        if target_demos is not None:
            columns = torch.searchsorted(self.hankel_starts, starts)
            excluded |= self.hankel_trajectory[columns][None] == target_demos[:, None]
        return excluded

    def _exclude_hankel(self, distance, starts, target_rows, target_demos):
        if target_rows is None and target_demos is None:
            return torch.zeros(len(distance), device=self.device, dtype=torch.long)
        mask = self._hankel_exclusion(starts, target_rows, target_demos)
        distance.masked_fill_(mask, float("inf"))
        return mask.sum(1)

    def _hankel_topk(self, queries, count, target_rows=None, target_demos=None, allowed=None):
        """Return the global nearest Hankel columns without copying the plane."""
        if self.g_lookup != "past":
            if self.g_lookup_matrix is None:
                raise RuntimeError(f"{self.g_lookup} lookup must be fitted before retrieval")
            if self.pf is None:
                self._project_hankel()
            queries = self._g_lookup_projection(queries)
        if self.pf is not None:
            distance = _squared_distance(queries, self.pf, self.pf_norm)
            for rate, keys, norm, first, valid in self._interpolated_keys():
                warped = _squared_distance(queries, keys, norm)
                warped.masked_fill_(~valid[None], float("inf"))
                if target_rows is not None:
                    warped.masked_fill_(
                        self._rate_target_overlap(
                            first[None], self.hankel_starts[None], rate, target_rows),
                        float("inf"))
                torch.minimum(distance, warped, out=distance)
            excluded = self._exclude_hankel(
                distance, self.hankel_starts, target_rows, target_demos)
            if allowed is not None:
                distance.masked_fill_(~allowed, float('inf'))
            values, columns = torch.topk(distance, count, largest=False)
            return columns, values.sqrt(), excluded

        excluded = torch.zeros(len(queries), device=self.device, dtype=torch.long)
        best_values = queries.new_empty((len(queries), 0))
        best_columns = torch.empty(
            len(queries), 0, device=self.device, dtype=torch.long)
        for start in range(0, len(self.hankel_starts), SEARCH_ROWS):
            stop = min(start + SEARCH_ROWS, len(self.hankel_starts))
            columns = torch.arange(start, stop, device=self.device)
            plane = self._past_columns(columns)
            distance = _squared_distance(
                queries, plane, torch.einsum("ij,ij->i", plane, plane))
            excluded += self._exclude_hankel(
                distance, self.hankel_starts[start:stop], target_rows, target_demos)
            if allowed is not None:
                distance.masked_fill_(~allowed[:, start:stop], float('inf'))
            local = min(count, stop - start)
            values, local_columns = torch.topk(distance, local, largest=False)
            values = torch.cat([best_values, values], 1)
            columns = torch.cat([best_columns, local_columns + start], 1)
            keep = min(count, values.shape[1])
            best_values, order = torch.topk(values, keep, largest=False)
            best_columns = torch.gather(columns, 1, order)
        return best_columns, best_values.sqrt(), excluded

    def _legacy_g_batch(self, count=None):
        """The former plane-bound batch, retained only as the exact fallback tile."""
        count = min(self.g_supports if count is None else count, len(self.hankel_starts))
        projected = self.g_lookup == "past" and self.pf is not None
        width = self.pf.shape[1] if projected else self.history * (
            self.action_dim + self.state_dim)
        itemsize = torch.empty((), dtype=self.g_fit_dtype).element_size()
        search = len(self.hankel_starts) if self.pf is not None else min(
            SEARCH_ROWS, len(self.hankel_starts))
        source_size = torch.empty((), dtype=self.dtype).element_size()
        solve = (0 if self.g_weights == "uniform" else
                 count * width * (source_size + 2 * itemsize)
                 + count**2 * itemsize)
        return max(1, G_PLANE_BYTES // (solve + search * (2 * itemsize + 9)))

    def _literal_g_supports(self, queries, count, rows, demos, batch):
        supports, excluded = [], []
        for start in range(0, len(rows), batch):
            stop = min(start + batch, len(rows))
            columns, _, removed = self._hankel_topk(
                queries[start:stop], count, rows[start:stop],
                None if demos is None else demos[start:stop])
            supports.append(columns)
            excluded.append(removed)
        return torch.cat(supports), torch.cat(excluded)

    def _temporal_support_topk(self, queries, count, rows, demos, legacy_batch):
        from .fast_neighbors import projection_error_bounds
        from .temporal_screen import TemporalTopK
        if self._support_screen is None:
            rates = [(self.pf, None, None, None)]
            rates += [(keys, valid, first, self.hankel_starts + self.history - 1
                       + (math.ceil(rate) if self.g_align_future else 0))
                      for rate, keys, _, first, valid in self._interpolated_keys()]
            self._support_screen = TemporalTopK(rates)
        # Bound coarse tiles, masks and indexing temporaries.
        block = max(1, (128 * 1024**2) // (len(self.pf) * 12))
        supports = torch.empty((len(rows), count), device=self.device, dtype=torch.long)
        removed = torch.empty_like(rows)
        certified = torch.zeros(len(rows), device=self.device, dtype=torch.bool)
        for start in range(0, len(rows), block):
            stop = min(start + block, len(rows))
            window = queries[start:stop]
            keys = self.g_lookup_scale * (window @ self.g_lookup_matrix)
            error = abs(self.g_lookup_scale) * projection_error_bounds(
                window, self.g_lookup_matrix)
            supports[start:stop], certified[start:stop], removed[start:stop] = (
                self._support_screen.select(keys, count, rows[start:stop],
                    None if demos is None else demos[start:stop], self.hankel_starts,
                    self.hankel_trajectory, self.history, reference_error=error))
        # Preserve the original GEMM shapes and tie ordering on fallback.
        pending = torch.nonzero(~certified).flatten()
        tiles = torch.unique(pending // legacy_batch).cpu().tolist()
        for tile in tiles:
            start, stop = tile * legacy_batch, min((tile + 1) * legacy_batch, len(rows))
            supports[start:stop], removed[start:stop] = self._literal_g_supports(
                queries[start:stop], count, rows[start:stop],
                None if demos is None else demos[start:stop], legacy_batch)
        return supports, removed

    def _g_support_topk(self, queries, count, rows, demos, legacy_batch):
        temporal = (getattr(self, 'g_temporal_screen', False)
                    and len(self.g_lookup_rates) > 1 and count == 64)
        single_rate = len(self.g_lookup_rates) == 1 and count == 200
        # These screens certify linear ridge projection error. Other retrieval
        # implementations retain their own projection through the literal path.
        if not ((temporal or single_rate) and self.g_lookup == "action_ridge"
                and retrieval.is_builtin_retrieval(self) and self.g_native
                and self.device.type == "cuda" and self.pf is not None
                and self.pf.dtype == torch.float64 and len(self.pf) > 1024):
            return self._literal_g_supports(queries, count, rows, demos, legacy_batch)
        if temporal:
            try:
                return self._temporal_support_topk(queries, count, rows, demos, legacy_batch)
            except (torch.cuda.OutOfMemoryError, ValueError):
                self._support_screen = None
                torch.cuda.empty_cache()
                return self._literal_g_supports(queries, count, rows, demos, legacy_batch)
        oom = False
        try:
            if self._support_screen is None:
                from .fast_neighbors import CertifiedTopK
                self._support_screen = CertifiedTopK(self.pf)
            from .fast_neighbors import projection_error_bounds
            keys = self.g_lookup_scale * (queries @ self.g_lookup_matrix)
            error = abs(self.g_lookup_scale) * projection_error_bounds(
                queries, self.g_lookup_matrix)
            supports, certified, _, removed = self._support_screen.select(
                keys, count, rows, demos, self.hankel_starts,
                self.hankel_trajectory, self.history, reference_error=error,
                ordered_contract=False)
            replay_batch = self._g_microbatch(count)
            tiles = torch.unique(
                torch.nonzero(~certified, as_tuple=False).flatten() // replay_batch)
            for tile in tiles.cpu().tolist():
                start, stop = tile * replay_batch, min(
                    (tile + 1) * replay_batch, len(rows))
                supports[start:stop], _ = self._literal_g_supports(
                    queries[start:stop], count, rows[start:stop],
                    None if demos is None else demos[start:stop], replay_batch)
            if bool((supports < 0).any()):
                raise RuntimeError("certified G support replay left unresolved rows")
        except torch.cuda.OutOfMemoryError:
            self._support_screen = None
            keys = supports = certified = tiles = None
            oom = True
        if oom:
            torch.cuda.empty_cache()
            return self._literal_g_supports(
                queries, count, rows, demos, legacy_batch)
        return supports, removed

    def _g_prior(self, plane, supports, queries, rows=None, rates=None, *,
                 futures=None, query_history=None):
        fixed_rates = future_queries = None
        if (futures is None and rates is None and plane is None and rows is not None
                and self.g_weights != "uniform" and self.g_align_history and self.g_align_future
                and len(self.g_lookup_rates) > 1 and len(rows) <= self._g_microbatch(supports.shape[1])):
            # These two decoders originally projected the identical FP64 query
            # with the identical batch shape. Share only in this case: a split
            # solve keeps its original per-microbatch rate decisions. Explicit
            # caller rates still describe futures only, as before.
            if query_history is None:
                query_history = self._queries(rows)
            future_queries = query_history
            fixed_rates = self._rate_trace(future_queries, supports, rows)["hankel_rates"]
            rates = fixed_rates
        if futures is None:
            if future_queries is None:
                if self.g_align_future and rows is not None:
                    if query_history is None:
                        query_history = self._queries(rows)
                    future_queries = query_history
                else:
                    future_queries = queries
            futures = self._future_columns(
                supports, future_queries, rows, rates)
        if self.g_weights == "uniform":
            coefficients = queries.new_full(
                supports.shape, 1 / supports.shape[1])
            prior = futures[:, 0].clone()
            prior[:, :self.action_dim] = futures[:, :, :self.action_dim].mean(1)
            return prior, coefficients
        if plane is None:
            fixed_args = {} if fixed_rates is None else {"rates": fixed_rates}
            # The solve input may already be FP32; temporal rate decisions must
            # retain the original history precision. Keep private overrides on
            # their existing signature and gathering behavior.
            if query_history is not None and getattr(self._fixed_g, "__func__", None) is Policy._fixed_g:
                fixed_args["query_history"] = query_history
            coefficients = self._fixed_g(supports, rows, **fixed_args)
            return (torch.einsum(
                "bkh,bk->bh", futures.float(), coefficients).to(self.dtype),
                coefficients)
        futures = futures.to(plane.dtype)
        Z = plane.transpose(1, 2)
        if self.g_solver == "exact":
            coefficients = self._solve_affine(Z, queries)
        else:
            queries = queries.to(Z.dtype)
            coefficients = self._solve_g_normal(
                Z.transpose(1, 2) @ Z,
                (Z.transpose(1, 2) @ queries[:, :, None]).squeeze(2)).to(futures.dtype)
        return (torch.einsum("bkh,bk->bh", futures, coefficients).to(self.dtype),
                coefficients)

    def _g_support_atlas(self, rows, count, leave_demo_out=False):
        supports, excluded = [], []
        for start in range(0, len(rows), G_SUPPORT_BATCH):
            selected = rows[start:start + G_SUPPORT_BATCH]
            queries = self._queries(selected)
            target_demos = self.trajectory_of[selected] if leave_demo_out else None
            columns, removed = self._g_support_topk(
                queries, count, selected, target_demos, self._legacy_g_batch())
            supports.append(columns)
            excluded.append(removed)
        return torch.cat(supports), torch.cat(excluded)

    def _offline_g(self, rows, leave_demo_out=False, atlas=None, *,
                   _collect_neighbors=True, query_history=None, _supports_validated=False):
        if getattr(self, "support_selector", None) is not None and atlas is None:
            queries = self._queries(rows) if query_history is None else query_history
            selected = self.support_selector.select(
                queries, target_rows=rows,
                target_demos=self.trajectory_of[rows] if leave_demo_out else None)
            prior = queries.new_empty(len(rows), self.horizon * self.action_dim)
            retained = rows.new_empty(len(rows))
            excluded = (torch.bincount(self.hankel_trajectory, minlength=self.demo_count)[
                self.trajectory_of[rows]] if leave_demo_out else
                torch.searchsorted(self.hankel_starts, rows, right=True)
                - torch.searchsorted(self.hankel_starts, rows - self.history))
            native_dispatch = getattr(self._offline_g, "__func__", None) is Policy._offline_g
            packed, validated = None, False
            if native_dispatch and all(columns.ndim == 1 and columns.dtype == torch.long
                                       and columns.device == rows.device for columns, _ in selected):
                from .ragged_supports import offline_supports_valid
                packed = RaggedColumns.from_rows([columns for columns, _ in selected])
                validated = offline_supports_valid(self, rows, packed, leave_demo_out)
            batched = None
            options = getattr(self, "target_execution_options", {})
            if native_dispatch and options.get("target_execution") == "batched":
                from .target_readout import offline_prior
                batched = offline_prior(self, rows, selected, queries,
                                        packed=packed, supports_validated=validated)
            if batched is not None:
                prior, retained = batched
                totals = getattr(self, "_target_readout_totals", {})
                for name, value in self._target_readout_last.items():
                    totals[name] = totals.get(name, 0) + value
                self._target_readout_totals = totals
            else:
                for indices, columns, _ in support_groups(selected, self.device, include_distances=False):
                    query_args = ({"query_history": queries[indices], "_collect_neighbors": False,
                                   "_supports_validated": validated} if native_dispatch else {})
                    values = self._offline_g(
                        rows[indices], leave_demo_out, atlas=(columns, excluded[indices]), **query_args)
                    prior[indices], retained[indices] = values[0], values[2]
            neighbors = None
            if _collect_neighbors:
                if packed is None:
                    packed = RaggedColumns.from_rows([columns for columns, _ in selected])
                neighbors = RaggedColumns(self._support_endpoints(packed.values), packed.offsets,
                                          _max_count=packed.shape[1])
            return prior, excluded, retained, neighbors
        support_count = (min(self.g_supports, len(self.hankel_starts))
                         if atlas is None else atlas[0].shape[1])
        legacy_batch = self._legacy_g_batch(support_count)
        packed = (self.g_native and self.device.type == "cuda"
                  and self.g_tolerance is None and self.g_weights != "uniform")
        solve_batch = len(rows) if packed or self.g_weights == "uniform" else legacy_batch
        projected = self.g_lookup == "past" and self.pf is not None
        plane_dtype = self.g_fit_dtype
        priors, exclusions, retained_counts, neighbors = [], [], [], []
        for start in range(0, len(rows), solve_batch):
            stop = min(start + solve_batch, len(rows))
            selected = rows[start:stop]
            queries = (self._queries(selected) if query_history is None else
                       query_history[start:stop])
            target_demos = self.trajectory_of[selected] if leave_demo_out else None
            if atlas is None:
                supports, excluded = self._g_support_topk(
                    queries, support_count, selected, target_demos, legacy_batch)
            else:
                supports, excluded = (value[start:stop] for value in atlas)
            if not _supports_validated:
                retained = self.hankel_starts[supports]
                if ((selected[:, None] - self.history <= retained)
                        & (retained <= selected[:, None])).any():
                    raise RuntimeError("target action entered an offline G window")
                if leave_demo_out and (self.hankel_trajectory[supports]
                                       == target_demos[:, None]).any():
                    raise RuntimeError(
                        "target demonstration entered a sequential G support")
            rates = None
            if (not packed and self.g_weights != "uniform" and not projected
                    and self.g_align_history and self.g_align_future and len(self.g_lookup_rates) > 1):
                # Reconstruction and future decoding consume the same original
                # query batch and exclusion rows in the materialized path.
                rates = self._rate_trace(queries, supports, selected)["hankel_rates"]
            plane = (None if self.g_weights == "uniform" or packed else
                     self.pf[supports].to(plane_dtype) if projected else
                     self._reconstruction_columns(supports, queries, plane_dtype, selected,
                         **({} if rates is None else {"rates": rates})))
            arguments = (plane, supports, queries.to(plane_dtype), selected)
            prior_args = {} if rates is None else {"rates": rates}
            if getattr(self._g_prior, "__func__", None) is Policy._g_prior:
                prior_args["query_history"] = queries
            prior, _ = self._g_prior(*arguments, **prior_args)
            priors.append(prior)
            exclusions.append(excluded)
            retained_counts.append((self.hankel_trajectory[supports]
                                    == self.trajectory_of[selected, None]).sum(1))
            if _collect_neighbors:
                neighbors.append(self._support_endpoints(supports))
        return (*map(torch.cat, (priors, exclusions, retained_counts)),
                torch.cat(neighbors) if _collect_neighbors else None)

    def fit_targets(self, folder, shard):
        return fitting.fit_targets(self, folder, shard)

    def fit(self, target_folder=None):
        return fitting.fit(self, target_folder)

    def fitted_payload(self, spec):
        return artifacts.fitted_payload(self, spec)

    def set_residual_action_source(self, source):
        """Select the neighbor action paired with the unchanged endpoint state."""
        if source not in {"native", "retimed_first"}:
            raise ValueError("unknown residual action source")
        residual = self.config.get("residual", {}) if hasattr(self, "config") else {}
        lookup = getattr(self, "residual_lookup", residual.get("lookup", "g_supports"))
        model = getattr(self, "residual_model", residual.get("model", "rff"))
        if source == "retimed_first" and (
                lookup != "g_supports" or model != "rff" or self.g_state_offset != 1
                or not self.g_align_future):
            raise ValueError("retimed_first requires aligned, independent G-support RFF correction")
        self.residual_action_source = source
        return self

    def load_fitted_payload(self, payload):
        return artifacts.load_fitted_payload(self, payload)

    def reset(self):
        self._retrieval_history = deque(maxlen=self.lookback if hasattr(self, "lookback") else 10)
        self._state_history = []
        self._action_history = []
        self.last_trace = None

    def select_method(self, method):
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")
        self.method = method
        self.reset()
        return self

    def _warmup_neighbors(self, raw_state):
        self._retrieval_history.append(torch.as_tensor(raw_state, dtype=self.dtype, device=self.device))
        history = self.scale_state(torch.stack(list(self._retrieval_history)))
        window = history.new_zeros(1, self.lookback, self.state_dim)
        window[0, -len(history):] = history
        neighbors, _ = self._legacy_retrieve(window, torch.tensor([len(history)], device=self.device))
        return neighbors[0]

    @property
    def batched_direct(self):
        return self.method in METHODS

    @torch.no_grad()
    def direct_actions(self, action_past, state_past, *, selected=None,
                       _prepared=None):
        """Shared single/batched BPC readout; histories are already standardized."""
        if selected is None and getattr(self, "support_selector", None) is not None:
            from .ragged_supports import direct_actions
            return direct_actions(self, action_past, state_past)
        query = (torch.cat([action_past.flatten(1), state_past.flatten(1)], 1)
                 if _prepared is None else _prepared[0])
        wanted = self.g_supports
        if selected is not None:
            supports, distances = selected
        else:
            supports, distances, _ = self._hankel_topk(
                query, min(wanted, len(self.hankel_starts)))
        if _prepared is None:
            rate_trace = self._rate_trace(query, supports)
            plane = self._reconstruction_columns(supports, query, rates=rate_trace.get("hankel_rates"))
            futures = self._future_columns(supports, query, rates=rate_trace.get("hankel_rates"))
        else:
            rate_trace, plane, futures = _prepared[1:4]
        prior, coefficients = self._g_prior(
            plane, supports, query, rates=rate_trace.get("hankel_rates"), futures=futures)
        base = prior[:, :self.action_dim].clone()
        neighbors = self._support_endpoints(supports)
        feature = None
        residual = torch.zeros_like(base)
        if self.method == "bpc":
            if getattr(self, "residual_action_source", "native") == "retimed_first":
                feature = self._query_features(
                    state_past[:, -1], neighbors, futures[..., :self.action_dim])
            else:
                feature = self._query_features(state_past[:, -1], neighbors)
            residual = feature @ self.residual_weights
        prior[:, :self.action_dim] += residual
        committed = prior[:, :self.action_dim]
        error = (plane.transpose(1, 2) @ coefficients[:, :, None]).squeeze(2) - query
        gradient = (plane @ error[:, :, None]).squeeze(2)
        trace = {
            **rate_trace,
            "hankel_distance": distances[:, 0],
            "g_uniform": coefficients.new_full((len(query),), self.g_weights == "uniform"),
            "hankel_supports": supports,
            "support_future_spread": (futures - futures.mean(1, keepdim=True)).square().sum(2).mean(1).sqrt(),
            "g_sum": coefficients.sum(1),
            "g_negative_mass": (-coefficients).relu().sum(1),
            "g_reconstruction": error.norm(dim=1) / query.norm(dim=1).clamp_min(1e-12),
            "g_objective": error.square().sum(1) / 2 + self.g_l1 * coefficients.abs().sum(1),
            "g_kkt": self._g_kkt(coefficients, gradient),
            "g_coefficients": coefficients,
            "predicted_proprio_z": torch.einsum(
                "bkhs,bk->bhs", (self._future_columns(
                    supports, query, rates=rate_trace.get("hankel_rates"), states=True)
                    if _prepared is None else _prepared[4]), coefficients),
            "predicted_action_z": prior.reshape(-1, self.horizon, self.action_dim),
            "base_z": base, "residual_z": residual, "prior_z": committed,
            "committed_z": committed,
            "refinement_invoked": query.new_zeros(len(query)),
            "rff_norm": feature.norm(dim=1) if feature is not None else query.new_zeros(len(query)),
        }
        return committed * self.action_scale + self.action_mean, trace

    @torch.no_grad()
    def action_for(self, raw_state, previous_action):
        """Apply the same direct readout to one observation/action history."""
        raw_state = np.asarray(raw_state, dtype=np.float64).ravel()
        previous_action = np.asarray(previous_action, dtype=np.float64).ravel()
        if raw_state.shape != (self.state_dim,) or previous_action.shape != (self.action_dim,):
            raise ValueError("policy received the wrong state or action width")
        self._state_history.append(self.scale_state(raw_state))
        self._action_history.append(torch.as_tensor(previous_action, dtype=self.dtype, device=self.device))
        record = getattr(self, "_record_history", True)
        if not record:
            del self._state_history[:-self.history - 1]
            del self._action_history[:-self.history - 1]
        if len(self._state_history) <= self.history:
            if self.method == "bpc_nnc":
                return previous_action.copy()
            neighbors = self._warmup_neighbors(raw_state)
            return (self.action[neighbors].mean(0) * self.action_scale + self.action_mean).cpu().numpy().astype(np.float64)
        action_past = (torch.stack(self._action_history[-self.history:]) - self.action_mean) / self.action_scale
        state_past = torch.stack(self._state_history[-self.history:])
        action, trace = self.direct_actions(action_past[None], state_past[None])
        self.last_trace = ({key: value[0].cpu().numpy() for key, value in trace.items()}
                           if record else None)
        return action[0].cpu().numpy().astype(np.float64)
