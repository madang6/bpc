# Radius-normalized sparsemax Fisher

```bash
python3 scripts/run_experiment.py configs/robosuite_fisher_sparsemax_local.json --detach
```

The headline entrypoint inherits shared settings from `robosuite_common.json`.
Its default configuration uses the full-data recovery-free update: arm plus
binary gripper supervision, 16,384 retrieval RFFs, all eligible fitting histories,
Fisher rank 70, individual-history candidates, query-density normalization,
teacher-coverage calibration, and adaptive sparsemax support. Compact rollout
traces remain disabled to match the recorded campaign.

For squared future-action distances `d` and the per-anchor radius `r`, the
teacher is `sparsemax(-d / r)`, with overlapping training windows excluded. The
same conversion is used for coverage calibration; held-out queries exclude
their whole demonstration. Teacher normalization and inference density
normalization are separate operations.

Classes with zero teacher mass contribute zero to Fisher covariance; their
recorded histories remain eligible retrieval candidates. Host teacher storage
bounds GPU memory to one class block while preserving the covariance reductions.
Only the radius-normalized sparsemax Fisher construction is selectable.

The recipe retains exact affine continuation and the shared-support RFF
correction, temporal retiming, coverage calibration, and
existing execution accelerations. See the [README](../README.md) for their
scope and the manuscript ablation entrypoints.

See [the new default and matched results](recovery_free_default.md). The legacy
capped-fit recovery-on config is preserved explicitly as
`robosuite_fisher_sparsemax_capped_recovery_legacy.json`.

## Historical capped-fit recovery-on results

The completed local/Endor campaign ran source `0c8975d`, with simulator image,
original configuration, per-task policy/bank hashes, and paired outcomes
recorded in [`fisher_sparsemax_evaluation.json`](fisher_sparsemax_evaluation.json).
Each task has 300 episodes, with seeds 100000–100299 and 1,000-demo banks.

| Task | Radius-normalized sparsemax |
|---|---:|
| Square | 218 / 300 |
| Threading | 231 / 300 |
| Coffee | 285 / 300 |
| Stack | 284 / 300 |
| Nut Assembly | 35 / 300 |
| Mug Cleanup | 237 / 300 |
| Stack Three | 147 / 300 |
| Hammer Cleanup | 297 / 300 |
| **Total** | **1,734 / 2,400 (72.25%)** |

`dev-fast-consolidated@48bc022` integrated this recipe; it did not run another
full rollout campaign. Its reported integration checks compared resolved method,
evaluation, and all eight task fit specifications with the recorded configuration
(after remapping data locations). Seeded CPU/CUDA probes matched the original
radius teacher's projection, calibration, selected columns, distances, and
reconstruction outputs. Those checks validate the tested fixtures, not another
full-bank campaign.

These historical results do not measure the new full-data default.

Historical execution studies remain in Git history at
`2cb8291:docs/execution.md`, alongside the standalone development tools.
They are not required to run the manuscript experiment entrypoints.
