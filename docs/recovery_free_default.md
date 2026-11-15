# Recovery-free full-data Fisher default

`configs/robosuite_fisher_sparsemax_local.json` now selects all eligible Fisher
histories and disables recovery. The existing standard residual corrector is
refitted normally; no whole-demonstration-exclusion corrector is substituted.
R3M and the affine continuation formula remain unchanged. A distinct fit-cache
namespace and fit-spec flag prevent reuse of capped-fit correctors.

Reference: `dev-final-consolidated@964259cc24e13ebc177fd45e38d3d860240b5a19`
(the policy-module rename inherited by this repository). The prior capped-fit,
recovery-on recipe remains in `robosuite_fisher_sparsemax_capped_recovery_legacy.json`.
Other ablation entrypoints retain their historical settings; they are not a
matched ablation of this new default.

## Fitting

The original 8,192-anchor permutation still defines the RFF initialization and
the following 64 calibration identities. It no longer caps the Fisher fit.
Every eligible history contributes uniform mass to the original future-action
sparsemax teacher and Fisher covariances. Calibration excludes the query's
whole demonstration from candidates; its histories are included in the metric
fit, so it is not held-out validation. Zero-mass classes contribute zero.

The streamed fitter is ported from the evaluated `fit_all_windows.py`: disk-backed
FP64 features, exact sparse teacher memberships, blocked covariance accumulation,
and the regularized generalized eigenproblem, retaining rank 70 and shrinkage 1.
The standard `Policy.fit()` residual procedure is unchanged.

Full fitting requires substantial temporary disk and host memory (about 45 GB
for the largest feature array alone). The Docker launcher places temporary
arrays in the mounted experiment cache; native users should set `TMPDIR` to a
sufficiently large writable filesystem. Temporary directories are unique and
removed on normal completion. SciPy and threadpoolctl are required (available
in the experiment environment). This is not the capped-fit runtime claim.

## Evidence and limits

The frozen policies were evaluated on fresh seeds 5140000–5140299, 300 episodes
per task per arm, using the same 1,000-demo bank and fitted policy between arms.
All 4,800 episodes and paired initial-state/warmup/configuration checks completed.
Recovery off: 1760/2400 (73.33%); recovery on: 1747/2400 (72.79%). Difference
+0.54 percentage points, paired stratified 95% interval [-1.17,+2.25]. This is
not a formal retention/noninferiority result. Historical capped-fit scores are
not matched comparators. Per-task results: [CSV](recovery_free_results.csv).
Fitted bank/policy identities: [JSON](recovery_free_artifacts.json).

The source experiment, protocol, detailed interpretation, observer audit and
reproduction commands are preserved locally at:
`/home/admin/StanfordMSL/pi05-geometry-analysis/failure_transfer/eight_task_matched_recovery/`.
This integration has small-fixture mathematical/configuration checks; the
4,800 rollouts evaluated the preserved source artifacts, not a newly fitted
campaign on this repository commit. The numerical backend remains consequential.

## Behavior beyond the three component names

- **Temporal retiming:** candidate rates [0.5, 0.75, 1, 1.5, 2], interpolation of
  histories and futures, and scaling the first six action coordinates while
  retaining the gripper command. This is a substantive retrieval/continuation
  extension that the manuscript does not currently describe.
- **Support calibration:** 64 queries choose a sparsemax scale by future-action
  teacher coverage. This is part of retrieval but must be specified to reproduce it.
- **Demonstration warmup:** five demonstrated actions for Square, ten for the
  other tasks, establish the initial history. It is an evaluation protocol
  advantage/assumption, not one of the three predictive components.
- **Execution:** actions are clipped to [-1,1], and executed actions enter the
  next history. The 600-step total budget and success-based termination are
  evaluation choices. Mixed-precision retrieval refinement, paging and batching
  are computational implementations, not extra learned objectives; numerical
  trajectory divergence still prevents identifying individual causal rescues.
- **Recovery:** the code remains available explicitly, but its trigger, opening
  intervention and candidate mask are disabled in this default. No replacement
  gripper-voting or manipulation-phase heuristic was introduced.

Run the default with:

```bash
python3 scripts/run_experiment.py configs/robosuite_fisher_sparsemax_local.json --detach
```

## Integration checks

The 215-test unit suite passes in the evaluated simulator image
`sha256:e11bfccc365a4d9c067098142c2011285cf878a0b43ef0f14b704210ca91a872`.
Ten targeted full-Fisher, configuration-inheritance and existing LDA checks also
pass. New checks compare sparse full-data covariance/eigenmetrics with the dense
Fisher solution and exercise all-history selector fitting plus the resolved
default fit specification. These checks do not constitute new policy rollouts.
