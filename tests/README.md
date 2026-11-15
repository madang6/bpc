# Tests

Run from the repository root with a runtime containing the project dependencies:

```bash
python scripts/run_tests.py --suite unit
python scripts/run_tests.py --suite integration
python scripts/run_tests.py --suite cuda
python scripts/run_tests.py                 # all groups; CUDA skips on CPU
python scripts/run_tests.py --list          # group and name of every case
```

The groups are disjoint: 213 unit, 72 integration, and 89 CUDA cases. They all
use standard `unittest`; ordinary discovery remains supported with
`PYTHONPATH=src:scripts python -m unittest discover -s tests`.

- **Unit:** component behavior, mathematical properties, validation, and
  bounded-work contracts. These are the default, undecorated tests.
- **Integration:** `@integration` marks workflows joining fitting, retrieval,
  actions, artifact persistence, or worker processes. Small synthetic data and
  lightweight simulator/encoder doubles keep these checks practical; they do
  not measure task success rates.
- **CUDA:** `@cuda` marks checks requiring actual GPU execution and takes
  precedence over integration classification. CPU/CUDA versions can share
  assertions while testing distinct numerical implementations. Explicitly
  selecting this group requires a CUDA-enabled runtime and visible GPU; native
  kernel tests also require the matching toolkit.

The runner discovers all modules before filtering, so every group uses the
same dependency environment. It rejects import errors and duplicate test IDs.

Shared fixtures belong in `support/`: policy/configuration factories, rollout
doubles, and numerical reference helpers. They return fresh mutable state and
must not import test modules, prototype scripts, or define `TestCase` classes.
Helpers used by only one test module should stay there. Keep each tested
behavior with its owning component; add integration checks for interactions.

Parameterize repeated scenarios with `subTest` or a shared assertion helper.
Keep expected numerical results independent of the implementation under test.
Assertions about internal calls belong only where call shape, work bounds,
resource lifetime, or fallback behavior is part of the required contract.

Tests cover the retained experiment implementation and launcher. The 61 cases
specific to retired prototypes, historical studies, optional plotting, and
factorial analysis were removed with those tools. One parameterized regression
check ensures every retained experiment configuration can analyze completed
episodes. Shared fixtures and independent numerical references remain for the
active implementation.

Tests exclusive to the removed encoders, frozen π₀.₅ experiment, and unused
density-search path are retired too. Parameterized boundary checks reject
retired options before worker launch; retained tests cover the shared numerical
helpers, mixed/fused search, and fine-tuned π₀.₅ comparator.
