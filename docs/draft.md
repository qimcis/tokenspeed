# V4.1 B300 kernel task draft

The executable task contract and implementation draft are in
[the performance implementation plan](plans/deepseek-v41-b300-performance.md),
sections C1, C2 and Kernel task contract. This draft exists before candidate work.

Baseline: upstream DSpark tile-candidate all-gather and exact host Engram gathers.
Candidates: compact packed-max candidate communication; fused cached Engram
lookup/gather/dequantization. Keep baseline available and candidates opt-in.

Risks: changed argmax tie behavior, packed-key special values, asynchronous source
ownership, UVA miss ordering, cache invalidation, graph pointer stability and
extra-kernel overhead. CPU references do not prove CUDA compilation or speed.

Local checks: activate `.venv`, run `PYTHONPATH=python pytest -q test/performance`,
then the named native scheduler tests and full repository hooks. GPU qualification
commands and candidate benchmark artifacts will be supplied by the implementation
owners and remain opt-in. No GPU execution is authorized in this work session.

Promotion requires compiled numerical and graph tests plus measured whole-path
and task-level benefit. Unmeasured candidates remain experimental.
