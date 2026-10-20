# DeepSeek V4.1 B300 work log

## Continuation contract

Read the [implementation plan](deepseek-v41-b300.md), this log, the
[operator guide](../guides/deepseek-v41-b300.md), and repository `AGENTS.md` before
continuing. Preserve user edits. Update this document after meaningful
implementation or verification changes. Distinguish code implemented, CPU
verified, GPU unverified, and research deferred. Do not invent measurements.
The user requested CPU checks only for the current implementation work and no
calendar dates or delivery estimates in the plan.

## Repository and provenance

- Branch: `deepseek-b300`. The user requires branch names without the assistant name prefix.
- Base: `qimcis/tokenspeed` main, `4fd85df66b4e3c127a587943c783dbc9d941c87c`.
- Origin: `https://github.com/qimcis/tokenspeed.git`.
- Publication: the user authorized pushing `deepseek-b300` and requires that no
  pull requests ever be opened. Push branch updates only; do not create a PR.
- Upstream: `https://github.com/lightseekorg/tokenspeed.git`.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
- Model revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- Local Python environment: `.venv`, Python 3.12. Activate before tools/tests.
- Package-version capture is not an immutable dependency closure. Container and
  complete runtime reproducibility remain part of hardware qualification.

## Coordination and ownership

The user requested ten subagents to implement, deliberate, and review. Shared
working-tree ownership is divided as follows; the coordinating agent integrates
CLI changes and verifies the combined result.

| Agent | Responsibility |
| --- | --- |
| `deployment` | Typed recipes, topology constraints, launcher, provenance |
| `telemetry` | JSONL metric observation plugin |
| `checkpoint` | Checkpoint tensor inventory and CPU audit |
| `analysis` | Paired statistics, calibration, replay and cache economics |
| `workload` | HTTP workload runner and experiment conversion |
| `epd_tests` | Opt-in endpoint qualification suite |
| `protocol` | EPD wire validation and compatible parsing fixes |
| `model_audit` | V4.1 model integration and unsupported-path audit |
| `docs` | Plan, operator guide, and this continuation log |
| `reviewer` | Independent read-only review |

API owners coordinate with the CLI owner. Protocol/model reviewers exchange
findings with runtime and deployment owners; analysis/workload owners align the
experiment schema. No subagent should commit or publish independently.

## Foundation completed

- Pulled upstream source and fetched the requested fork's main; both resolved to
  the same base revision. Created the implementation branch without changing main.
- Audited existing V4.1 model, FlatKV geometry and recipe, Engram, DSpark Markov
  sampler, encoder service, PD launcher, EPD launcher, context-parallel guards,
  and tree-speculation restrictions.
- Saved the corrected plan and this separate persistent work log; added a linked
  operator guide with runnable CPU validation and explicit hardware instructions.
- Created an isolated CPU tooling environment with pytest, Pydantic, NumPy,
  requests, Pillow, and pre-commit. Added CPU PyTorch, Transformers and lightweight
  frontend dependencies for deeper import checks; vendor Triton remains unavailable.
  Installed this checkout editable without its GPU dependency set. No model weights
  were downloaded and no GPU jobs ran.

## Implementation record

| Component | Implemented scope | Evidence boundary |
| --- | --- | --- |
| Deployment recipes | Aggregate8, P4/D4, E2/P2/D4; strict typed TOML and argv manifest | Candidate configurations; no B300 fit or topology claim |
| Memory ledger | Known tensor and cache payloads with placement assumptions and explicit exclusions | Analytical lower bound; no measured allocator peaks |
| Launcher | Pinned checkpoint, hardware/geometry preflight, owned workers, logs and manifest | Process behavior can be CPU tested; serving needs hardware |
| Checkpoint inventory | Safetensors metadata audit without allocating weights | Stored bytes do not establish runtime residency |
| Experiment analysis | Complete task population, itemized cost, paired cluster quality/cost bounds | Conditional statistical evidence; no automatic release approval |
| Calibration and replay | Per-phase linear fit and FCFS role queues | No batching or admission model; extrapolation is flagged |
| Route and locality screens | Small-window exact future-route oracle and LRU hit curves | No online scheduler or GPU hot-row cache implemented |
| Frontend image positions | Decode language-model-only path retains physical image padding while preserving authored IDs | Pure padding tests pass; full frontend role tests require runtime dependencies |
| EPD transfer hardening | Compatible wire validation and drain-before-terminal fanout barrier | CPU fake-transport checks; no external cancellation/DMA or CUDA recovery proof |
| Endpoint qualification | Opt-in text/image/prefix/concurrency/streaming/tool smoke suite | Skipped without endpoint; no quality or recovery claim |
| Workload measurement | Explicit bounded HTTP requests and external grade/cost join into experiments | Closed-loop request driver, not an agent runner; no automatic quality claims |
| JSONL metrics | Bounded asynchronous per-process metric records and loss/error accounting | Not request-quality evidence; dropped records must be disclosed |
| Cache economics | Explicit serial remote-hit versus local-recompute costs | Excludes misses, overlap, and fleet effects |
| Documentation | Corrected plan, operator guide, continuation log | Describes implemented boundaries and remaining gates |

The component owners exchanged implementation and review findings, and the
coordinating agent integrated the CLI, schemas, and combined test suite. The
records below distinguish integrated CPU evidence from deferred hardware work.

## Findings and decisions

- V4.1 already exposes encoder-only construction and multimodal encoder specs.
  The missing integration evidence is a V4.1-specific EPD recipe and test lane.
- Existing V4.1 PD launcher is language-model-only and therefore cannot establish
  multimodal EPD correctness. Do not reuse that flag for an EPD prefill role.
- Existing generic EPD launchers force an intra-node Mooncake transport that the
  V4.1 PD launcher explicitly avoids. The specialized launcher removes inherited
  forcing flags rather than silently selecting that path.
- The upstream encoder ring can reserve substantial memory. Transport buffers
  belong in deployment metadata and memory accounting.
- Decode KVP with V4.1 PD and Engram/SWA tree verification remain unsupported.
  Preserve these constraints rather than enabling unqualified configurations.
- Model audit found decode language-model-only processing dropped image-padding
  positions needed to match prefill. The fix shares padding logic while keeping
  authored token IDs and excluding decode vision features. See the
  [model audit](deepseek-v41-b300-model-audit.md) for source evidence.
- EPD prefill still allocates the vision module. Removing it requires a dedicated
  remote-embedding consumer capability; `language_model_only` is not equivalent.
- Identical observed outcomes in a small paired sample do not prove parity.
  Conservative bounds retain uncertainty beyond percentile bootstrap diagnostics.
- FCFS replay, route-union reduction, and LRU hit curves are separate offline
  screens. None establishes end-to-end throughput or cost improvement.

## Verification record

- Git working tree was clean before branch creation.
- Source configuration and analytical byte counts were inspected; these are not
  allocation measurements.
- The initial CPU suite passed 37 tests before parallel implementation. This is
  a baseline result, not the final combined test result.
- Operator-guide `measure` and `grade` options were checked against actual CLI help.
- Operator-guide `plan` commands succeeded for all three recipes; generated output
  remained explicitly analytical and hardware-unvalidated.
- Analysis owner reports 18 CPU tests passed; telemetry owner reports 15 CPU tests
  passed. These component results do not replace combined verification.
- Endpoint suite CPU invocation collected 16 skipped tests because no endpoint was
  configured. The image-last case came from cross-agent model audit.
- Protocol owner reports 37 passing wire/fanout CPU tests, including real thread
  execution with a fake synchronous transport. External failures during DMA and
  CUDA copy-event recovery remain unvalidated.
- Independent reviewer inspected the plan, guide, log, metrics documentation, and
  wire contract and found no unsupported capability claim.
- Deployment owner reports 35 configuration/lifecycle tests passed and local
  checkpoint configuration validation succeeded for all three modes. The ports
  availability check is advisory; dynamic transport ports remain upstream-owned.
- Workload owner reports 24 CPU-local HTTP/parser tests passed; no external
  endpoint was contacted for those checks.
- Model owner reports 6 passing contract tests and 2 frontend regressions skipped
  because the local runtime lacks vendor Triton.
- Combined CPU verification passed **188 tests**, with **18 skipped**: 16 explicit
  endpoint checks and 2 full frontend regressions requiring vendor Triton. The
  same result held after repository formatters ran:

  ```bash
  source .venv/bin/activate
  PYTHONPATH=python pytest -q test/deployment
  ```

- `git diff --check` passed after formatting.
- The independent code reviewer reported no unresolved findings.
- `pre-commit run --all-files` passed on the frozen tree after automatic formatting.
  All hooks ran; hooks without matching files reported skipped.
- Final staged review found no unresolved actionable issues. All available reviewers
  were GPT-family agents; this is independent review, not cross-family review.
- No B300 execution, performance benchmark, or model-quality evaluation has run.

## Remaining acceptance and continuation

1. Preserve the recorded CPU checks and review evidence. Resolve the runtime
   dependency limitations before running the two skipped frontend regressions.
2. Provision an explicitly authorized B300 environment with a pinned runtime
   closure, then qualify aggregate/PD/EPD correctness and recovery.
3. Collect real per-rank peak memory, bandwidth, collective, route, Engram reuse,
   queue, and speculative-round observations.
4. Evaluate topology under equal offered workload and whole-system cost; retain
   failures, rejections, retries, and idle resource allocations.
5. Implement conditional scheduling, caching, or kernel changes only after their
   mechanism and correctness gates are supported by evidence.
6. Keep model co-design separate from engine-only evaluation. Router changes
   require separately versioned artifacts and independent quality acceptance.

The implementation is tracked on branch `deepseek-b300`; use
`git log -1 --oneline` to identify its current commit. Its authorized remote
destination is `origin/deepseek-b300`. The user prohibits opening pull requests.
CPU completion must never be reported as full hardware or optimization-plan
completion.
