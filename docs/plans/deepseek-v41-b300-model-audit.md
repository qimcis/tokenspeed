# V4.1 model integration audit

This is a source audit of the pinned implementation, not B300 or checkpoint
execution evidence. The [implementation plan](deepseek-v41-b300.md) and
[continuation log](deepseek-v41-b300-work-log.md) track the broader work.

## Corrected defect: decode-side physical image history

`InputProcessor.tokenize_one_request` previously discarded precomputed multimodal
metadata immediately for a language-model-only decode worker. Prefill replaced
image spans with out-of-vocabulary content IDs, but decode retained authored
placeholder IDs. `ngram_inputs_for_forward` seeds Engram from the last three
physical predecessor IDs, so generation next to an image could use different
history on the two workers. Remote KV transfer does not fix those host-side IDs.

The decode branch now resolves the same item pad values and rewrites the same
inclusive spans as prefill, preserves the authored IDs for detokenization, and
still omits multimodal features from the tokenized decode request. The common
`pad_input_tokens` implementation now lives in the dependency-light
`runtime/multimodal/token_padding.py`; its old embedder import remains available.

CPU tests in `test/deployment/test_deepseek_v41_contract.py` exercise inclusive
sentinel spans, multiple images, repeated image occurrences, image-adjacent
physical history, and unchanged text. A regression through the real frontend
checks both roles when the normal runtime dependencies are available. It is
explicitly skipped when `tokenspeed_triton` is unavailable; the CPU span tests
are not claimed as end-to-end Engram validation.

## Existing contracts verified in source

| Concern | Source and conclusion |
|---|---|
| Image sentinels | `models/deepseek_v41_vision.py` validates types as start, per-row patch/newline runs, and end. `embed_media` outputs the entire sentinel-bearing block in item order. E-to-P token counts must include these rows, not just aligned patches. |
| Image masks | `models/deepseek_v41.py::_image_mask_from_spans` intersects inclusive image spans with each extend chunk and leaves decode/padding rows false. It uses spans because execution already clamps physical placeholder IDs. The existing `test_deepseek_v41_inputs.py::test_image_mask_survives_input_clamp` covers chunk intersections. |
| Vision routing | `DeepseekV41MoE._routing_inputs` selects `bias_vl` only for image rows. `DeepseekV41ForCausalLM` constructs and loads those biases when multimodal execution is active. Prefill must keep this support even when embeddings arrive from E. |
| Encoder allocation | `configs/model_config.py` marks encode workers `encoder_only`; `BaseCausalLM` then omits language-model construction. The V4.1 adapter builds the vision tower for this role. |
| Decode allocation | Explicit `language_model_only` disables active multimodal execution and therefore vision-tower construction. This is already supported; a decode worker without that flag still allocates vision. The physical-history fix preserves image request identity without enabling the tower. |
| Prefill allocation | An active multimodal P worker still allocates its vision tower even when all image embeddings arrive remotely. Removing it needs a separate embedding-consumer capability that preserves the image-mask path and `bias_vl`; using `language_model_only` on P would be incorrect. No unverified role-specific model rewrite was introduced. |
| Engram ownership | `engine/scheduler_utils.py::ngram_inputs_for_forward`, `execution/input_buffer.py`, and `execution/runtime_states.py` use bounded physical-token seeds and accepted tails. Slot restore invalidates/reseeds history; it is not another KV group to copy blindly. |
| DSpark windows | `layers/attention/kv_cache/recipes/deepseek_v41.py` places stage context fields in the target SWA group. `execution/drafter/deepseek_v41_dspark.py` writes captured rows there and reads relative to the accepted anchor; no separate private context transfer is needed. |
| DSpark bootstrap | `pd/prefill_executor.py::_cache_decode` sends the first sampled token and candidate IDs. `pd/decode_executor.py` receives those alongside the cache transfer. Candidate restoration and retained SWA fields must both succeed before decode. |
| Prefix semantics | The V4.1 cache recipe marks SWA and compressor tails replayable while global fields remain reusable. PD applies transfer policies to those groups. A transferred retained tail is distinct from rebuilding a local prefix hit. |

## Remaining acceptance boundaries

- Run the real frontend regression and existing model/input/cache/DSpark tests
  in a complete runtime environment. The local CPU installation cannot import
  vendor Triton registrations, even when a particular test uses CPU tensors.
- Validate E-to-P sentinel ordering, actual image masks, and `bias_vl` numerics
  with the exact checkpoint. API color/OCR checks alone do not establish these.
- Compare aggregated and disaggregated generation for an image at the end of
  authored input, repeated/replaced images, chunked prefill, speculative
  acceptance/rejection, cancellation, and slot reuse.
- Check actual DSpark cache-field transfer and bootstrap candidates, including
  prefix hits, differing supported TP layouts, and worker failure. Source
  ownership is not proof of RDMA completion or GPU visibility.
- Bind encoder results and reusable prefixes to model, processor, format and
  tenant identities. Current content-derived pad IDs alone do not establish
  that complete namespace contract across independent deployments.
- Keep decode-side KVP under PD and V4.1 tree speculation disabled under their
  existing guards. B300 capacity, quality and performance remain unmeasured.
