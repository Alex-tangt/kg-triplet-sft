# 08 — Capacity runs (Qwen3-1.7B / Qwen3-4B)

**What to build:** The two additional capacity-curve points on the GraphRAG 2575 set.
Qwen3 has no 1.5B/3B, so per ADR-0006 the points are **Qwen3-1.7B and Qwen3-4B**
(base). Config identical to the bf16 curve line (ADR-0006) except the base-model
field — Unsloth bf16 on one CompShare RTX 4090 (`finetune/compshare/train_unsloth.py`).
Adapters + merged models pushed/mirrored for the eval thread.

**Superseded by:** ADR-0006 (sizes, framework, precision, platform)
**Deferred:** (none — ablation is issue 09)

**Blocked by:** 05, 06

**Status:** done (2026-09-05: 0.6B bf16 anchor, 1.7B, 4B all trained on CompShare 4090)

- [ ] A config-diff check that only --model / --out differ across the three runs.
      **NOT run — falsely ticked 2026-09-05.** Actual differences: eval on/off
      (0.6B/1.7B yes, 4B no) and checkpoint cadence; the optimizer-relevant config
      (bs2/ga8/6144/bf16/r32/lr5e-5/cosine/5ep/seed42) is identical across the
      three, but the training FORMAT deviated from the reference line (plain vs
      chat) — that protocol deviation, not a config-diff, is what invalidates the
      curve (see Finding below).
- [x] 1.7B completes on 4090 within budget; loss curves logged (final 0.535)
- [x] 4B completes on 4090 within budget; loss curves logged (final 0.467)
- [x] Adapters mirrored locally (`outputs/adapters/`) for 0.6B/1.7B/4B
- [ ] Pull merged models (~12 GB) or push adapters to Hugging Face (per ticket)

## Finding 2026-09-06 — capacity eval protocol mismatch (OPEN, blocks reading the curve)

**One-liner.** The capacity line (0.6B/1.7B/4B, bf16 CompShare-4090 / Unsloth) was
trained with **plain Alpaca concatenation** (`train_unsloth.py` `build_text`:
`instruction\npassage\n\noutput`, no chat markers), but the 200-sample inference
reused the reproduction line's **chat-template protocol** (`infer_full.py` family,
`apply_chat_template(enable_thinking=False)` + schema-prefix). Reference 0.6B is
chat-trained + chat-evaluated (self-consistent); capacity models are plain-trained +
chat-evaluated (mismatch). Result: the eval gap is a mix of real capability + protocol
penalty, and the penalty is size-asymmetric (bigger models tolerate format shift
better) — so the capacity-curve **slope may be flattened**, not just absolute values
shifted.

**Why "base models have none" (ADR-0006) was wrong.** The uploaded adapters'
`base_model_name_or_path` is `Qwen/Qwen3-0.6B` etc., but the mirrored
`outputs/adapters/*/merged/` tokenizers carry `chat_template.jinja`, thinking tokens,
and `eos=<|im_end|>` — i.e. training actually loaded the chat-capable main release
(= Kaggle slug `qwen-lm/qwen-3/transformers/{0.6b,1.7b,4b}`, NOT `-base`). So the base
COULD have wrapped chat markers; the plain format was a choice, not a constraint.

**Measured on cap06 (bf16/4090) vs reference 0.6B (fp16/T4), same 200-sample referent eval:**
entity recall 0.534→0.440, precision 0.723→0.600, F1 0.614→0.508; schema compliance
0.931→0.808; type OOV 5→155, `-` placeholder 0→144, empty description 18→383; student
exact-dup 0.33%→1.65% (5.28% in the first pass was an artifact of blank
placeholder titles normalizing onto one shared key; fixed in referent_dup); unresolved endpoints 298→598; hallucination 0.142→0.257;
numeric retention ~flat (0.578/0.584). The schema/OOV/dup/endpoint degradation is the
signature of a plain-trained model seeing a chat start token it never saw + no learned
EOS (runaway/repetition), NOT ordinary capability loss.

**Next actions (owner-approved 2026-09-06).**
1. Chat-protocol full runs of cap17/cap4b were stopped (sunk GPU; outputs discarded).
2. cap06 **plain-protocol** full-200 A/B (~45 min GPU) → quantify the penalty on the
   API-free axes (schema/OOV/`-`/empty-desc/exact-dup/unresolved-endpoints) vs the
   chat-protocol file already at `outputs/capacity_eval/qwen3-0.6b/predictions.jsonl`.
3. If plain recovers materially, re-run all three capacity models in plain protocol for
   the real curve; fold a rule into ADR-0006 ("capacity-line eval must match its
   training format") and correct the "base models have none" claim there.

**A/B result (2026-09-07, plain cap06 full-200 = `outputs/capacity_eval/qwen3-0.6b-plain/`).**
Plain protocol vs chat protocol, same model (cap06/bf16-4090), API-free axes on raw
predictions (NO referent pairing — free axes only):
- entities 3635→**7114**; relationships 1526→**2734**
- empty descriptions 383→**0**; entities missing any key 383→**0**
- same-row duplicate entities 144→**33**
Chat protocol strongly suppresses the plain-trained model (withheld extraction,
empty-field placeholder rows); plain recovers volume and schema discipline. This
confirms the issue-08 Finding: the earlier referent table (recall 0.534→0.440,
F1 0.614→0.508, schema 0.93→0.81) was **not** a clean model-capability gap — a
large share is chat/plain protocol mismatch.

**Cost warning for plain reruns.** The plain cap06 run took **~5.6 h wall** (kernel
internal 20039 s, first→DONE log timestamps) vs ~43 min chat, because plain prompts
carry no chat stop marker: raw decode length median ~33.5k chars (≈ max_new_tokens
8192), i.e. nearly every passage generates to the cap (runaway decode; 200/200 rows
completed, NOT a stall). A plain rerun of all three sizes at this budget is
impractical. Before re-running 1.7B/4B in plain, must fix the stop signal — options:
(a) bake a plain EOS / truncate decode earlier (the tolerant parser only needs the
JSON object, so decoding to 8192 is wasteful), (b) add `early_stopping`/smaller
max_new_tokens for the JSON tail, (c) re-examine whether the full referent evaluation
even needs whole-passage reruns vs a delimited prompt. Reference-consistency note:
reference 0.6B is chat-trained+chat-eval (self-consistent); if capacity must compare
against it on the referent axes, that comparison is inherently cross-protocol and
needs an explicit call-out, not a silent assumption.

Context: `train_unsloth.py` (plain concat), `lffull.py:111` (template `qwen3_nothink`,
chat), kernel_inferfull/cap_infer (chat eval protocol). See `docs/adr/0006`, `0007`.

## Resolution 2026-09-07 — root cause is TRAINING (no EOS), decision = chat-format retrain (OWNER-APPROVED)

**Root cause (re-attributed).** The runaway decode is NOT an inference-protocol
artifact and NOT a matching error in the plain inference path (the plain concat is
byte-identical to `train_unsloth.py` `build_text`). It is a **training defect**: the
plain format packs samples with no per-sample EOS (`packing=True`, EOS only at the
packed-sequence end, ADR-0006), so the model never learns to stop after one output.
Evidence — same 95 passages, cap06:
- chat protocol: 2/95 rows >10k chars (median 4.3k) — base's own `<|im_end|>` EOS
  (151645) stops decode;
- plain protocol: 88/95 rows >10k chars (median 34k) — no learned stop → generates
  to the 8192 cap; 26/119 (22%) never even close their braces, and those 22% cannot
  be salvaged by truncation (empty/fragment parse).
So plain was never a viable eval line for these models regardless of early-stop
engineering; the cost (5.6h/200, fragile `_JsonClose` criteria, runaway rows) is the
training defect surfacing, not a fixable inference problem.

**Schema-prefix injection is protocol-agnostic** (both chat and plain kernels force
`{"entities": [`); it was NOT the differentiator. The differentiator is the stop
signal, which only chat prompts inherit (from the base's chat template/EOS).

**Decision (owner-approved 2026-09-07): retrain the capacity line in CHAT format** —
keep the Unsloth/4090 stack but render each Alpaca sample through the Qwen3 chat
template (im_start/im_end markers, thinking OFF) so the model learns to end its output
with `<|im_end|>`: train = eval = serve = deploy format, self-consistent like the
reference 0.6B line, stopping for free, no early-stop hack, comparable to the fp16
anchor. ~5.5h / 11 CNY on the CompShare 4090 (cost accepted). Option (b): modify
`train_unsloth.py` text construction to apply the chat template (do NOT switch to
LLaMA-Factory); verify whether unsloth `packing=True` still omits per-sample EOS and
whether chat im_end then survives packing — if not, the minimal change is appending
the EOS/im_end token per sample instead of a full chat re-template.

**Superseded.** The 2026-09-06 "Next actions" item 3 ("re-run in plain if it
recovers") is void — plain cannot be the eval line. The early-stop prototype
(`cap_infer.py` EARLY_STOP / `_JsonClose` / `_truncate_at_json_close`, kernel
`kg-tracer-cap06ps`) is shelved for the capacity curve; may be reused only for a
future plain-trained model whose training DID include a stop token.

**Done artifacts.** A/B + redundancy + dup analyses live in
`outputs/capacity_eval/` (`redundancy_{reference,cap06_chat,cap06_plain}.json`,
`dup_{reference,cap06_chat,cap06_plain}.json`, `qwen3-0.6b[-plain]/predictions.jsonl`).

## Resolution 2026-09-07b — deeper root cause is LOSS COVERAGE; canonical masked recipe validated (OWNER-APPROVED)

**One-liner.** Retraining in chat alone did NOT close the gap to reference: the
0.6B chat-format run (think-preamble training, no mask) still scored ~0.44 referent
recall — identical to the plain-trained baseline. A recipe audit against the anchor
kernel (`finetune/kaggle/kernel_lffull/lffull.py`) found the real structural
difference: LLaMA-Factory SFT defaults to **response-only loss** (masks the prompt),
while every capacity run fed Unsloth a raw concatenated `text` field, which is a
**full-sequence continued-pretraining loss** — with the long prompt ≈10x the JSON
answer, ~85–90% of gradient trained prompt continuation, not extraction. This is why
two different text wrappers (plain/chat) scored identically: both lacked masking.

**Evidence chain (all on the same fixed 200-sample referent eval, micro metrics):**
| run | recall | precision | F1 | schema valid | halluc |
|---|---|---|---|---|---|
| reference fp16 (ref200 baseline) | 0.533 | 0.723 | 0.614 | 0.931 | 0.142 |
| plain-train / chat-eval (cap06) | 0.440 | 0.600 | 0.508 | 0.808 | 0.257 |
| chat-train think-preamble / no mask (0.6b_chat) | 0.419 | 0.557 | 0.478 | 0.836 | 0.321 |
| **canonical masked 0.6B (0.6b_masked)** | **0.536** | **0.699** | **0.607** | **0.919** | **0.166** |

The masked run meets the reference within the n=200 CI (±~2.7pt) — root cause
confirmed and the fix validated.

**Canonical masked recipe** (`finetune/compshare/train_unsloth.py`, one 4090):
no-think chatml render via messages→`apply_chat_template` (stock Qwen3 template
injects an empty `<think>` block the reference does not train with — override +
restore), response-only loss via
`unsloth.chat_templates.train_on_responses_only("<|im_start|>user\n",
"<|im_start|>assistant\n")`, eff batch 8 (bs2×ga4), warmup 0.1, dropout 0.1,
`adamw_torch`, bf16, lr 5e-5 cosine, 5 epochs, cutoff 6144, non-packed
(`UNSLOTH_RETURN_LOGITS=1` disables packing by design — dodges unsloth#5230 and
mirrors reference step semantics), `save_every` 322. Evidence per run:
`run_config.json` / `format_check.json` / `mask_check.json` / `receipt.json`.
0.6B full run: 1610 steps, final loss 0.355, ~50 min, ~2 CNY.

**Inference/kernel fixes folded in:**
- `cap_infer.py` tolerant parser now drops non-object entity/relationship rows on
  the strict-parse path too (wikipedia-02211 style bare-string rels crashed
  `referent_eval` and forced a 199/200 report); predictions are schema-clean
  (validated: 0 non-dict items across the 200).
- 4090 efficiency: `make_capinfer.py --batch N` (bake `BATCH_SIZE`); bs4→12 for
  0.6B on one 4090 ≈ 5x wall (GPU 24%→95%). fp16 keeps parity with the reference
  eval line (fp16 == bf16 on Ampere).

**Decision (owner-approved): the canonical masked recipe IS the capacity line.**
The 2026-09-07 "chat-format retrain" (Resolution 2026-09-07) produced
`0.6b_chat` (0.44 — think preamble + no mask, superseded); the validated recipe is
`0.6b_masked` (`outputs/adapters/qwen3-0.6b_masked/`, out
`outputs/capacity_eval/qwen3-0.6b-masked/`). Remaining curve work: train
**1.7B and 4B with the same canonical masked recipe** (~2.5h + ~5–6h on the 4090,
~16–18 CNY est) and infer via the parser-fixed kernel; then tick the open
checkboxes from actual run-dir artifacts.

Context: `finetune/compshare/train_unsloth.py` (canonical), `kernel_lffull/lffull.py`
(anchor), unsloth docs + issues #1017/#5230, `docs/2026-09-07-capacity-line-repair.md`.

## Run log 2026-09-07 evening — 1.7B masked trained, inferred, evaluated (issue 08 curve point 2/3)

**Training (CompShare 4090 `kg-tracer`, host run `out/1.7b_masked`, same canonical
masked recipe).** `train_unsloth.py --model /root/models/Qwen3-1.7B --train-jsonl
data/alpaca_full_train.jsonl --out out/1.7b_masked --bs 2 --epochs 5 --ga 4
--save-every 322 --optim adamw_torch --mask --merge`. PLAN + FORMAT_OK + MASK_OK +
full receipt logged in `run_17b_masked.log`; artifacts `run_config.json` /
`format_check.json` / `mask_check.json` / `receipt.json` in the run dir. 1610
optimizer steps (~322/epoch), trainable 34,865,152 / 1,755,440,128 (1.99%),
bf16, eff bs 8. **Wall ~1 h 37 m** (~3.5 s/step steady), peak VRAM ~9.5 GB.
Receipt: `adapter_saved: true`, `merged_saved: true`, 5 checkpoints (322…1610).
(receipt `final_step_loss` 0.2948 is the checkpoint-966 trainer-state snapshot —
same quirk as the 0.6B run; last-logged losses ran 0.20–0.29.) Data sha matches
the 0.6B masked card; diff-cards PASS vs 0.6B (only model/model_dir/out differ).

**Inference (same 4090, kernel derived from the parser-fixed `kernel_cap06m`).**
Host `infer_cap17m.py` = sed-clone of the 0.6B masked infer script with
`MODEL_SLUG=…1.7b/1`, `DATASET=kg-cap17b-masked-adapter`,
`WORK=/root/kg/infer_out_17m`, `BATCH_SIZE=8`, chat protocol, EARLY_STOP off.
`/kaggle` symlink tree: `models/…/transformers/1.7b/1 → /root/models/Qwen3-1.7B`,
`kg-cap17b-masked-adapter → out/1.7b_masked/adapter`; passages read from
`cap-upload/sample_200_passages.jsonl`. **200/200 rows, ~32 min** (GPU 96%,
~9.6 GB); predictions pulled to
`outputs/capacity_eval/qwen3-1.7b-masked/predictions.jsonl` (200 lines, schema-clean).

**Eval (200-sample referent, thinking judge; `report_qwen3-1.7b-masked.json`).**

| run | macro R/P/F1 | micro R/P/F1 | schema valid | halluc |
|---|---|---|---|---|
| reference (ref200) | 0.551 / 0.721 / 0.602 | 0.534 / 0.723 / 0.614 | 0.931 | 0.142 |
| 0.6B masked | 0.559 / 0.713 / 0.604 | 0.536 / 0.699 / 0.607 | 0.919 | 0.166 |
| **1.7B masked** | **0.608 / 0.714 / 0.609** | **0.590 / 0.597 / 0.594** | 0.844 | 0.234 |

**Curve reading.** Recall rises with size (micro 0.536→0.590; macro 0.559→0.608)
but micro precision falls 0.699→0.597 (>CI ±0.03) and **F1 is flat** (~0.60 for
every point including reference). Schema discipline and grounding degrade at 1.7B:
valid 0.919→0.844, empty_description 43→641, type OOV 3→16, hallucination
0.166→0.234 — the larger model over-produces (student entities 3805→4905, rels
1308→2315; items 7220) and long-decodes get truncated into schema-empty rows. Open
question for the curve: precision/schema cost may be a decode-length/verbosity
artifact (MAX_NEW_TOKENS 8192, verbose model truncation) as much as a capability
limit; a 4B point (or a length/`temperature` axis) would disambiguate. 4B masked
not yet run — the owner stopped the instance to digest this before spending the
~3–4 h / ~6–8 CNY a 4B run needs.

## Run log 2026-09-07 night — 4B masked + full curve + uppercase-key casing finding (curve point 3/3)

**Training (CompShare 4090 `kg-tracer`, host `out/4b_masked`, canonical masked recipe).**
Note: the pre-downloaded `/root/models/Qwen3-4B` was BROKEN (safetensors shards
failed to fetch in the original `dl_4b.sh` — xet 401, only `.index.json` present).
The real 4B base is the ModelScope image path `/model/ModelScope/Qwen/Qwen3-4B`
(the same path the original plain 4B run used). Run:
`train_unsloth.py --model /model/ModelScope/Qwen/Qwen3-4B --train-jsonl
data/alpaca_full_train.jsonl --out out/4b_masked --bs 2 --epochs 5 --ga 4
--save-every 322 --optim adamw_torch --mask --merge` — 1610 steps,
**~3 h 47 m wall** (~8 s/step, GPU 100%, VRAM ~15.5 GB), final losses
~0.17–0.24. Receipt: adapter + merged saved, 5 checkpoints. Same evidence set
(`run_config`/`format_check`/`mask_check`/`receipt`) + data sha identical.

**Inference (same 4090, `infer_cap4m.py` = kernel clone, BATCH_SIZE=4).**
Symlinks: `models/…/transformers/4b/1 → /model/ModelScope/Qwen/Qwen3-4B`,
`kg-cap4b-masked-adapter → out/4b_masked/adapter`. 200/200 done (~1 h; the
longest-tail rows dominate wall). Predictions (canonical parse) →
`outputs/capacity_eval/qwen3-4b-masked/predictions.jsonl`.

**Casing finding (important).** ~17% of 4B rows emit schema keys in UPPERCASE
(`"TITLE"/"TYPE"/"DESCRIPTION"`; raw examples in `raw_<id>.txt`), which the
tolerant parser drops (salvage keys on lowercase `title`/`source` only) →
those rows parse "failed" and land EMPTY in predictions. Offline re-parse of the
raws with case-normalized keys (`build` script, key regex → lowercase) recovers
them: empty 34→13 (10 remaining are genuinely unparseable output), entities
4100→4432 (+8%), rels 2758→3126. Reference/0.6B/1.7B never emit uppercase keys.

**Eval (200-sample referent, thinking judge).**

| run | micro R/P/F1 | edge F1 | schema valid | halluc | empty rows |
|---|---|---|---|---|---|
| reference (ref200) | 0.534 / 0.723 / 0.614 | 0.086 | 0.931 | 0.142 | 5 |
| 0.6B masked | 0.536 / 0.699 / 0.607 | 0.091 | 0.919 | 0.166 | 5 |
| 1.7B masked | 0.590 / 0.597 / 0.594 | 0.160 | 0.844 | 0.234 | 0 |
| 4B masked canonical | 0.595 / 0.720 / 0.651 | 0.217 | 0.941 | 0.099 | 34 |
| **4B masked clean (case-tolerant)** | **0.664 / 0.743 / 0.701** | **0.233** | **0.953** | **0.084** | 13 |

**Final curve reading.** Recall rises monotonically with size (0.536→0.590→0.664);
in the case-tolerant (consumer-robust) reading F1 also rises to 0.701 — the
earlier 1.7B "bigger = sloppier" impression was size-specific (1.7B over-produces,
4B does not). 4B clean beats the reference 0.6B on every axis including schema
(0.953 vs 0.931) and hallucination (0.084 vs 0.142). The **canonical (strict)
numbers are the deploy-true ones** for a consumer that requires lowercase keys;
the clean numbers quantify the ceiling once a tolerant extractor (case-insensitive
key normalization) absorbs the casing quirk — worth an ADR if deploy adopts it.
Cost this leg: ~4B train ~3h47 (~8 CNY) + infer ~1 h (~2 CNY) + two 200-pairing
eval passes.

## Comments
