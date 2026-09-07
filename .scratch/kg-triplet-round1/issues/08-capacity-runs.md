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

## Comments
