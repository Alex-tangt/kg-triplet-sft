# 02 — Attribute student target-fit and sparse relationship extraction

**Status:** needs-triage
**Type:** research
**Related:** `01-teacher-gold-quality.md`; ADR-0004 (teacher/student prompt and protocol), ADR-0007 (referent evaluation), ADR-0009 (base baseline); `docs/2026-09-07-capacity-line-repair.md`.

## Problem

The open-line SFT demonstrably teaches entity extraction, but relationship
reproduction remains sparse. We have measured the failure on held-out passages;
we have **not** measured free-generation fidelity on passages seen during training.
Thus the owner's concern that the student may be learning the teacher target
inefficiently is plausible but not yet established as a train-fit result. Teacher
label quality is a separate issue (#01), not an explanation to assume for poor
student imitation.

## Reproducible observations

- Fixed stratified 200-passage, same chat/no-think inference and referent judge:
  base 0.6B entity micro-F1 0.2683 / edge F1 0.0236; masked 0.6B 0.6065 /
  0.0914; masked 4B with case-tolerant parsing 0.7010 / 0.2331.
  Sources: `outputs/referent_eval/report_qwen3-0.6b-base.json`,
  `report_qwen3-0.6b-masked.json`, and `report_qwen3-4b-masked-clean.json`
  in the same directory.
- On those 200 passages, teacher has 4558 relationships; masked 0.6B emits
  1308 (28.7% of teacher volume), including 46 passages with zero emitted
  relationships. Masked 4B-clean emits 3126. This is not "no learning";
  it is a sharply asymmetric entity/edge result.
- Decomposing 0.6B teacher-edge recall: 1887/4558 (41.4%) teacher edges
  have both endpoints referent-matched; 268/1887 (14.2%) of those score as
  matched directed edges. For 4B-clean the corresponding rates are 55.8%
  and 35.2%. Entity omission and failure to connect found entities are both
  material. The edge metric tests endpoint topology, **not** relation-description
  or strength correctness (`eval/referent_eval.py`).
- Exact-title matching already gives a conservative entity signal; on the
  699-passage reference run, entity micro-recall is 0.4358 by title versus
  0.5408 with thinking-mode referent pairing
  (`report_title.json`, `report_think.json`). Neither measure alone establishes
  train-set fit or which causal component failed.
- The teacher target merges r1 and continuation r2; the student generates one
  JSON response. The 10-passage prompt gate also shows lower single-shot
  coverage, but it changes prompt/serialization as well as round count and is
  too small to attribute the gap (`dataset/graphrag_batch.py`, ADR-0004).
- The first no-mask run and canonical masked run changed multiple axes together:
  effective batch 16→8, dropout 0.05→0.1, warmup 0.03→0.1, optimizer,
  template, packing, and masking (`outputs/adapters/qwen3-0.6b_{chat,masked}/run_config.json`).
  The recipe improved outcomes; masking's isolated causal contribution is unknown.
- The repair report's "prompt ≈10× JSON answer / 85–90% gradient" rationale is
  contradicted by the current 2575-row Alpaca file: median answer-to-prompt
  character ratio is 2.29; answers are longer in 2556 rows. In the three saved
  `mask_check.json` examples, active response labels occupy 64–80% of tokens.
  These samples do not establish the full-corpus token ratio, and token counts
  alone do not measure gradient contribution. Response-only masking remains
  the correct recipe; this specific explanation needs correction.
- Training evaluation is off and no seen-train inference report is saved. The
  masked receipt reads a lexicographically last checkpoint state at step 966
  (epoch 3) despite listing checkpoint 1610. Its `final_step_loss` is not a
  reliable final-training loss; the receipt bug does **not** imply training
  stopped at epoch 3 (`finetune/compshare/train_unsloth.py`).

## Diagnostic work, in order

1. **Measure fit before retraining.** Freeze prompt, no-think protocol, decode
   settings, parser policy, and judge. Score a preregistered, stratified sample
   of *seen* training passages and a matched sample of *unseen* test passages
   with the existing 0.6B and 4B adapters. Report entity exact-title and
   referent P/R, teacher/student relationship volumes, endpoint-match ceiling,
   conditional edge R/P, schema/parse failures, and breakdowns by source,
   label density, passage length, and output position. Never use the test
   sample for training or checkpoint selection.
2. **Separate protocol from student capacity.** On the same passages, compare
   teacher official-prompt r1 with official-prompt r1+r2, holding the model,
   prompt, serialization, and parser fixed. The existing derived-prompt
   single-shot gate is supporting context, not a one-variable ablation.
   Audit a small sealed human set or teacher repeat runs so label omissions
   and inconsistent targets are not mistaken for student failures (#01).
3. **Follow the result, not a preferred explanation.** If seen-train edge fit
   is already low, test generation/position effects and a true one-variable
   masking/optimization ablation with identical recipe otherwise; inspect
   checkpoint-level validation and correct the receipt. If seen-train fit is
   high but held-out fit low, prioritize data scale, label consistency, and
   distribution/generalization probes. Treat model-size trends as evidence,
   not proof of capacity alone (base weights and 4B parser differ).

## Exit criteria

- Publish the seen-vs-unseen and teacher-r1-vs-r1+r2 tables with fixed IDs,
  commands, artifact paths, and uncertainty where applicable.
- State which hypotheses the evidence supports or rejects; leave unresolved
  causes explicitly unresolved. Do not report a single root cause from a
  multi-axis recipe comparison or test-only metric.
- Correct the prompt/answer-length claim and receipt interpretation before
  citing training-efficiency or final-loss conclusions.
- Keep #01's independent teacher-gold audit open; do not use its existence to
  waive the student-fit measurement.

## Comments

- 2026-09-25 (discussion, owner): 教师数据质量确实存在问题，但不能因此忽视微调
  过程中暴露的其他问题。现有精确实体匹配和关系评测已经提供学习效果的证据；疑问是
  学生相对于教师目标的学习是否仍偏弱，以及原因究竟在数据、模型还是算法。
  “学习效率低”目前是待验证的推断，不应写成既定根因。
- 2026-09-25 (investigation): 修正此前表述：不是“不知道学生有没有学会”，
  而是“知道实体有显著增益、关系覆盖仍低，却尚不能归因”。0.6B 基座→微调
  的实体 F1 为 0.268→0.607，关系 F1 为 0.024→0.091；可评分教师关系
  中仍只有 14.2% 被连对。教师噪声审计与学生拟合诊断并行，不能互相替代。
- 2026-09-25 (proposed diagnostic gate, pending triage): 建议先用现有 adapter 做
  已见训练篇章与未见篇章的同协议对照，并把端点覆盖和条件关系召回分开；
  再做教师同提示词的单轮/双轮对照。没有这些判别实验前，不把“教师差”、
  “0.6B 太小”或“掩码修复”任何一项宣布为唯一根因，也不建议仅凭猜测
  启动新训练。旧报告的 10× 长度论证和“final loss”回执需要更正。
