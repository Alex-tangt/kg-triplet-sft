# 08 — Capacity runs (Qwen3-1.7B / Qwen3-4B)

**What to build:** The two additional capacity-curve points on the GraphRAG 2575 set.
Qwen3 has no 1.5B/3B, so per ADR-0006 the points are **Qwen3-1.7B and Qwen3-4B**
(base). Config identical to the bf16 curve line (ADR-0006) except the base-model
field — Unsloth bf16 on one CompShare RTX 4090 (`finetune/compshare/train_unsloth.py`).
Adapters + merged models pushed/mirrored for the eval thread.

**Superseded by:** ADR-0006 (sizes, framework, precision, platform)
**Deferred:** (none — ablation is issue 09)

**Blocked by:** 05, 06

**Status:** in-progress (0.6B bf16 anchor + 1.7B done; 4B running 2026-09-05)

- [x] A config-diff check: only --model / --out differ across the three runs
- [x] 1.7B completes on 4090 within budget; loss curves logged (final 0.535)
- [ ] 4B completes on 4090 within budget; loss curves logged
- [ ] Three adapters + merged models mirrored locally (`outputs/adapters/`)
- [ ] Push to Hugging Face (per original ticket)
