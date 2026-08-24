# 01 — Contract module + vendored harness

**What to build:** The cross-phase contract — the 20 relation constants, the triplet schema
validator, and the prompt template (target prompt reproduced verbatim from the original, with
the ontology implicit) — as one module consumed by dataset, finetune, eval, and serve. The HGR
evaluation harness is vendored into the repo unchanged and verified to run offline. A drift test
asserts the contract's relation set stays byte-identical to the harness's frozen set.

**Blocked by:** None — can start immediately

**Status:** done

- [x] Contract module exports the 20 relations, the triplet validator, and the prompt template;
      unit tests pass on fixtures (valid/invalid triplets, empty array, hard negatives)
- [x] Target prompt string asserted character-for-character against the original prompt
- [x] Harness vendored unchanged; the offline run (`--no-embeddings --limit 50`) prints a
      composite in [0, 1] and exits 0
- [x] Drift test fails if the relation set diverges from the harness constants
- [x] Teacher prompt module skeleton defined separately from the target prompt (five elements,
      fixed weight anchors, in-chunk grounding rule) per ADR-0001
