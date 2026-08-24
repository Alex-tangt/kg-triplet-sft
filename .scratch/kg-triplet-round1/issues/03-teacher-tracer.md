# 03 — Teacher labeling tracer

**What to build:** The full teacher-labeling machinery — teacher prompt (relation dictionary,
entity-naming rules, exhaustiveness demand, fixed weight anchors 0.2/0.5/0.8, empty-array rule),
DashScope client with retry/backoff and a budget cap — proven on 100 pilot passages before the
full run spends real money.

**Blocked by:** 01, 02

**Status:** ready-for-agent

- [ ] Teacher prompt contains all five elements from the grilling decisions
- [ ] 100 pilot passages labeled within budget; injected-failure tests prove retry works
- [ ] Raw labels parse into triplets that pass the contract's schema validation
- [ ] Hard-negative judgment works: passages with no valid triplets get `[]`
- [ ] Weight anchors observed in pilot labels; outliers flagged
