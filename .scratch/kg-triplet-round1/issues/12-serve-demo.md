# 12 — Serve & demo

**What to build:** Deployment on this CPU-only laptop: GGUF Q4 exports for the 0.6B and 3B
models, Ollama Modelfiles for both, and a Gradio app — paste text, get the triplet list plus a
pyvis node graph; 0.6B default, 3B selectable; fully offline.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Both models served by Ollama locally and respond to a triplet-extraction request
- [ ] Gradio app renders triplet list and pyvis graph from a pasted passage
- [ ] The prompt the app sends is rendered from the single template (ADR-0001)
- [ ] Demo verified end-to-end with network disabled
