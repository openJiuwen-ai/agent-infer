# Pinned vLLM image for vllm-evolve real-bench.
# Pin chosen by the T0.4 API study (docs/reviews/t0_4_vllm_api_drift_2026-05-28.md):
# vLLM 0.21.0 on CUDA 12.9. The official image already bundles vLLM + the
# OpenAI-compatible server as its entrypoint, so request args appended to
# `docker run ... <image> --model ... --scheduler-cls ...` reach the server.
#
# The evolved scheduler plugin is NOT baked in: it is rendered per-evaluation
# and mounted read-only at /plugins (on PYTHONPATH) by
# vllm_evolve.bench.backend.build_docker_command. This Dockerfile exists to pin
# the exact base and provide a place to add bench-only deps if ever needed.
#
#   Build:  docker build -t vllm-evolve-bench:0.21.0 -f src/vllm_evolve/bench/Dockerfile .
#   Run:    orchestrated by bench.backend (do not run by hand)
FROM vllm/vllm-openai:v0.21.0-cu129-ubuntu2404

LABEL org.vllm-evolve.vllm-version="0.21.0"
LABEL org.vllm-evolve.cuda-version="12.9"
LABEL org.vllm-evolve.scheduler-cls="generated_scheduler:EvolvedScheduler"

# Plugins are mounted at run time; expose the convention for clarity.
ENV PYTHONPATH=/plugins
