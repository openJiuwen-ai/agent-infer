FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.23.0

# Install deps before COPY so unchanged deps hit layer cache across PRs (vllm-omni deps-cache pattern).
RUN pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      'huggingface-hub>=1.5.0,<2.0' \
      datasets matplotlib zstandard pytest pytest-timeout \
 && pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      'huggingface-hub>=1.5.0,<2.0' --force-reinstall

ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/home/models \
    HF_HUB_CACHE=/home/models/hub

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      -e . --no-deps

# CI image runs pytest; clear the base image `vllm serve` ENTRYPOINT.
ENTRYPOINT []
CMD ["/bin/bash"]
