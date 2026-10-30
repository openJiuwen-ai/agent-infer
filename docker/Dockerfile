FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.23.0

# Unit tests create throwaway repos with `git init` / `git commit`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Install deps before COPY so unchanged deps hit layer cache across PRs (vllm-omni deps-cache pattern).
RUN pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      'huggingface-hub>=1.5.0,<2.0' \
      datasets matplotlib zstandard \
      pytest pytest-timeout pytest-html pytest-xdist \
      'setuptools>=68,<85' wheel \
 && pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      'huggingface-hub>=1.5.0,<2.0' --force-reinstall

ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/home/models \
    HF_HUB_CACHE=/home/models/hub \
    HOME=/tmp/ci-home

# Copy the entrypoint before the repo so dependency layers stay cached.
# It drops root to whatever UID owns the checkout. See docker/ci-entrypoint.sh.
COPY docker/ci-entrypoint.sh /usr/local/bin/ci-entrypoint.sh
RUN chmod 755 /usr/local/bin/ci-entrypoint.sh \
 && mkdir -p /tmp/ci-home

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      -e . --no-deps

# CI image runs pytest. The entrypoint replaces `vllm serve` and drops privileges.
ENTRYPOINT ["/usr/local/bin/ci-entrypoint.sh"]
CMD ["/bin/bash"]
