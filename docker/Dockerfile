FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.23.0

# Unit tests create throwaway repos with `git init` / `git commit`.
# curl/xz unpack the Node tarball used by the DSH functional E2E.
# sudo is required when the DSH step stays root and wraps plan-subagent
# with `sudo -u test-user`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git sudo xz-utils \
 && rm -rf /var/lib/apt/lists/*

# DSH 0.2.1-rc.2 for tests/e2e/function/agentbench/dsh. Install under /usr/local
# so the non-root user from ci-entrypoint.sh can execute node and dsh.
# Node 24 is inside the supported range (^22.19.0 || >=24).
ARG NODE_VERSION=24.22.1
# Prefer upstream nodejs.org / registry.npmjs.org over npmmirror so freshly
# published transitive deps (e.g. @aws-sdk/*) are visible during CI image build.
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
      -o /tmp/node.tar.xz \
 && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
 && rm /tmp/node.tar.xz \
 && npm install --global --prefix /usr/local \
      --registry https://registry.npmjs.org \
      --allow-scripts=@deepseek-ai/dsh-subprocess-local,koffi,node-pty,@google/genai,protobufjs \
      @deepseek-ai/dsh@0.2.1-rc.2 \
 && node --version \
 && dsh --version

# Install deps before COPY so unchanged deps hit layer cache across PRs (vllm-omni deps-cache pattern).
RUN pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      'huggingface-hub>=1.5.0,<2.0' \
      datasets matplotlib zstandard \
      pytest pytest-html pytest-xdist \
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
