# Build the exact ZCode CLI supported by the adapter, with its pinned toolchain.
FROM node:24.14.0-bookworm AS zcode-build
ARG ZCODE_COMMIT=a9fe401eede6e6e5e01be9048679c206fe6f02d9
RUN npm install -g pnpm@10.33.2
WORKDIR /opt/zcode
RUN git init . \
 && git remote add origin https://github.com/zai-org/ZCode.git \
 && git fetch --depth 1 origin "${ZCODE_COMMIT}" \
 && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${ZCODE_COMMIT}" \
 && pnpm install --frozen-lockfile \
 && pnpm --filter @zcode/cli... build \
 && test "$(node apps/zcode-cli/packages/cli/dist/zcode.cjs --version)" = "0.17.1"

FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.23.0

# Unit tests create throwaway repos with `git init` / `git commit`.
# curl/xz unpack the Node tarball used by the DSH functional E2E.
# sudo is required when the DSH step stays root and wraps plan-subagent
# with `sudo -u test-user`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git sudo xz-utils tmux bubblewrap \
 && rm -rf /var/lib/apt/lists/*

# DSH 0.2.1-rc.2 for the unified AgentBench functional suite. Install under /usr/local
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

ARG CLAUDE_CODE_VERSION=2.1.323
RUN npm install --global --prefix /usr/local \
      --registry https://registry.npmjs.org \
      --allow-scripts=@anthropic-ai/claude-code \
      "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
 && claude --version

# Keep the workspace and adjacent resources: the CLI bundles external runtime assets.
COPY --from=zcode-build /opt/zcode /opt/zcode
COPY --from=zcode-build /usr/local/bin/node /opt/zcode-node/bin/node
RUN sed -i '1{/^#!/d;}' /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs \
 && sed -i '1i#!/opt/zcode-node/bin/node' /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs \
 && chmod 755 /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs \
 && ln -s /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs /usr/local/bin/zcode \
 && test "$(zcode --version)" = "0.17.1"

# JiuwenSwarm uses its own Python environment so its dependencies cannot replace vLLM's.
ARG JIUWENSWARM_VERSION=0.2.3
RUN python3 -m venv --system-site-packages /opt/jiuwenswarm \
 && /opt/jiuwenswarm/bin/pip install --no-cache-dir "jiuwenswarm==${JIUWENSWARM_VERSION}" \
 && for command in jiuwenswarm jiuwenswarm-init jiuwenswarm-start jiuwenbox-server; do \
      test -x "/opt/jiuwenswarm/bin/$command" \
      && ln -s "/opt/jiuwenswarm/bin/$command" "/usr/local/bin/$command" || exit 1; \
    done

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
