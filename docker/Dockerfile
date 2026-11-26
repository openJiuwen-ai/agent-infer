# Build the exact ZCode CLI supported by the adapter, with its pinned toolchain.
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/node:24.14.0-bookworm AS zcode-build
ARG ZCODE_COMMIT=a9fe401eede6e6e5e01be9048679c206fe6f02d9
ARG ZCODE_REPOSITORY=https://github.com/zai-org/ZCode.git
RUN npm install -g pnpm@10.33.2
WORKDIR /opt/zcode
RUN git init . \
 && git remote add origin "${ZCODE_REPOSITORY}" \
 && for fetch_attempt in 1 2 3; do \
      echo "Fetching ZCode source (attempt ${fetch_attempt}/3)"; \
      if git -c http.version=HTTP/1.1 fetch --depth 1 origin "${ZCODE_COMMIT}"; then break; fi; \
      if [ "${fetch_attempt}" -eq 3 ]; then exit 1; fi; \
      sleep 5; \
    done \
 && git checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${ZCODE_COMMIT}" \
 && pnpm install --frozen-lockfile \
 && pnpm --filter @zcode/cli... build \
 && test "$(node apps/zcode-cli/packages/cli/dist/zcode.cjs --version)" = "0.17.1"

FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.23.0 AS ci-deps

# Unit tests create throwaway repos with `git init` / `git commit`.
# curl/xz unpack the Node tarball used by the DSH functional E2E.
# sudo is required when the DSH step stays root and wraps plan-subagent
# with `sudo -u test-user`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git sudo xz-utils tmux bubblewrap socat \
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

# Accept the CI agent's Python index without invalidating the Node layers.
ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple/

# Install deps before COPY so unchanged deps hit layer cache across PRs (vllm-omni deps-cache pattern).
RUN pip install --no-cache-dir \
      'huggingface-hub>=1.5.0,<2.0' \
      datasets matplotlib zstandard \
      pytest pytest-timeout pytest-html pytest-xdist \
      'setuptools>=68,<85' wheel \
 && pip install --no-cache-dir \
      'huggingface-hub>=1.5.0,<2.0' --force-reinstall

ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/home/models \
    HF_HUB_CACHE=/home/models/hub \
    HOME=/tmp/ci-home

# Jiuwen's sandbox mounts /usr but not /etc/alternatives. Keep the base image's
# Python interpreter reachable inside the sandbox before creating the venv.
RUN python3_target="$(readlink -f /usr/bin/python3)" \
 && test -x "$python3_target" \
 && if [ "$python3_target" != /usr/bin/python3 ]; then \
      ln -sfn "$python3_target" /usr/bin/python3; \
    fi

# Copy the entrypoint before the repo so dependency layers stay cached.
# It drops root to whatever UID owns the checkout. See docker/ci-entrypoint.sh.
COPY docker/ci-entrypoint.sh /usr/local/bin/ci-entrypoint.sh
RUN chmod 755 /usr/local/bin/ci-entrypoint.sh \
 && mkdir -p /tmp/ci-home

ENTRYPOINT ["/usr/local/bin/ci-entrypoint.sh"]
CMD ["/bin/bash"]

# The functional suite keeps all upstream agent/profile runtimes.
FROM ci-deps AS function-deps

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

# WorkSwarm provides the JiuwenSwarm commands in an isolated Python environment.
ARG JIUWENSWARM_VERSION=0.2.8
RUN python3 -m venv --system-site-packages /opt/jiuwenswarm \
 && /opt/jiuwenswarm/bin/pip install --no-cache-dir "workswarm==${JIUWENSWARM_VERSION}" \
 && for command in jiuwenswarm jiuwenswarm-init jiuwenswarm-start jiuwenbox-server; do \
      test -x "/opt/jiuwenswarm/bin/$command" \
      && ln -s "/opt/jiuwenswarm/bin/$command" "/usr/local/bin/$command" || exit 1; \
    done

# Shared scoring and Jiuwen dependencies for ready and nightly accuracy.
FROM ci-deps AS jiuwen-accuracy-deps

RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io \
 && rm -rf /var/lib/apt/lists/*

# Keep scoring and Jiuwen dependencies separate from vLLM.
RUN python3 -m venv /opt/agentbench/swebench-venv \
 && /opt/agentbench/swebench-venv/bin/python -m pip install \
      'swebench==4.1.0' 'numpy>=2' \
 && /opt/agentbench/swebench-venv/bin/python -m pip check \
 && printf '%s\n' \
      'harness_python: /opt/agentbench/swebench-venv/bin/python' \
      >/opt/agentbench/evaluation.yaml

# Read dependency metadata separately so Jiuwen also caches AgentInfer dependencies.
COPY pyproject.toml /opt/agentbench/agentinfer-pyproject.toml
ARG JIUWENSWARM_VERSION=0.2.8
RUN python3 -m venv /opt/agentbench/jiuwenswarm-venv \
 && python3 -c 'import tomllib, sys; p = tomllib.load(sys.stdin.buffer); print(*p["build-system"]["requires"], *p["project"]["dependencies"], sep="\n")' \
      </opt/agentbench/agentinfer-pyproject.toml >/opt/agentbench/agentinfer-requirements.txt \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip install \
      -r /opt/agentbench/agentinfer-requirements.txt \
      "workswarm==${JIUWENSWARM_VERSION}" \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip check \
 && for name in jiuwenswarm jiuwenswarm-init jiuwenswarm-start jiuwenbox-server; do \
      ln -s /opt/agentbench/jiuwenswarm-venv/bin/$name /usr/local/bin/$name; \
    done

# Ready accuracy installs Jiuwen and the scorer, without Claude/ZCode CLIs.
FROM jiuwen-accuracy-deps AS accuracy-jiuwen

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip install \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip check

# Nightly keeps the accuracy-tested Claude version and reuses the pinned ZCode build.
FROM jiuwen-accuracy-deps AS accuracy

RUN npm install --global --prefix /usr/local \
      @anthropic-ai/claude-code@2.1.327

# Keep the workspace and adjacent resources: the CLI bundles external runtime assets.
COPY --from=zcode-build /opt/zcode /opt/zcode
COPY --from=zcode-build /usr/local/bin/node /opt/zcode-node/bin/node
RUN sed -i '1{/^#!/d;}' /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs \
 && sed -i '1i#!/opt/zcode-node/bin/node' /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs \
 && chmod 755 /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs \
 && ln -s /opt/zcode/apps/zcode-cli/packages/cli/dist/zcode.cjs /usr/local/bin/zcode \
 && test "$(zcode --version)" = "0.17.1"

RUN ln -s /usr/local/bin/zcode /usr/local/bin/zcode-launcher

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip install \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip check

# Ready functional and CPU jobs use the upstream functional runtime.
FROM function-deps AS ci

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -e . --no-deps
