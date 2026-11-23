FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.23.0 AS ci-deps

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

ENTRYPOINT ["/usr/local/bin/ci-entrypoint.sh"]
CMD ["/bin/bash"]

# Shared scoring and Jiuwen dependencies for ready and nightly accuracy.
FROM ci-deps AS jiuwen-accuracy-deps

RUN apt-get update \
 && apt-get install -y --no-install-recommends bubblewrap docker.io \
 && rm -rf /var/lib/apt/lists/*

# Keep scoring and Jiuwen dependencies separate from vLLM.
RUN python3 -m venv /opt/agentbench/swebench-venv \
 && /opt/agentbench/swebench-venv/bin/python -m pip install \
      -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
      'swebench==4.1.0' 'numpy>=2' \
 && /opt/agentbench/swebench-venv/bin/python -m pip check \
 && printf '%s\n' \
      'harness_python: /opt/agentbench/swebench-venv/bin/python' \
      >/opt/agentbench/evaluation.yaml

# Read dependency metadata separately so Jiuwen also caches AgentInfer dependencies.
COPY pyproject.toml /opt/agentbench/agentinfer-pyproject.toml
RUN python3 -m venv /opt/agentbench/jiuwenswarm-venv \
 && python3 -c 'import tomllib, sys; p = tomllib.load(sys.stdin.buffer); print(*p["build-system"]["requires"], *p["project"]["dependencies"], sep="\n")' \
      </opt/agentbench/agentinfer-pyproject.toml >/opt/agentbench/agentinfer-requirements.txt \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip install \
      -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
      -r /opt/agentbench/agentinfer-requirements.txt \
      'jiuwenswarm==0.2.4b3' 'openjiuwen[claude,codex]==0.1.16.post2' 'openai-codex==0.144.4' \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip check \
 && for name in jiuwenswarm jiuwenswarm-init jiuwenswarm-start jiuwenbox-server; do \
      ln -s /opt/agentbench/jiuwenswarm-venv/bin/$name /usr/local/bin/$name; \
    done

# Ready accuracy installs Jiuwen and the scorer, without Claude/ZCode CLIs.
FROM jiuwen-accuracy-deps AS accuracy-jiuwen

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip install \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip check

# Nightly extends the same cached dependencies with all remaining agents.
FROM jiuwen-accuracy-deps AS accuracy

RUN apt-get update \
 && apt-get install -y --no-install-recommends tmux socat \
 && rm -rf /var/lib/apt/lists/*

RUN npm install --global --prefix /usr/local \
      @anthropic-ai/claude-code@2.1.327

# Build the Node bundle required by the ZCode adapter, preserving NODE_OPTIONS.
RUN curl -fsSL --retry 3 \
      https://nodejs.org/dist/v24.14.0/node-v24.14.0-linux-x64.tar.xz \
      -o /tmp/zcode-node.tar.xz \
 && tar -xJf /tmp/zcode-node.tar.xz -C /opt/agentbench \
 && rm /tmp/zcode-node.tar.xz \
 && /opt/agentbench/node-v24.14.0-linux-x64/bin/node \
      /usr/local/lib/node_modules/npm/bin/npm-cli.js install --global \
      --prefix /opt/agentbench/node-v24.14.0-linux-x64 pnpm@10.33.2 \
 && curl -fsSL --retry 3 \
      https://codeload.github.com/zai-org/ZCode/tar.gz/a9fe401eede6e6e5e01be9048679c206fe6f02d9 \
      -o /tmp/zcode-source.tar.gz \
 && tar -xzf /tmp/zcode-source.tar.gz -C /opt/agentbench \
 && rm /tmp/zcode-source.tar.gz

RUN export PATH="/opt/agentbench/node-v24.14.0-linux-x64/bin:$PATH" \
 && cd /opt/agentbench/ZCode-a9fe401eede6e6e5e01be9048679c206fe6f02d9 \
 && pnpm install --frozen-lockfile \
 && pnpm --filter @zcode/cli... build \
 && printf '%s\n' '#!/bin/sh' \
      'exec /opt/agentbench/node-v24.14.0-linux-x64/bin/node /opt/agentbench/ZCode-a9fe401eede6e6e5e01be9048679c206fe6f02d9/apps/zcode-cli/packages/cli/dist/zcode.cjs "$@"' \
      >/usr/local/bin/zcode-launcher \
 && chmod 755 /usr/local/bin/zcode-launcher \
 && zcode-launcher --version

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -i https://mirrors.huaweicloud.com/repository/pypi/simple/ \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip install \
      -e . --no-deps --no-build-isolation \
 && /opt/agentbench/jiuwenswarm-venv/bin/python -m pip check

# Keep the default build lightweight; ready and CPU use this target explicitly.
FROM ci-deps AS ci

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --trusted-host mirrors.aliyun.com \
      -e . --no-deps
