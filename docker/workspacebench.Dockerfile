# Build the pinned upstream office image as workspace-bench:local first.
FROM workspace-bench:local AS official
RUN git init /src && git -C /src fetch --depth 1 https://github.com/OpenDataBox/Workspace-Bench.git 3fbd0f1a136720fece86786545983e26642c3db2 && git -C /src checkout --detach FETCH_HEAD

FROM workspace-bench:local
WORKDIR /opt/workspacebench/evaluation
COPY --from=official /src/evaluation/package*.json ./
RUN npm ci --omit=dev
COPY --from=official /src/evaluation/baselines ./baselines
COPY --from=official /src/evaluation/vendor ./vendor
COPY --from=official /src/evaluation/src ./src
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap && rm -rf /var/lib/apt/lists/*
# jiuwenbox obtains the home from passwd for known UIDs (including nobody).
# The task's disposable /tmp is writable for both mapped and unmapped users.
RUN awk -F: 'BEGIN {OFS=":"} $3 != 0 {$6="/tmp"} {print}' /etc/passwd > /tmp/passwd && cat /tmp/passwd > /etc/passwd
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv venv --python 3.12 /opt/wsb-venv
ENV PATH="/opt/wsb-venv/bin:${PATH}"
RUN uv pip install 'workswarm==0.2.8' 'openjiuwen==0.1.20' 'deepseek-harness-sdk==0.1.0rc7'
WORKDIR /opt/agentinfer
COPY pyproject.toml README.md setup.py ./
COPY agentinfer ./agentinfer
RUN uv pip install . python-docx openpyxl python-pptx pypdf pillow && python -c 'import jiuwenbox, agentinfer'
WORKDIR /tmp
