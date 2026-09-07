# CI / UT 修复记录：让 `ut.sh` 在 PR 门禁上真正跑通

> 本文记录为了修复 PR 的 UT（单元测试）门禁而对 `ut.sh`、测试文件布局等所做的变更，
> 以及每一处变更背后的原因。方便后续同学理解“为什么这里要这么写”，而不是当成黑盒。

## 背景

每次提交 PR，CI 都会运行仓库根目录的 `ut.sh`。它做的事情大致分四步：

1. 用 `uv` 装 Python 3.11 并建虚拟环境 `.venv-ci`；
2. **把本项目（`agentinfer`）打成 wheel 并安装**，同时带上运行时依赖；
3. 装 pytest 系列测试框架；
4. 跑 `pytest tests ...` 并上传报告。

在本次修复前，CI 依次在“安装依赖”和“pytest 收集”两个阶段报错，导致门禁 FAILED。
下面按“遇到的顺序 + 如何修”逐条记录。

***

## 变更一：安装本项目改用 venv 自带的 pip，不再用 uv

### 现象

`uv pip install` 报错：

```
Failed to parse metadata from built wheel
Metadata field Name not found
```

### 排查结论

- 本地用 `pip`（Windows / Linux 都验过）构建本项目，wheel 的 METADATA 里 **`Name`** **字段是完好的**；

- 出错的不是 `pyproject.toml` 也不是 `setup.py`，而是 **CI 上的** **`uv`** **在解析“本地构建出的 wheel”元数据时的缺陷**；

- `uv` 创建的虚拟环境**默认不带 pip**，所以第一步需要先用 `uv` 把 pip 注入进去。

### 变更

把“装本项目”这一句从：

```bash
uv pip install --python "$PYBIN" "$BASE_DIR"
```

改成两步：

```bash
# 先让 uv 把 pip 注入虚拟环境（uv 建的 venv 默认没 pip）
uv pip install --python "$PYBIN" pip
# 再用 pip 安装本项目，绕开 uv 解析本地 wheel 的缺陷
"$PYBIN" -m pip install -i "$PY" --disable-pip-version-check "$BASE_DIR"
```

***

## 变更二：pip 安装时显式带上华为云源 `-i "$PY"`

### 现象

上一步改用 pip 后，出现新报错：

```
ERROR: No matching distribution found for uvicorn
```

### 排查结论

- 脚本给 **uv** 设了华为云源（`export UV_DEFAULT_INDEX=$PY`）；

- 但 **`pip`** **不读** **`UV_DEFAULT_INDEX`** 这个变量，于是 pip 走了容器默认的镜像，而那个默认镜像里缺 `uvicorn`（其余依赖都能拉到，就它没有）；

- 之前 uv 能装上 `uvicorn==0.52.4`，正是因为用的是华为云源。

### 变更

给 pip 显式加 `-i "$PY"`，让 pip 和 uv 用同一个华为云源：

```bash
"$PYBIN" -m pip install -i "$PY" --disable-pip-version-check "$BASE_DIR"
```

***

## 变更三：pytest 排除 vLLM 集成测试目录 `tests/agentcache/core`

### 现象

安装打通后，pytest 在“收集（collection）”阶段报 8 个错误，例如：

```
ModuleNotFoundError: No module named 'vllm'
ModuleNotFoundError: No module named 'openai'
ImportError: cannot import name 'LLM' from 'agentinfer'
```

### 排查结论

- 依赖列表（`pyproject.toml` 的 `[project].dependencies`）里**故意没有放** **`vllm`、`openai`**：
  `vllm` 是重度依赖（需要 GPU/CUDA），`agentinfer/__init__.py` 对 vllm 做了“可选的、try/except 优雅降级”处理；

- 但 `tests/agentcache/core/` 整目录的测试（`test_agent_scheduler*.py`、`test_request_queue.py`、
  `test_vllm_*.py` …）以及核心模块 `scheduler.py`，都是**直接在模块顶层硬 import vllm**；

- 这些测试本质是 **vLLM 集成测试**，而当前 CI 节点是 **CPU-only（无 GPU）**，既装不上也没法跑 vllm，
  所以它们在 CI 上**本就不该被收集**。这正好和脚本里已有的 `--ignore=tests/e2e`、`-m "not gpu_test ..."` 是同一个设计意图。

### 变更

在 pytest 命令上对 `tests/agentcache/core` 加一行 `--ignore`：

```bash
"$PYBIN" -m pytest tests -q \
    --ignore=tests/e2e \
    --ignore=tests/agentcache/core \
    -m "not gpu_test and not e2e_perf" \
    --html="$HTML" --self-contained-html --junitxml="$JUNIT"
```

> 说明：这些 vLLM 集成测试仍是有效的，只是**不适合在 CPU-only 的 CI 上跑**。需要跑的时候，
> 在装有 vllm + GPU 的开发/测试机上单独执行 `pytest tests/agentcache/core` 即可。

***

## 变更四：重命名重复的 `tests/scheduling/test_contracts.py`

### 现象

收集时报错：

```
import file mismatch: imported module 'test_contracts' has this __file__ attribute:
  .../tests/agentbench/agents/test_contracts.py
which is not the same as the test file we want to collect:
  .../tests/scheduling/test_contracts.py
```

### 排查结论

仓库里有两个同名测试文件 `test_contracts.py`，pytest 默认按“不带路径的模块名”去重，导致模块名冲突、收集报错。

### 变更

用 `git mv` 把 `tests/scheduling/test_contracts.py` 改名为 `test_scheduling_contracts.py`（保留历史），消除同名冲突。

***

## 结果验证

在本地（与 CI 相同的“只装运行时依赖、不装 vllm/openai”环境）验证：

- `pytest tests --ignore=tests/e2e --ignore=tests/agentcache/core -m "not gpu_test and not e2e_perf"`，
  结果 `417 passed`，收集无错误，退出码 0。

***

## 未动的部分（避免误改）

- **`pyproject.toml`** **的** **`setuptools>=68,<85`** **上限**：有一轮曾怀疑 setuptools 版本是元凶，后经排查确认与
  “Name not found”无关，该上限保留（CI 实际装到 `setuptools==84.0.0`，构建正常）。若后续想精简 PR，
  可考虑去掉 `<85` 封顶，但**不是必须**，与本问题无关。

- **`scheduler.py`** **内部的 vllm 硬 import**：没有改。因为它深度继承 vllm 的类，改成惰性导入风险高，
  且并不能让那批 vLLM 集成测试带动；所以本次只做“在 CI 上不收集它”，不动核心逻辑。

- **`setup.py`**：保留未动（同样与本次问题无关）。

