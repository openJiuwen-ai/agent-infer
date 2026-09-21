# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""No-op Cordis HMR service for AgentBench's headless DSH boot.

The headless bundle already marks ``id: hmr`` as ``disabled: true``. After boot
``ctx.get('hmr')`` is therefore undefined, and ``runProfile`` does
``loader.create({ name: '@deepseek-ai/cordis-plugin-hmr' })``. That constructor
requires Node to be started with ``--expose-internals``. AgentBench does not
pass that flag when launching DSH, so the plugin is constructed without
internals and fails with ``--expose-internals is required for HMR service`` —
the task dies in headless startup before any ``/v1/chat/completions`` request.

Disabling HMR again in the profile policy makes the service more certainly
missing. ``--patch`` on the original ``id: hmr`` only rewrites the already
disabled row, so a stub's ``apply()`` never runs and the real plugin is still
created. Passing ``node --expose-internals`` from ``runner.py`` is not
acceptable for AgentBench.

The working fix is a task-local no-op plugin that ``provide('hmr')`` during
boot and implements ``registerConfig()`` as a disposer. It does not watch files
and does not touch ``loader.internal``. Insert it as a new plugin id so boot
occupies the HMR service and ``runProfile`` skips the real module.
"""

AGENTBENCH_HMR_STUB = """\
export function apply(ctx) {
  ctx.provide('hmr', {
    registerConfig() {
      return () => undefined
    },
  })
}
"""
