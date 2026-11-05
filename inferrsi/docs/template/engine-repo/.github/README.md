# GitHub scaffolding

Copy these files only after checking the engine's existing contribution process.
The [scoped task](ISSUE_TEMPLATE/engine-task.md) records bounded implementation
scope; it is not a performance acceptance shortcut. The
[PR template](PULL_REQUEST_TEMPLATE.md) reports actual artifacts and checks.

The [CI draft example](workflows/ci-draft.yml.example) is deliberately disabled.
Replace its placeholder runner, toolchain setup, dependency installation and
checks with real engine commands, then rename it to `ci-draft.yml` only when
reviewed. Do not use no-op/echo checks or add automatic model downloads, hardware
benchmarks or campaign thresholds. Integrate with existing engine workflows
instead of creating duplicate lanes. This example does not change InferRSI's
draft/ready/merge/nightly workflows.
