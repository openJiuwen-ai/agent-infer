# Configurations

Keep versioned configuration examples for actual supported combinations. Explain
each file's purpose, model, device selection/count, precision, runtime versions,
defaults and overrides, plus the exact command that consumes it. Preserve the
engine's own format; no shared YAML runtime configuration is required.

```text
Configuration file: <actual-config-path>
Consumer / command: <actual-entrypoint-and-complete-command>
Recipe ID / deployment version: <recipe-id-and-ref>
Supported scope / limitations: <exact-combination-and-evidence>
```

Do not commit credentials or local absolute model paths as portable defaults.
Keep campaign-specific thresholds and baseline/candidate pins in acceptance
issues and evidence. Refer to [reproducibility](../docs/reproducibility.md).
