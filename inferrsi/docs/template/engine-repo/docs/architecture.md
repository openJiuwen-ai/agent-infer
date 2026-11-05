# Engine architecture

Document the **implemented** model-loading, preprocessing, execution, decoding
and serving path. Explain the actual language/build layout, device ownership,
memory/parallelism strategy, configuration boundaries and extension contracts.
Separate implemented features from proposals and unsupported combinations.

```text
Entry / request contract: <actual-entrypoint-and-request-schema>
Model / weights: <actual-loading-and-model-boundary>
Backend / hardware: <actual-execution-interface-and-supported-scope>
Outputs / streaming: <actual-decoding-and-response-contract>
Build / runtime dependencies: <actual-toolchain-and-pinned-requirements>
Extension and affected consumers: <actual-interface-and-consumers>
```

Use diagrams only when they explain the engine's own behavior. Preserve upstream
sources and licenses, and record local image ownership in
[assets](assets/README.md). There is no requirement to copy InferRSI's module
layout or add its runtime. Link real code after paths exist; leave substitution
examples in code fences until then.

Use [reproducibility](reproducibility.md) for environment and validation identity;
keep experiment-specific pins and numeric gates in the acceptance issue and
campaign evidence.
