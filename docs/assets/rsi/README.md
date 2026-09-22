# RSI architecture assets

- [Chinese presentation](agentinfer-rsi-guide-zh.pptx): 16 slides with Chinese speaker notes.
- [Layered architecture](rsi-layered-architecture.png): [SVG](rsi-layered-architecture.svg) / [Mermaid](rsi-layered-architecture.mmd).
- [Backend responsibility layers](rsi-backend-layers.png).
- [Numbered evolution loop](rsi-numbered-loop.png).
- [Dense feedback](rsi-dense-feedback.png): [SVG](rsi-dense-feedback.svg) / [Mermaid](rsi-dense-feedback.mmd).

The deck uses editable native diagram shapes. UI screenshots and values are synthetic demonstrations, not measured improvements.
Target architecture and implemented bootstrap boundaries are explicitly distinguished. GPU/NPU/model compatibility is unverified.

Build source: [tools/rsi/build_presentation.mjs](../../../tools/rsi/build_presentation.mjs).
The script requires Node.js, `@oai/artifact-tool`, and a presentations skill installation containing the referenced finalizer.
Rendering/screenshot prerequisites are documented in the script. This is an optional artifact build tool,
not an RSI runtime dependency.

```bash
node tools/rsi/build_presentation.mjs --build-dir /outside/repo/ppt-build \
  --skill-dir /path/to/presentations/skill --python /path/to/python
```

Use `ARTIFACT_TOOL_MODULES` when Node dependencies are supplied outside normal module resolution,
and `--font` to choose an installed CJK font.
Temporary renders and validation reports remain outside the repository. The checked-in presentation is usable without rebuilding.
