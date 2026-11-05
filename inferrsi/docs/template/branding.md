# InferMatrixLab engine branding

Use **InferMatrixLab** as the organization name and the engine's own name as the
product title. Add a visible “An InferMatrixLab project” attribution linking to
[InferMatrixLab](https://github.com/InferMatrixLab), plus the actual implementation
source and license. Attribution does not imply that an upstream dependency or
external project belongs to the organization.

## Confirmed assets and ownership

The current InferRSI [banner](../assets/inferrsi-banner.svg) identifies **InferRSI**.
It may illustrate InferRSI in an explanation of the workflow; do not relabel or
copy it as another engine's product banner. The architecture diagrams in that
directory are documentation illustrations, not organization logos.

The existing organization-maintained **InferMatrix** logo is copied locally in
[the scaffold's asset directory](engine-repo/docs/assets/README.md), with source
repository/path/revision recorded there. Its upstream spelling remains
InferMatrix; text attribution uses the GitHub organization name InferMatrixLab.
The source is the
[organization website at a fixed revision](https://github.com/InferMatrixLab/infermatrix.github.io/blob/b3903bdaffd4e9550263bf42791995b83f514480/assets/infermatrix-logo.png).
Do not invent or relabel a different official logo. The recipe website uses the
same source asset.

Keep the local copy and provenance together. Future updates come deliberately
from that canonical source and are reviewed by the project maintainer. The source
does not supply a separately confirmed asset-license grant; this copy retains
organization-branding provenance and does not assert general redistribution
rights. Preserve upstream notices for reused code, diagrams and fonts. Avoid
branch-relative raw remote image URLs that can change silently.

## Placement and accessibility

| Surface | Presentation |
| --- | --- |
| README header | Engine title, local organization mark, engine banner if approved, organization attribution and short purpose |
| Documentation | Engine name, organization attribution, navigation to installation, reproduction, evidence and recipes |
| Recipe website | InferMatrixLab wordmark, recipe identity, actual engine/upstream owner, source and evidence |
| Light/dark backgrounds | Existing white-background logo on its intact light surface; text attribution remains readable on either surface |

The current PNG has its own white background; keep it intact on dark pages.
No dark/transparent variant is supplied. Use descriptive alt text for meaningful
images. Keep names and essential
navigation readable as text. Check desktop/mobile widths, transparent assets on
light and dark surfaces, contrast, aspect ratio and long engine names. Do not
stretch a logo or embed unsupported performance/quality claims in a banner.

Add a local image only after the file exists and permission is recorded. The
following is substitution guidance in a code fence, not a live template link:

```text
Image alt text: <engine-name> — <short-purpose>
Local image path: docs/assets/<approved-banner-file>
Attribution: An InferMatrixLab project (https://github.com/InferMatrixLab).
Source: <actual-engine-or-upstream-repository-url>
License: <actual-license-and-upstream-notices>
```

## Current links and historical evidence

Use current organization repository URLs in current clone/install commands,
navigation and recipe source declarations. Keep exact upstream authorship,
licenses and historical source/commit links where they identify an experiment.
Do not rewrite an old evidence bundle to appear newly run or organization-owned.
Engine publication and catalog registration do not change an acceptance verdict.
