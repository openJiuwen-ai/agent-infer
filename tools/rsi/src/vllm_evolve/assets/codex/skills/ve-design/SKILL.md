---
name: ve-design
description: Record one falsifiable scheduling mechanism before Codex returns candidate source.
---

# ve-design — state the mechanism

For a manual round, run `ve phase set DESIGN`, then
`ve design --note "<mechanism, workload regime, metric, expected effect>"`.

Prefer a discrete algorithmic mechanism—queue-state switching, lifecycle ordering, bounded aging,
or hierarchical admission—over weighted-score soup. State what an ablation removes and which
failure would falsify the idea. Cite the selected ExpertBrief mechanism IDs and map the design to
specific repository symbols. Do not assume a paper's reported gain transfers to this environment.
Choose different mechanism families for generation 0; later generations may refine, combine,
reject, switch, or repair using measured feedback. Do not edit the policy in this phase. Continue
with `ve-generate`.

When `operator.kind=crossover`, name one semantic contribution from each selected parent, explain
compatibility/conflict resolution, describe the new combined control flow, and define one ablation
per parent contribution. If two distinct verified/scored parents are unavailable, record mutation;
never call self-crossover or source-text splicing crossover.
