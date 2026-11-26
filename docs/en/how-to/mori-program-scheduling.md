# MORI Program admission

Select MORI with `vllm serve MODEL --agentinfer --agentinfer-strategy mori`.
The scheduler requires a Program cache adapter that reports effective HBM and DRAM capacities
and applies ordered placement and release operations for a single EngineCore and DP rank.
`MoriProgramAdmissionController` runs the Program policy before native Chat dispatch.
The default Program policy is Progress-TTL.

MORI checks the server's bearer credentials for Chat and `POST /programs/release` before
sending an event to EngineCore. Both endpoints read `program_id` from the request body, then
from `extra_body`. Null and blank values fall through to the next source. Chat uses
`default` when neither source supplies an ID. Release requires an explicit ID, including
`{"program_id":"default"}` to release the default Program.

Overlapping requests keep the Program in REASONING until its last request finishes.
A release that the cache cannot acknowledge returns `released: false` and can be retried.
Successful release cancels pending admissions. A permit already granted can finish dispatch.
