# ADR 0002: Formal Data Contracts & JSON Schema Interfaces

## Status
Accepted

## Context
When disparate components (e.g. P2 classification, P3 detection, P4 triage) are developed concurrently on separate git branches, mismatches in dictionary keys or tensor shapes cause silent integration failures.

## Decision
All inter-component boundaries are formalized using JSON Schemas under `contracts/` and mirrored by type-checked Pydantic v2 classes in `src/tbcore/schemas.py`. Any schema change is treated as an API-breaking change requiring approval from all team members.

## Consequences
- Mock data packets (`mock/sample_pipeline_packet.json`) can be checked in CI against schemas.
- Downstream developers (e.g. P4 triage or P1 dashboard) can build against mock schemas before upstream models finish training.
