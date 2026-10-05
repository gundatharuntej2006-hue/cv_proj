# ADR 0001: Modular Micro-Package Architecture

## Status
Accepted

## Context
The project involves 4 team members (Tharun, Atul, Ritika, Kunal) working across distinct stages of a clinical ML pipeline. Without strict module boundaries and package separation, merge conflicts, hidden circular imports, and uncoordinated dependency breaks would occur.

## Decision
We organize the codebase into person-owned functional subpackages within `src/`:
- `tbcore`: Shared foundational models, schemas, enums, IO utilities
- `p1_data`, `p1_app`, `p1_viewhead`: Owned by Tharun (P1)
- `p2_cls`, `p2_gate`: Owned by Atul (P2)
- `p3_det`, `p3_xai`, `p3_ops`: Owned by Ritika (P3)
- `p4_stats`, `p4_pipeline`, `p4_eval`: Owned by Kunal (P4)

## Consequences
- Every module has a designated owner responsible for code quality and test coverage.
- Inter-module communication is mediated exclusively via `contracts/` and `tbcore.schemas`.
