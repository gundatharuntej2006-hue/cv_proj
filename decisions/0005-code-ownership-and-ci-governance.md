# ADR 0005: Branch Ownership & CI Governance

## Status
Accepted

## Context
Four students must work concurrently on GitHub without blocking each other or accidentally rewriting another team member's code.

## Decision
1. `main` branch serves as the clean integration trunk and always contains passing tests.
2. Feature development proceeds on 4 persistent branches: `p1`, `p2`, `p3`, `p4`.
3. GitHub `CODEOWNERS` enforces that reviews are required for changes to specific folders.
4. CI checks verify contract conformance, unit tests, and branch boundary isolation.

## Consequences
- Zero branch conflicts during parallel sprints.
- Clear traceability of work for evaluation and grading.
