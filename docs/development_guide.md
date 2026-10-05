# Team Development Guide

## Workflow for Team Members
1. Clone the repository and checkout your designated branch:
   - Tharun: `git checkout p1`
   - Atul: `git checkout p2`
   - Ritika: `git checkout p3`
   - Kunal: `git checkout p4`

2. Install your role's specific dependencies:
   `make install-p1` (or `p2`, `p3`, `p4`)

3. Implement features strictly within your assigned directories.
4. Run tests before pushing:
   `make test`
   `make verify-ownership`

5. Create Pull Requests targeting `main`.
