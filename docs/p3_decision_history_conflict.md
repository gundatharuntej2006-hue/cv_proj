# P3 decision history conflict — research review required

Detected 2026-11-03 while checking the actual rebased team history.

The current D16–D19 files are unlocked rule-only records. However, the same paths already held different
criteria and locked values in team commit `675e0529a8`. That commit precedes P3's rule-only commits:

| Decision | Current rule-only commit | Earlier conflicting contents |
|---|---|---|
| D16 | `dbccc8d460` | Input size 512, development AP50 0.421, locked; different criterion |
| D17 | `e7ba134e13` | 36 epochs, AP50 0.421, shortened checkpoint hash; different criterion |
| D18 | `b9585216a9` | Earlier locked result under a different criterion |
| D19 | `01ddf6878c` | Earlier locked result under a different criterion |

The earlier records use placeholder-looking provenance (`git_sha: c0ffee1`); some recorded lock dates are
later than their containing commit. This suggests scaffold records, but is not sufficient evidence to
silently discard them or claim that the current criteria were registered before every result.

## Implemented handling

- The history verifier continues to report FAIL. It does not delete, hide, or reinterpret old commits.
- Result-recording tools now refuse a conflicting history before writing new values.
- Current decisions and guide acknowledgements remain unchanged and unlocked.
- The audit's duplicate B4 row was fixed; a failing history is recorded once.
- The ordinary gate unit tests use temporary synthetic decisions, with an additional test proving that an
  unlocked D16 is refused. They do not assert that this research checkout is ready to evaluate.

## Required disposition from P4/team

Inspect the original evidence behind `675e0529a8` and document whether those values were scaffold examples
or actual measurements. Record the disposition in the project research record. If criteria changed after
real results, report that fact and establish the new protocol and eligible data before proceeding. If the
records were scaffold-only, document their withdrawal and a reviewed version/boundary for the real protocol.
Only then implement the agreed versioned decision-history policy and run its tests. Do not rewrite history,
backdate a registration, set fake locks, or drop old result commits merely to make the audit green.

The original Grad-CAM D19 also predates the regional P2 model. Its replacement needs a stated target score,
preprocessing, layer, sanity rule and measured VRAM; do not treat the adapter code as a locked protocol.
