# =============================================================================
# CLEAR concept audit - Vietnam TUNE only.  Paste this whole thing into ONE cell.
#
# Diagnostic. It cannot improve, and must not be reported as improving, any screening
# metric. The reported Vietnam screening AUROC is 0.7242 and this run does not touch it.
#
#   data    : Vietnam TUNE only, 1,352 cases, by allow-list. CC0-1.0 (Dryad API).
#             CC0 waives copyright, NOT ethics, privacy or platform obligations.
#   locked  : Vietnam REPORT and TBX11K internal test - never attached, never read.
#   model   : CLEAR, Apache-2.0, pinned revision 142f169c8b00...
#   concepts: 9 exact vocabulary entries. cavity, nodule and reticular opacity are
#             EXCLUDED for lacking one - not approximated.
#
# Settings required: Accelerator = GPU, Internet = ON, Persistence = Variables and Files.
# Every stage raises on failure, which aborts Run All. A raised guard is the package
# working, not something to route around.
# =============================================================================
import json, sys, time, csv, io
from pathlib import Path

INP, WORK = Path('/kaggle/input'), Path('/kaggle/working')
t_start = time.perf_counter()

# Set this when more than one copy of the code is attached - for instance when an older
# copy is still sitting inside the data dataset. Leave as '' to discover it automatically.
# Being explicit is better than a rule about which copy wins: the run records which
# directory it used, so there is no question afterwards.
CODE_DIR = ''


def hdr(n, title):
    print('\n' + '=' * 72 + '\n%d. %s\n' % (n, title) + '=' * 72)


# ---------------------------------------------------------------- 1. find the code
hdr(1, 'Locate the code package')
# Every directory holding a complete copy of the modules. More than one means two
# datasets carry the code, and picking one silently would run whichever sorts first -
# which is exactly how a stale copy got used once already.
if CODE_DIR:
    GOOD = [Path(CODE_DIR)]
    for f in ('concept_audit.py', 'paths.py', 'preflight.py'):
        if not (GOOD[0] / f).is_file():
            raise RuntimeError('CODE_DIR is set to %s but %s is not there.' % (CODE_DIR, f))
    print('code dir     : taken from CODE_DIR, not discovered')
else:
    GOOD = []
CANDS = sorted({q.parent for q in INP.rglob('concept_audit.py')})
GOOD = GOOD or [c for c in CANDS
                if (c / 'paths.py').is_file() and (c / 'preflight.py').is_file()]
if not GOOD:
    tree = [str(q.relative_to(INP)) for q in sorted(INP.rglob('*'))[:40]]
    raise RuntimeError('The code dataset is not attached, or is incomplete.\n'
                       'Looked for concept_audit.py, paths.py and preflight.py together,\n'
                       'anywhere under /kaggle/input. What IS attached:\n  '
                       + '\n  '.join(tree or ['nothing']))
if len(GOOD) > 1:
    import hashlib
    # Hash the three modules together. Hashing only preflight.py once printed two
    # identical digests for copies that in fact differed in concept_audit.py, which is
    # worse than printing no hash at all.
    def _fp(c):
        h = hashlib.sha256()
        for f in ('concept_audit.py', 'paths.py', 'preflight.py'):
            h.update((c / f).read_bytes())
        return h.hexdigest()[:16]
    lines = ['%s  %s' % (_fp(c), c) for c in GOOD]
    same = len({l.split()[0] for l in lines}) == 1
    raise RuntimeError(
        '%d copies of the code package are attached%s.\nSet CODE_DIR at the top of this '
        'cell to the one you want, or detach the other:\n  %s'
        % (len(GOOD), ' (identical)' if same else ' (they DIFFER)', '\n  '.join(lines)))
sys.path.insert(0, str(GOOD[0]))
import paths as P, preflight as F, concept_audit as A

print('code package :', GOOD[0])
print('pinned model :', A.HF_REPO, A.HF_REV)

# CLEAR's own loader is used rather than a reimplementation: it applies the visual
# projection and the CXR preprocessing, both of which a hand-rolled encoder got wrong.
# --no-deps because CLEAR's dependency pins would otherwise reinstall torch.
try:
    import clear as _clear_pkg
    print('clear package: already installed')
except ImportError:
    import subprocess
    print('installing CLEAR pinned at %s ...' % A.CLEAR_COMMIT[:12])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                    'git+https://github.com/peterhan91/CLEAR.git@' + A.CLEAR_COMMIT],
                   check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ftfy', 'regex'],
                   check=True)
    import clear as _clear_pkg
    print('clear package: installed')
print('clear from    :', Path(_clear_pkg.__file__).parent)

# ---------------------------------------------------------------- 2. preflight
hdr(2, 'Preflight')
rep = F.run(check_weights=True)
(WORK / 'tables').mkdir(parents=True, exist_ok=True)
json.dump(rep, open(WORK / 'tables/preflight_kaggle.json', 'w'), indent=1, default=str)
for k, v in rep['runtime'].items():
    print('  %-22s %s' % (k, v))
print('  %-22s %s' % ('model revision', rep['model']['pinned_revision']))
res = (rep['model'].get('resolved') or {}).get('files', {})
for name, got in res.items():
    want = rep['model']['pinned_assets'].get(name, {})
    same = got.get('sha256') == want.get('sha256')
    print('  %-30s %12d B  sha256 %s' % (name, got.get('bytes', -1),
                                         'MATCHES PINNED' if same else 'DIFFERS'))
for w in rep['warnings']:
    print('  ! %s' % w)
if rep['verdict'] != 'GO':
    raise RuntimeError('PREFLIGHT STOP\n- ' + '\n- '.join(rep['blockers']))
print('\npreflight: GO')

# ---------------------------------------------------------------- 3. cohort gate
hdr(3, 'Cohort and partition gate')
p = P.resolve()
print(P.describe(p))
d = A.load_tune_only(p)
m, primary = A.concept_map()
names = [c['concept'] for c in primary]
missing, extra = A.check_images(p, d['pid'])
print()
print('  cases               :', len(d['pid']))
print('  TB positive         :', int(d['y'].sum()))
print('  previous TB         :', int(d['past_tb'].sum()))
print('  cohort fingerprint  :', d['fingerprint'][:16], '(matches pinned)')
print('  concepts (exact)    :', len(names), names)
print('  excluded            :', [c['concept'] for c in m['mapping']
                                  if c['concept_idx'] is None])
print('  images missing      :', len(missing))
print('  images not approved :', len(extra), '(never read)')
print('  expected shape      :', (len(d['pid']), len(names)))
if missing:
    raise RuntimeError('approved images are missing: %s' % missing[:5])
print('\ncohort gate: PASSED')

# ---------------------------------------------------------------- 4. inference
hdr(4, 'Inference  (downloads ~1.73 GB on first use, then encodes 1,352 images)')
import torch
import numpy as np

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('  device:', dev, '|', torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU')
t0 = time.perf_counter()
S, prov = A.encode_concepts(d['pid'], device=dev, out_dir=WORK)
print('\n  similarities %s in %.1f min' % (S.shape, (time.perf_counter() - t0) / 60))
print('  ' + json.dumps(prov, default=str))
if S.shape != (len(d['pid']), len(names)):
    raise RuntimeError('case-id alignment broken: %s for %d ids' % (S.shape, len(d['pid'])))
if not np.isfinite(S).all():
    raise RuntimeError('non-finite concept similarities')

# ---------------------------------------------------------------- 5. analysis
hdr(5, 'Diagnostic analysis')
rows, incr = A.analyse(d, S, names)
dist = A.distributions(d, S, names)
disc = A.discrimination(d, S, names)
for nm, r in (('concept_associations', rows), ('concept_incremental_r2', incr),
              ('concept_distributions', dist), ('concept_discrimination', disc)):
    A._write(WORK / 'tables' / (nm + '.csv'), r)
json.dump(A.run_manifest(d, p, prov), open(WORK / 'run_manifest.json', 'w'),
          indent=1, default=str)

import pandas as pd

R = pd.DataFrame(rows)
print(R[R.group == 'negatives (all)'][['concept', 'n', 'spearman_rho', 'ci95_lo',
                                       'ci95_hi', 'p_raw', 'p_bh_adjusted']]
      .to_string(index=False))
print('\n' + pd.DataFrame(incr).to_string(index=False))
print('\n' + pd.DataFrame(disc).to_string(index=False))
print('\n' + pd.DataFrame(dist).to_string(index=False))

# ---------------------------------------------------------------- 6. validate outputs
hdr(6, 'Validate the outputs')
EXPECT = {
    'tables/concept_associations.csv': {'ci95_hi', 'ci95_lo', 'concept', 'group', 'n',
                                        'p_bh_adjusted', 'p_raw', 'spearman_rho'},
    'tables/concept_incremental_r2.csv': {'concept', 'delta_r2', 'r2_clinical_only',
                                          'r2_with_concept'},
    'tables/concept_distributions.csv': None,
    'tables/concept_discrimination.csv': None,
    'tables/preflight_kaggle.json': None,
    'run_manifest.json': None,
}
bad = []
for rel, cols in EXPECT.items():
    f = WORK / rel
    if not f.is_file():
        bad.append('missing: ' + rel)
        continue
    if rel.endswith('.csv'):
        got = list(csv.DictReader(io.open(f, encoding='utf-8')))
        if cols and set(got[0]) != cols:
            bad.append('%s columns %s' % (rel, sorted(set(got[0]) ^ cols)))
        if len(got) > 200:
            bad.append('%s has %d rows - looks per-case' % (rel, len(got)))
    print('  %-42s %8d bytes' % (rel, f.stat().st_size))
if any(q.name.startswith('per_case') for q in WORK.rglob('*')):
    raise RuntimeError('a per-case table was written; do not export it')
if bad:
    raise RuntimeError(str(bad))

print('\noutput gate: PASSED - download the files listed above from the Output tab.')
print('Aggregate tables only. The per-case shards in _shards/ stay in the session.')
print('\ntotal elapsed: %.1f min' % ((time.perf_counter() - t_start) / 60))
print("""
Reminders for whoever writes this up:
  - a similarity to the sentence "fibrosis" is NOT fibrosis detection; each concept
    score is a weakly supervised proxy, not a validated detector
  - TUNE is exposed; this is exploratory, never validation
  - association is not causation: this does not show the classifier uses a visual cue
  - past_tb is clinical history, not an image annotation
  - cavity, nodule and reticular opacity were EXCLUDED, not measured
  - BH correction covers the 9 concepts in 'negatives (all)' only; subgroup p values
    are uncorrected and exploratory
  - the screening AUROC is unchanged at 0.7242
""")
