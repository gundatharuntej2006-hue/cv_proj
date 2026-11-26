# Comparative Experiments Summary (Section 14, Table 14)

**Status:** Pending measured, provenance-backed results and approved decisions.

The prior version of this section contained illustrative figures presented under a
"Sealed Test Result" heading. Those figures were not supported by run artifacts and
have been withdrawn. `p4_eval.comparisons.generate_table_14_comparisons` now refuses
to write Table 14 until a generator is connected to verified inputs.

The planned comparisons cover backbone, output head, lung crop, CLAHE,
calibration, triage threshold, detector link rule, OOD gate, and detector input
size. Each result must cite the input manifest and decision versions, run ID,
metric definition, sample counts, interval method and parameters, and output
artifact hash. Sealed results require a valid one-time gate token and must be
reported only after the approved protocol is complete.
