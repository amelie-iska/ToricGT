# ToricBLM FoT Full Graph Dataset

This directory contains the local ToricGT/ToricBLM graph/FoT training dataset
prepared for full graph-in/graph-out biomedical and scientific reasoning runs.

The curated training streams are:

- `toricblm_nonprotein_fot_splits/v2_parallel`: split-safe DNA, RNA, and
  small-molecule graph/FoT records with ConvexTok DAGs and training views.
- `uniprot_fot/splits/protrek_v2`: ProTrek-derived protein sequence/function
  split records.
- `uniprot_fot/structures`: coordinate-bearing AFDB, PDB modal, and PubChem3D
  records for structure-flow training. Missing structures are not imputed.
- `toricblm_late_mixed_fot_structure`: late-stage Codex 5.5 ToT/FoT and
  structure-association training shards.

Raw upstream data under `uniprot_fot/raw_sources` is represented locally by
symlinks and is excluded from Hub upload. Training uses the curated Parquet
records, manifests, and schema sidecars in this directory, not raw upstream
source paths.

