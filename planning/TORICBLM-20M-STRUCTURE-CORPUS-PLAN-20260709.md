# ToricBLM 20M+ Structure Corpus Plan

## Requirement

The structure phase target is not the small curated seed.  The target corpus is:

- 5M coordinate-bearing small-molecule structures or conformers.
- 5M coordinate-bearing protein structures.
- 5M coordinate-bearing RNA structures.
- 5M coordinate-bearing DNA structures.
- All available PDB biomolecular complexes, including protein-ligand, protein-RNA, protein-DNA, RNA/DNA, and mixed complexes.
- When coordinate structures are not locally available, train on the maximum available sequence/text/annotation data for the same modality.

The current local coordinate corpus does not satisfy this target.  It contains about 95K coordinate-bearing train rows:

- AFDB/protein: about 94.8K train rows across the seed AFDB, AFDB v6 full, and UniRef50 AFDB shards.
- PDB modal structures: hundreds of rows, because the earlier RCSB query was capped for smoke testing.
- PubChem3D: tens of rows, because only a tiny curated CID seed and one decoded CID from the old SELFIES dialect were used.

The old AFDB curation stopped because the disk guard hit the configured free-space floor.  This was correct for disk safety but means the corpus is not a full structure corpus.

## Corrections Implemented

1. The launcher now encodes the actual target:
   - `TORICBLM_MIN_STRUCTURE_COORDINATE_ROWS=20000000`
   - `TORICBLM_MIN_STRUCTURE_MODALITY_COUNTS=protein_afdb=5000000,pubchem_3d_or_ligand=5000000,pdb_rna=5000000,pdb_dna=5000000`
   - `TORICBLM_ALLOW_STRUCTURE_TARGET_SHORTFALL=0`

   Shortfall is allowed only because the instruction says to use the maximum available when the requested target is unavailable.  The manifest records exact missing counts instead of silently treating the partial corpus as full.

2. The structure readiness manifest now records:
   - total coordinate rows,
   - required modalities,
   - target modality counts,
   - exact target shortfalls,
   - whether shortfall is allowed or strict-fail.

3. The trainer now reads `coordinate_rows` from the manifest and logs target shortfall metadata.  The prior `coordinate_records:0` startup field was a schema mismatch.
4. Training must not start when the 20M total / 5M-per-modality coordinate targets are unmet.  Shortfall mode is now disabled by default and should only be used for inventory/audit commands, not training.

4. The structure stream now reports total configured candidate rows and the current shard separately.  `current_shard_rows:239` should not be confused with total structure rows.

5. The PDB curation script now supports paginated RCSB enumeration.  `--rcsb-query-limit 0` means all available IDs for each queried modality, enabling all-PDB-complex curation rather than a capped smoke sample.

6. The training config now includes full train-only raw sequence/text streams:
   - UniRef50/UniProt sequence rows: about 59.6M.
   - UniProt function text rows: about 464K.
   - RFAM rows: about 20.0M.
   - RNAcentral rows: about 7.34M.
   - DNA coding-region rows: about 1.68M.
   - PubChem SELFIES rows: about 10.0M.

   These rows are graphified by `GraphParquetTokenStream`; they are not treated as coordinate-native structure targets.

## What Must Happen Before a True 20M+ Structure Run

1. Free or provision storage.  The current filesystem has roughly 50GB free.  Even compressed Parquet coordinate shards for 20M rows will require much more than this unless the pipeline streams directly to remote storage and deletes local shards after upload.

2. Run AFDB curation to the protein target:
   - Use the UniRef/UniProt accession sources.
   - Keep `--resume`.
   - Keep raw mmCIF cache deletion enabled.
   - Emit Foldseek/3Di strings through ProTrek utilities.
   - Upload completed shards and delete local copies after verified upload if local disk remains the limiting factor.

3. Run all-PDB curation:
   - Use `scripts/build_pdb_modal_structure_fot_dataset.py --rcsb-query-limit 0`.
   - Include modalities `protein rna dna protein_rna protein_dna nucleic_acid complex ligand`.
   - Keep all complexes, not just modality smoke samples.

4. Fix PubChem scale:
   - The old PubChem10M SELFIES resolver only decoded 1 CID from a 2K scan because the installed decoder rejects old `[Branch]`/`[pop]` style SELFIES.
   - A direct CID source is now available from the local NatureLM PubChem mirror:
     `data/uniprot_fot/pubchem/pubchem_cids_5m_from_naturelm_cid_smiles.txt`.
   - Run PubChem3D SDF curation from real 3D conformer records only.  The curation script must not generate conformers locally or treat SMILES/SELFIES as coordinate targets.

5. ProTrek clustering:
   - Use sequence embeddings for protein, RNA, and DNA sequences where applicable.
   - Use structure/Foldseek-derived representations for coordinate rows.
   - Use function/text embeddings for annotations.
   - Split by connected components or conservative clustering in the union graph of sequence, structure, and function similarities.
   - For small molecules, use structure/conformer features plus text/property fingerprints and avoid near-duplicate scaffold leakage.

6. Training readiness:
   - Do not restart a run and call it full structure training until the readiness manifest reports the target or explicitly records the target shortfall policy.
   - If target shortfall remains, train on the maximum available coordinate structures plus all train-only graphified sequence/text rows.

## Current Decision

The stopped run was a partial-corpus run.  It should not be reported as satisfying the 20M+ structure requirement.  The next correct step is full-scale curation or streaming curation with verified remote upload, followed by a training launch whose manifest records either target satisfaction or exact shortfall.

## 2026-07-09 Active Curation Correction

The operational curation launcher has been corrected to match the requirement under tight local disk:

1. PDB/RCSB curation now runs first, before AFDB, so all available PDB complexes, protein, RNA, DNA, nucleic-acid, ligand, and mixed-complex coordinate rows are not starved by AFDB disk use.
2. PubChem3D curation now runs second from the 5M CID list extracted from the local NatureLM/PubChem mirror.  It emits only real PubChem3D SDF conformer coordinates and skips compounds already present when resumed.
3. AFDB curation runs third with a 5M target and `--resume`, preserving the existing AFDB shards while continuing the scan.
4. The launcher writes `data/uniprot_fot/manifests/toricblm_multimodal_structure_curation_inventory.json` by default.  The training launcher separately writes the strict readiness manifest; training remains blocked while the strict 20M / 5M-per-modality targets are unmet.
5. ProTrek readiness has been repaired by linking the existing local `ProTrek_35M` weights into `external/ProTrek/weights/ProTrek_35M`.  A strict two-row smoke test using real ProTrek sequence, text, and saved Foldseek/3Di structure embeddings succeeded.

The sibling cleanup pass removed unrelated output/log artifacts from `../iska-net` and `../iska-net-2`, while preserving raw biomolecular sources and processed graph data that feed ToricGT/ToricBLM.  After cleanup the current filesystem has about 931GB free.  The retained sibling data are:

- `../iska-net-2/data/raw/rcsb/cif`: local all-PDB mmCIF cache, used for all available PDB complexes and modal biomolecular coordinate rows.
- `../iska-net-2/data/raw/alphafold_db/cif`: local AFDB mmCIF cache, used before online AFDB fallback.
- `../iska-net-2/data/current_snapshot` and `../iska-net-2/dynamics`: relevant processed graph/dynamics sources for later ToricBLM phases.
- `../iska-net/data/raw_hf_bio_scale`: UniProt, UniRef, RNAcentral, RFAM, DNA, PubChem, and related text/sequence streams used by the training config.
- `../iska-net/data/raw/naturelm_pubchem_local/CID-SMILES.gz`: CID source for PubChem/PubChem3D audit and fallback.

Unrelated freed paths:

- `../iska-net/outputs`
- `../iska-net/wandb`
- `../iska-net/logs`
- `../iska-net-2/outputs`

## 2026-07-09 Minimum Strict Readiness Gate

The immediate full-structure minimum has been corrected to the currently attainable target the user specified after the stricter 5M-each modality request:

- 5M AFDB/protein coordinate rows.
- 5M PubChem3D/small-molecule coordinate rows.
- all locally available PDB mmCIF complexes and modal protein/RNA/DNA/nucleic-acid/ligand records.
- maximum available RNA/DNA sequence and annotation streams to compensate where million-scale coordinate structures do not exist locally.

The strict training watcher therefore now requires:

```text
protein_afdb >= 5,000,000
pubchem_3d_or_ligand >= 5,000,000
coordinate_rows >= 10,000,000
required PDB modalities present: pdb_protein, pdb_rna, pdb_dna, pdb_complex
```

The manifest still reports RNA/DNA coordinate availability and target shortfalls explicitly.  It should not claim 5M RNA or 5M DNA coordinate rows unless a real coordinate source is curated.

## 2026-07-09 PubChem3D Bulk Fix

The REST CID stream was too slow and initially accepted too few records because it was crawling CID batches.  The PubChem3D builder now supports the public PubChem3D FTP mirror:

```text
ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF
```

It processes real gzip SDF bulk files, writes atomic Parquet shards, deduplicates against existing output shards on resume, rejects flat/non-coordinate rows, and deletes downloaded bulk files after processing when configured.  A smoke test on `00000001_00025000.sdf.gz` emitted 50 real 3D coordinate rows with no parser fallbacks.  The live curation restarted from FTP bulk and the first full shard emitted 18,171 accepted PubChem3D conformer rows.

## 2026-07-09 AFDB Parallelization

Single-process AFDB curation was too slow for a 5M protein-structure target.  The AFDB builder now accepts explicit `--input-list` files, and `scripts/launch_afdb_parallel_curation.sh` partitions UniRef/UniProt Parquet shards across worker processes.  Each worker writes to an isolated output directory:

```text
data/uniprot_fot/structures/afdb_parallel_uniref50_full/worker_XX/{train,validation,test}
```

This avoids shared Parquet shard counters and shared progress files while keeping every row coordinate-native and Foldseek/3Di-derived when `AFDB_EMIT_3DI=1`.  The watcher, readiness manifest, ProTrek split command, and 170M structure-training config now include:

```text
data/uniprot_fot/structures/afdb_parallel_uniref50_full/worker_*/train/*.parquet
```

The sequential AFDB output directories remain valid and are still included, but the 5M protein target should be met by the parallel UniRef50 pass rather than by a single long-running process.

## 2026-07-09 AFDB EBI/GCS Correction

The per-accession AFDB fallback is real but too slow for a 5M coordinate target.  The official Google DeepMind AFDB documentation states that the full UniProt release is hosted in Google Cloud Storage and that the full dataset is roughly 23 TiB.  The local environment can list the public GCS bucket metadata anonymously, but object downloads return `storage.objects.get` permission errors without a Google Cloud account or service-account credentials.  The current `keys.txt` contains Hugging Face and GitHub tokens only; no Google/GCS credential material is present.  Therefore the full GCS AFDB route cannot be completed non-interactively on this machine until credentials are supplied.

The implemented public-data route now uses EMBL-EBI AFDB v6 proteome tar archives:

```text
https://ftp.ebi.ac.uk/pub/databases/alphafold/v6/
```

New implementation:

- `scripts/build_afdb_ebi_tar_structure_fot_dataset.py` streams official AFDB/EBI tar archives, extracts real `model_v6.cif.gz` members into temporary files, parses real CA/backbone coordinates, optionally emits real Foldseek/3Di sequences for ProTrek splitting, writes graph/FoT Parquet shards, and deletes raw archives/CIFs unless explicitly asked to keep them.
- `scripts/launch_afdb_ebi_tar_curation.sh` partitions the EBI tar set by approximate archive size into worker-specific URL lists and writes worker outputs under `data/uniprot_fot/structures/afdb_ebi_tar_v6/worker_XX`.
- The downloader prefers conda-installed `aria2c` with resumable multi-connection downloads and falls back to resumable `curl` if `aria2c` is unavailable.
- The strict watcher, ProTrek split command, structure-flow training glob, and graph-training glob now include both root-level and worker-level EBI AFDB shards.

Operational status:

- The old slow AFDB per-accession parallel session was stopped.
- A new EBI tar session is active with eight workers and resumable downloads.
- PubChem3D FTP curation and local all-PDB curation remain active.
- Training remains blocked by the strict readiness gate until the manifest reports the real required counts.  This is intentional; the run should not resume on a partial corpus and call it full.
- Hugging Face dataset upload is now also gated on strict readiness by default.  `RUN_HF_UPLOAD_IF_NOT_READY=1` is the explicit override if a partial/diagnostic upload is ever wanted, but the normal path is ready dataset, ProTrek split, Hugging Face upload, training launch.

## 2026-07-09 GCS AFDB Cap, Enzyme Bias, And Strict Trimodal Split Update

The AFDB source strategy has been changed from archive-order mirroring to a
capped, diversity-selected Google Cloud Storage pull.  The active production
path is:

1. `scripts/plan_afdb_gcs_diverse_accessions.py`
   - Scans local UniProt function-text rows and UniRef50 rows.
   - Selects UniProt accessions only; UniParc-only rows remain trainable as
     sequence/function records but are not requested from AFDB GCS.
   - Preserves sequence, function text, organism/taxon evidence, GO labels, EC
     evidence, protein names, and source row provenance in JSONL worker plans.
   - Caps the AFDB pull with `AFDB_GCS_TARGET_RECORDS`, currently `5,000,000`.
   - Upweights enzymes without collapsing the corpus: default enzyme target is
     40% of selected AFDB structures.  Within the enzyme-positive pool the
     target is 50% high catalytic/kinetic evidence, 25% medium evidence, and
     25% low or weakly characterized evidence when enough candidates exist.
   - The high/mid/low enzyme tiers are audit fields, not labels invented for
     biology: they are derived from explicit text evidence such as EC numbers,
     catalytic-activity descriptions, `GO:0003824`, kinetic terms (`kcat`,
     `Km`, `Vmax`, turnover, catalytic efficiency), and activity qualifiers.

2. `scripts/build_afdb_gcs_structure_fot_dataset.py`
   - Reads the selected JSONL plan.
   - Batch-copies only selected files from
     `gs://public-datasets-deepmind-alphafold-v4`.
   - Parses real AFDB mmCIF coordinates with the same coordinate graph/FoT
     schema as the previous AFDB builders.
   - Emits real Foldseek/3Di strings when `AFDB_EMIT_3DI=1`, so the downstream
     ProTrek splitter can use an actual structure modality after raw mmCIF
     cache cleanup.
   - Writes `enzyme_tier`, `enzyme_evidence_score`, and `selection_json` into
     every emitted row for later analysis and sampling audits.

3. `scripts/split_with_protrek_trimodal_streaming.py`
   - Runs actual ProTrek sequence, text, and Foldseek/3Di structure embedding
     inference on GPU.
   - Clusters only rows where all three modalities exist.
   - Writes a split map and, when `--write-full-rows` is set, full
     train/validation/test Parquet shards that carry the ProTrek split.
   - Rows with sequence and function but no structure remain trainable through
     the ordinary protein graph/FoT streams.  They are not counted as trimodal
     and do not satisfy the protein-structure leakage split target.

4. `scripts/watch_structure_curation_upload_and_train.sh`
   - Includes AFDB GCS worker outputs in strict readiness.
   - Requires the streaming ProTrek trimodal report before training launch.
   - Exports `PROTREK_STRUCTURE_TRAIN_GLOB` so the structure-flow training path
     can consume full-row ProTrek train shards rather than raw accession-hash
     AFDB shards.

5. FoT training/inference verification update
   - `src/toricgt/embedding_forest_of_thought.py` now uses an explicit sparse
     forest topology rather than a chain-like interleaving of hidden states.
     The trace contains forest-root, tree-expansion, self-correction, and
     consensus-vote edges, matching the operating concepts in
     `external/Forest-of-Thought`.
   - `amelie-iska/parameter-golf/train_gpt.py` now applies the same FoT head to
     selected long graph/FoT segments through `LONG_ENTRY_FOT_LOSS_WEIGHT`, so
     biomedical FoT rows train sparse forest activation, UCB-like branching,
     correction, trajectory balance, and consensus directly.
   - `scripts/export_oai_fot_trace.py` exports checkpoint FoT traces for
     inference/reporting and has a smoke mode for topology tests.
   - The launcher corpus audit confirms sequence/function-only rows remain
     trainable even when no structure field is present.  These rows are not
     counted as trimodal and do not satisfy structure-flow readiness targets.

The slow EBI tar AFDB session was stopped after GCS authentication succeeded,
because archive-order ingestion is not diversity selected and is not the right
source for the revised capped `~5M` AFDB requirement.  Partial EBI rows already
written can remain as supplementary structure examples, but the main AFDB
protein-structure target is now the GCS-selected corpus.  The active GCS route
uses the authenticated `gsutil` path, writes a selected-accession worker plan,
and blocks training until strict readiness plus ProTrek trimodal splitting pass.
