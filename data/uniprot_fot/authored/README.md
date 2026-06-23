# Handwritten UniProt FoT Reasoning Records

This directory is for accepted Codex-authored Forest-of-Thought and
Tree-of-Thought reasoning artifacts grounded in the graphified biological source
data under `data/uniprot_fot`.

Rules:

- Each record is an individually authored JSON file in `records/`.
- Source text can ground the problem, but copied external reasoning traces are
  forbidden.
- Biomedical records must be educational, mechanism-focused, and not
  patient-specific advice.
- Scripts may validate, count, hash, convert to JSONL/Parquet, and publish.
  Scripts should not generate accepted reasoning text.
- Accepted batches should be logged in `progress_log.md` and converted to
  Parquet before Hugging Face pushes.

Validation:

```bash
/home/iska/miniconda3/envs/iska-net-2/bin/python scripts/validate_uniprot_fot_authored.py \
  --records-dir data/uniprot_fot/authored/records \
  --output-dir data/uniprot_fot/authored/accepted \
  --write-parquet
```

The validator enforces authorship fields, directed edge integrity, active
support node references, GFlowNet reward metadata, continuous embedding fields,
and TokenGT/TropicalGT/ToricGT metadata.

## Source Anchors

Use `scripts/extract_uniprot_fot_anchors.py` to prepare compact source-row
summaries for the next handwritten batch. These anchors are not reasoning
records; they only expose provenance, labels, snippets, sequence lengths, GO/EC
fields, and suggested modality axes for author inspection.
