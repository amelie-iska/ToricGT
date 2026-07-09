# Handwritten UniProt FoT Progress Log

## 2026-06-23

- Created the handwritten FoT acceptance workspace.
- Added validation-only tooling for authored JSON records.
- Seeded the first accepted records as small, inspectable technical artifacts.
- Expanded the first authored mini-batch to 8 records covering enzyme mechanism,
  cellulolytic synergy, HLA coding-region graph reasoning, RFAM 5S rRNA
  structure reasoning, SELFIES medicinal-chemistry graph scoping, and UniRef GO
  mechanical-protein annotation.
- Added a 144-row source-anchor pool for the next authored records, balanced
  across UniProt function text, UniRef50 GO clusters, RFAM, RNAcentral, coding
  regions, and PubChem SELFIES. These anchors are provenance summaries only.
- Current target remains iterative batches of 200 authored records, with Parquet
  conversion and Hugging Face push milestones after validation.
