# Active Goal: UniProt FoT/ToricGT Data Campaign

Build a durable biological graph-reasoning data workspace for ToricGT on branch
`toricblm-data`, centered at `/home/iska/Documents/amelie/bio/ToricGT/data`.
The source evidence is the local bio-scale Parquet mirror at
`/home/iska/Documents/amelie/bio/iska-net/data/raw_hf_bio_scale`, especially
UniProt function text and UniRef50 sequence/annotation records, but also RFAM,
RNAcentral, DNA coding-region, and PubChem SELFIES shards. Preserve provenance,
avoid unnecessary duplication on the nearly full filesystem, and make every
derived artifact inspectable, hashable, schema-valid, and suitable for Parquet
publication.

First, graphify the raw datasets into training rows that match the ToricGT
papers' graph-first contract: stable graph JSON, node and edge token streams,
directed source-field dependencies, clustered splits, content hashes, and rich
metadata for TokenGT/TropicalGT/ToricGT loaders. For each biological row, keep
the original sequence and source annotations together. UniProt-style rows should
join sequence, protein/function text, GO labels, EC labels, taxonomy, UniRef
cluster fields, representative accessions, sequence hashes, and AFDB/PDB-style
lookup hooks whenever present or safely derivable. DNA rows should expose
accession, organism, exons, introns, proteins, and sequence. RNA rows should
expose UPI/id, type/family/clan, description, and sequence. PubChem SELFIES rows
should be treated as medicinal-chemistry molecular strings that can later be
lifted into atom/bond graphs if RDKit/selfies tooling is added.

Second, add Forest-of-Thought structure without pretending that deterministic
source conversion is authored reasoning. Each graphified row should carry a
forest sidecar whose trees correspond to sequence evidence, annotation evidence,
structure lookup evidence, and future design-condition evidence. Include sparse
activation fields, consensus node sets, trajectory-balance metadata,
continuous/hybrid latent coordinate proxies, GraphCG axis hints, tropical active
support nodes, tropical margin proxies, toric phase-basis tags, and ConvexTok
byte-packing notes. These sidecars should guide hidden-space FoT/GFlowNet
training and test-time scaling without copying or fabricating reasoning traces.

Third, create the next authored reasoning layer in reviewed batches. Each
accepted authored record should be a rigorous FoT/ToT technical artifact
grounded in UniProt-style evidence: difficult biochemistry, biomedicine,
biophysics, cell biology, pathway reasoning, gene regulatory and adaptive
systems, RNA medicine design, medicinal chemistry, molecular dynamics,
neurobiology, morphology/phenotype/perturbation reasoning, expression and
abundance reasoning, scientific coding, boolean circuit and graph grammar
problems in biological settings, and advanced mathematics useful for the model.
The forest should have multiple trees, genuine branch/evaluate/compare/merge
logic, occasional rejected branches, safety-aware biomedical framing, and clear
final targets. Scripts may validate, count, hash, shard, convert to Parquet, and
publish accepted records; scripts must not mass-invent accepted reasoning text.

Fourth, prepare Hugging Face publication for `AmelieSchreiber/uniprot_fot`.
Publication should be Parquet-first, include dataset cards/manifests/split
reports, preserve source provenance, and avoid uploading raw mixed-license data
unless licensing has been reviewed. Push periodically after each 1K accepted
authored entries or after meaningful graphified-data milestones. Any key usage
must remain secret; never print tokens or commit secrets. Prefer existing
authenticated HF tooling in the `iska-net-2` or `tokengt` conda environment.

Completion requires evidence, not intention: raw source organization under
ToricGT/data, graphified JSONL and Parquet outputs, successful validation of
JSON parseability and directed edge integrity, manifests proving source shard
counts and row counts, stable split clusters, records containing GFlowNet,
FoT, continuous embedding, TokenGT, TropicalGT, and ToricGT fields, authored
reasoning batches accepted in logs, and verified Parquet publication steps for
the HF dataset repository.
