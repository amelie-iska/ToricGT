"""ToricGT sidecar losses for the OAI Parameter-Golf baseline.

This module deliberately leaves the OAI FineWeb language-model stream
untouched.  It consumes curated graph Parquet rows as an auxiliary stream and
trains graph/analogy/retrieval heads from the same GPT hidden states.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .combinatorial_toric_metrics import CombinatorialToricConfig, combinatorial_toric_cca_topology_loss
from .derived_category_metrics import DerivedCategoryConfig, derived_category_feature_summary
from .koszul_persistence import KoszulPersistenceConfig, koszul_persistence_loss
from .trajectory_memory import TrajectoryMemoryConfig, TrajectoryRetrievalHead
from .toric_bgg import ToricBGGConfig, ToricBGGProbe
from .toric_geometry_tasks import LowRankToricGeometryProbe, ToricGeometryConfig
from .toric_vector_bundles import ToricVectorBundleConfig, ToricVectorBundleProbe


class GraphParquetTokenStream:
    """Stream curated graph rows as tokenizer chunks for graph LM training.

    Every row is graphified before SentencePiece encoding.  Rows with a
    ``graph_json`` payload are serialized as explicit node and edge token
    records.  DAG-like payloads use causal topological node order and decode
    each edge after its endpoints are visible; cyclic or edgeless payloads use a
    deterministic random reveal order.  Rows without graph JSON are converted to
    a linear causal text graph, so the OAI adaptation never silently falls back
    to plain unstructured text for this stream.
    """

    TEXT_COLUMNS = (
        "text",
        "question",
        "reasoning",
        "solution",
        "answer",
        "annotation_text",
        "graph_json",
        "metadata_json",
        "forest_json",
        "thought_forest_json",
        "convextok_dag_json",
        "training_views_json",
        "sequence",
        "function",
        "entry",
        "entry_name",
        "protein_name",
        "id",
        "name",
        "updated",
        "member_count",
        "common_taxon",
        "common_taxon_id",
        "seed_id",
        "go_mf",
        "go_bp",
        "go_cc",
        "member_ids",
        "rep_member_id",
        "rep_member_id_type",
        "rep_organism",
        "rep_organism_tax_id",
        "rep_protein_name",
        "rep_accessions",
        "rep_uniparc_id",
        "rep_uniref90_id",
        "rep_uniref100_id",
        "rep_is_seed",
        "sequence_length",
        "sequence_crc64",
        "sequence_xxh128",
        "accession",
        "organism",
        "introns",
        "exons",
        "proteins",
        "SELFIES",
        "selfies",
        "family",
        "clan",
        "description",
        "upi",
        "type",
    )

    def __init__(self, pattern: str, tokenizer: Any, seq_len: int, batch_size: int):
        self.pattern = pattern
        self.files = self._expand_patterns(pattern)
        if not self.files:
            raise FileNotFoundError(f"No graph Parquet files found for pattern: {pattern}")
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.file_idx = 0
        self.row_idx = 0
        self.rows: list[str] = []
        self.token_buffer: list[int] = []
        self.policy_counts: dict[str, int] = {}
        self.rows_graphified = 0
        self.parquet_file: Any = None
        self.batch_iter: Any = None
        self.active_columns: list[str] = []
        self.batch_size_rows = 256
        self.skipped_files: list[str] = []
        self._load_file()

    @staticmethod
    def _expand_patterns(pattern: str) -> list[Path]:
        pieces = [piece.strip() for piece in re.split(r"[,;]", pattern) if piece.strip()]
        files: list[Path] = []
        seen: set[str] = set()
        for piece in pieces or [pattern]:
            for item in sorted(glob.glob(piece)):
                path = Path(item)
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    files.append(path)
        return files

    def refresh_files(self) -> None:
        refreshed = self._expand_patterns(self.pattern)
        if refreshed:
            current = self.files[self.file_idx] if self.files and self.file_idx < len(self.files) else None
            self.files = refreshed
            if current in self.files:
                self.file_idx = self.files.index(current)
            else:
                self.file_idx = min(self.file_idx, max(0, len(self.files) - 1))

    def _encode(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer.encode(text, out_type=int)
        except TypeError:
            ids = self.tokenizer.encode(text)
        return [int(token_id) for token_id in ids]

    @staticmethod
    def _stable_key(value: Any, record_id: str) -> int:
        digest = hashlib.blake2b(f"{record_id}:{value}".encode("utf-8", errors="ignore"), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    @staticmethod
    def _safe_attr(value: Any, *, max_chars: int = 80) -> str:
        text = str(value if value is not None else "").strip()
        if not text:
            return "unknown"
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9_.:/#@+-]", "_", text)
        return text[:max_chars] or "unknown"

    @staticmethod
    def _payload_text(value: Any, *, max_chars: int = 420) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
        else:
            try:
                text = json.dumps(value, ensure_ascii=True, sort_keys=True)
            except Exception:
                text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    @staticmethod
    def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
        for key in keys:
            if key not in row:
                continue
            value = row[key]
            if value is None:
                continue
            if isinstance(value, str) and value == "":
                continue
            if isinstance(value, (list, tuple, dict)) and not value:
                continue
            return value
        return default

    @staticmethod
    def _as_list(value: Any, *, limit: int = 32) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        if isinstance(value, list):
            return value[:limit]
        if isinstance(value, tuple):
            return list(value[:limit])
        if isinstance(value, str) and not value:
            return []
        return [value]

    @staticmethod
    def _sequence_chunks(sequence: Any, *, max_chunks: int = 20, chars_per_chunk: int = 420) -> list[dict[str, Any]]:
        if not isinstance(sequence, str) or not sequence:
            return []
        chunks: list[dict[str, Any]] = []
        for index, start in enumerate(range(0, len(sequence), chars_per_chunk)):
            if index >= max_chunks:
                chunks.append(
                    {
                        "index": index,
                        "start": start,
                        "end": len(sequence),
                        "length": max(0, len(sequence) - start),
                        "text": f"remaining sequence omitted from serialized graph node text; length={max(0, len(sequence) - start)}",
                    }
                )
                break
            end = min(len(sequence), start + chars_per_chunk)
            chunks.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "text": sequence[start:end],
                }
            )
        return chunks

    @staticmethod
    def _selfies_tokens(value: Any, *, max_tokens: int = 96) -> list[str]:
        if not isinstance(value, str) or not value:
            return []
        return re.findall(r"\[[^\]]+\]", value)[:max_tokens]

    @classmethod
    def _node_id(cls, node: Any, index: int) -> str:
        if isinstance(node, dict):
            raw = cls._first_present(node, ("id", "node_id", "local_id", "name", "key"), f"n{index}")
        else:
            raw = f"n{index}"
        return cls._safe_attr(raw, max_chars=72)

    @classmethod
    def _normalize_nodes_edges(
        cls,
        payload: Any,
        fallback_text: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        nodes_raw: Any = []
        edges_raw: Any = []
        if isinstance(payload, dict):
            nodes_raw = payload.get("nodes", [])
            edges_raw = payload.get("edges", [])
        if isinstance(nodes_raw, dict):
            nodes_raw = [{"id": key, **(value if isinstance(value, dict) else {"text": value})} for key, value in nodes_raw.items()]
        if not isinstance(nodes_raw, list):
            nodes_raw = []
        if not isinstance(edges_raw, list):
            edges_raw = []

        nodes: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, node in enumerate(nodes_raw):
            node_dict = node if isinstance(node, dict) else {"text": node}
            node_id = cls._node_id(node_dict, index)
            if node_id in seen:
                node_id = f"{node_id}_{index}"
            seen.add(node_id)
            node_type = cls._safe_attr(
                cls._first_present(node_dict, ("type", "kind", "role", "label"), "node"),
                max_chars=48,
            )
            text = cls._payload_text(
                cls._first_present(
                    node_dict,
                    ("text", "payload", "content", "value", "question", "answer", "label", "name"),
                    node_type,
                )
            )
            nodes.append({"id": node_id, "type": node_type, "text": text})

        edges: list[dict[str, str]] = []
        for index, edge in enumerate(edges_raw):
            edge_dict = edge if isinstance(edge, dict) else {}
            src = cls._safe_attr(cls._first_present(edge_dict, ("source", "src", "from", "u", "tail"), ""))
            dst = cls._safe_attr(cls._first_present(edge_dict, ("target", "dst", "to", "v", "head"), ""))
            if not src or not dst or src == "unknown" or dst == "unknown":
                continue
            edge_id = cls._safe_attr(cls._first_present(edge_dict, ("id", "edge_id", "key"), f"e{index}"))
            edge_type = cls._safe_attr(cls._first_present(edge_dict, ("type", "kind", "role", "label"), "edge"), max_chars=48)
            text = cls._payload_text(cls._first_present(edge_dict, ("text", "payload", "content", "value", "label"), edge_type))
            edges.append({"id": edge_id, "source": src, "target": dst, "type": edge_type, "text": text})
            for endpoint in (src, dst):
                if endpoint not in seen:
                    seen.add(endpoint)
                    nodes.append({"id": endpoint, "type": "inferred_endpoint", "text": endpoint})

        if not nodes:
            text = fallback_text.strip() or "empty record"
            chunks = cls._chunk_text(text)
            nodes = [{"id": f"n{idx}", "type": "text_span", "text": chunk} for idx, chunk in enumerate(chunks)]
            edges = [
                {"id": f"e{idx}", "source": f"n{idx}", "target": f"n{idx + 1}", "type": "next_text_span", "text": "next"}
                for idx in range(max(0, len(nodes) - 1))
            ]
        return nodes, edges

    @classmethod
    def _raw_bio_nodes_edges(cls, row: dict[str, Any], record_id: str) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
        bio_keys = (
            "sequence",
            "SELFIES",
            "selfies",
            "function",
            "protein_name",
            "entry",
            "entry_name",
            "go_mf",
            "go_bp",
            "go_cc",
            "introns",
            "exons",
            "proteins",
            "family",
            "clan",
            "description",
            "upi",
            "accession",
            "rep_accessions",
        )
        if not any(key in row for key in bio_keys):
            return None

        nodes: list[dict[str, str]] = []
        edges: list[dict[str, str]] = []
        seen: set[str] = set()
        edge_index = 0

        def add_node(node_id: str, node_type: str, text: Any, *, max_chars: int = 900) -> str:
            safe = cls._safe_attr(node_id, max_chars=72)
            if safe in seen:
                base = safe
                suffix = 1
                while f"{base}_{suffix}" in seen:
                    suffix += 1
                safe = f"{base}_{suffix}"
            seen.add(safe)
            nodes.append(
                {
                    "id": safe,
                    "type": cls._safe_attr(node_type, max_chars=48),
                    "text": cls._payload_text(text, max_chars=max_chars),
                }
            )
            return safe

        def add_edge(src: str, dst: str, edge_type: str, text: Any = "") -> None:
            nonlocal edge_index
            if src not in seen or dst not in seen:
                return
            edges.append(
                {
                    "id": f"e{edge_index}",
                    "source": src,
                    "target": dst,
                    "type": cls._safe_attr(edge_type, max_chars=48),
                    "text": cls._payload_text(text or edge_type, max_chars=360),
                }
            )
            edge_index += 1

        entry_id = cls._first_present(row, ("entry", "id", "accession", "upi", "seed_id", "rep_member_id", "entry_name", "name"), record_id)
        root = add_node("record", "biomedical_record", f"record={entry_id}; source={record_id}")

        sequence = row.get("sequence")
        if isinstance(sequence, str) and sequence:
            alphabet = set(sequence.upper())
            if alphabet <= set("ACGTNURYKMSWBDHV.-"):
                seq_type = "nucleotide_or_rna_sequence"
            elif alphabet <= set("ACDEFGHIKLMNPQRSTVWYXBZUOJ*-"):
                seq_type = "protein_sequence"
            else:
                seq_type = "mixed_sequence"
            seq_root = add_node("sequence", seq_type, f"length={len(sequence)}")
            add_edge(root, seq_root, "has_sequence", f"length={len(sequence)}")
            previous = seq_root
            for chunk in cls._sequence_chunks(sequence):
                chunk_id = add_node(
                    f"seq_{chunk['index']}",
                    "sequence_chunk",
                    f"range={chunk['start']}:{chunk['end']} length={chunk['length']} {chunk['text']}",
                    max_chars=760,
                )
                add_edge(seq_root, chunk_id, "has_sequence_chunk", f"range={chunk['start']}:{chunk['end']}")
                add_edge(previous, chunk_id, "next_sequence_chunk", f"chunk={chunk['index']}")
                previous = chunk_id

        selfies = cls._first_present(row, ("SELFIES", "selfies"))
        if isinstance(selfies, str) and selfies:
            molecule = add_node("molecule", "selfies_molecule", f"SELFIES length={len(selfies)}")
            add_edge(root, molecule, "has_molecule")
            previous = molecule
            for idx, token in enumerate(cls._selfies_tokens(selfies)):
                tok = add_node(f"selfies_{idx}", "selfies_token", token, max_chars=96)
                add_edge(molecule, tok, "has_selfies_token", f"position={idx}")
                add_edge(previous, tok, "next_selfies_token", f"position={idx}")
                previous = tok

        scalar_fields = (
            ("entry_name", "entry_name"),
            ("protein_name", "protein_name"),
            ("function", "function_annotation"),
            ("name", "cluster_name"),
            ("updated", "updated"),
            ("member_count", "member_count"),
            ("common_taxon", "common_taxon"),
            ("common_taxon_id", "common_taxon_id"),
            ("seed_id", "seed_id"),
            ("rep_member_id", "representative_member"),
            ("rep_member_id_type", "representative_member_type"),
            ("rep_organism", "representative_organism"),
            ("rep_organism_tax_id", "representative_organism_tax_id"),
            ("rep_protein_name", "representative_protein_name"),
            ("rep_uniparc_id", "representative_uniparc"),
            ("rep_uniref90_id", "representative_uniref90"),
            ("rep_uniref100_id", "representative_uniref100"),
            ("rep_is_seed", "representative_is_seed"),
            ("sequence_length", "sequence_length"),
            ("sequence_crc64", "sequence_crc64"),
            ("sequence_xxh128", "sequence_xxh128"),
            ("accession", "accession"),
            ("organism", "organism"),
            ("family", "rfam_family"),
            ("clan", "rfam_clan"),
            ("description", "description"),
            ("upi", "rnacentral_upi"),
            ("type", "rna_type"),
        )
        for key, node_type in scalar_fields:
            value = row.get(key)
            if value in (None, ""):
                continue
            node = add_node(key, node_type, value)
            add_edge(root, node, f"has_{node_type}")

        for key, node_type in (("go_mf", "go_molecular_function"), ("go_bp", "go_biological_process"), ("go_cc", "go_cellular_component")):
            for idx, value in enumerate(cls._as_list(row.get(key), limit=48)):
                node = add_node(f"{key}_{idx}", node_type, value, max_chars=180)
                add_edge(root, node, f"has_{node_type}")

        for idx, value in enumerate(cls._as_list(row.get("rep_accessions"), limit=24)):
            node = add_node(f"rep_accession_{idx}", "representative_accession", value, max_chars=180)
            add_edge(root, node, "has_representative_accession")

        for idx, value in enumerate(cls._as_list(row.get("member_ids"), limit=24)):
            node = add_node(f"member_{idx}", "cluster_member", value, max_chars=180)
            add_edge(root, node, "has_cluster_member")

        for feature_key, feature_type in (("introns", "intron"), ("exons", "exon"), ("proteins", "translated_protein")):
            for idx, feature in enumerate(cls._as_list(row.get(feature_key), limit=24)):
                if isinstance(feature, dict):
                    label = {key: feature.get(key) for key in ("gene", "start", "end", "sequence") if feature.get(key) not in (None, "")}
                else:
                    label = feature
                node = add_node(f"{feature_key}_{idx}", feature_type, label, max_chars=460)
                add_edge(root, node, f"has_{feature_type}", f"{feature_key}[{idx}]")

        return (nodes, edges) if nodes else None

    @staticmethod
    def _chunk_text(text: str, *, max_nodes: int = 24, chars_per_node: int = 360) -> list[str]:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return ["empty record"]
        chunks = [clean[i : i + chars_per_node].strip() for i in range(0, len(clean), chars_per_node)]
        return [chunk for chunk in chunks[:max_nodes] if chunk] or [clean[:chars_per_node]]

    @classmethod
    def _causal_ranks(
        cls,
        nodes: list[dict[str, str]],
        edges: list[dict[str, str]],
        record_id: str,
        *,
        force_linear: bool = False,
    ) -> tuple[dict[str, int], str]:
        node_ids = [node["id"] for node in nodes]
        if force_linear:
            return {node_id: idx for idx, node_id in enumerate(node_ids)}, "linear_causal"
        valid = [(edge["source"], edge["target"]) for edge in edges if edge["source"] in node_ids and edge["target"] in node_ids]
        if not valid:
            ordered = sorted(node_ids, key=lambda node_id: cls._stable_key(node_id, record_id))
            return {node_id: idx for idx, node_id in enumerate(ordered)}, "random_order_no_edges"
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
        for src, dst in valid:
            adjacency[src].append(dst)
            indegree[dst] += 1
        frontier = sorted([node_id for node_id in node_ids if indegree[node_id] == 0], key=node_ids.index)
        ordered: list[str] = []
        while frontier:
            node_id = frontier.pop(0)
            ordered.append(node_id)
            for dst in adjacency[node_id]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    frontier.append(dst)
            frontier.sort(key=node_ids.index)
        if len(ordered) == len(node_ids):
            return {node_id: idx for idx, node_id in enumerate(ordered)}, "causal_topological_dag"
        ordered = sorted(node_ids, key=lambda node_id: cls._stable_key(node_id, record_id))
        return {node_id: idx for idx, node_id in enumerate(ordered)}, "random_order_cycle"

    @classmethod
    def _serialize_graph(
        cls,
        nodes: list[dict[str, str]],
        edges: list[dict[str, str]],
        record_id: str,
        *,
        force_linear: bool = False,
    ) -> tuple[str, str]:
        ranks, policy = cls._causal_ranks(nodes, edges, record_id, force_linear=force_linear)
        node_lookup = {node["id"]: node for node in nodes}
        ordered_nodes = sorted(nodes, key=lambda node: (ranks.get(node["id"], 10**9), node["id"]))
        ordered_edges = sorted(
            edges,
            key=lambda edge: (
                max(ranks.get(edge["source"], 10**9), ranks.get(edge["target"], 10**9)),
                ranks.get(edge["source"], 10**9),
                ranks.get(edge["target"], 10**9),
                edge["id"],
            ),
        )
        lines = [
            (
                f"<graph record_id={cls._safe_attr(record_id, max_chars=96)} "
                f"decode_policy={policy} node_count={len(nodes)} edge_count={len(edges)} "
                "edge_token_decoding=after_endpoint_nodes>"
            )
        ]
        for node in ordered_nodes:
            lines.append(
                f"<node_token decode_rank={ranks.get(node['id'], 0)} node_id={node['id']} node_type={node['type']}> "
                f"{node['text']} </node_token>"
            )
        for edge_index, edge in enumerate(ordered_edges):
            src_rank = ranks.get(edge["source"], 0)
            dst_rank = ranks.get(edge["target"], 0)
            decode_rank = max(src_rank, dst_rank)
            endpoint_text = ""
            if edge["source"] in node_lookup and edge["target"] in node_lookup:
                endpoint_text = f" source_type={node_lookup[edge['source']]['type']} target_type={node_lookup[edge['target']]['type']}"
            lines.append(
                f"<edge_token decode_rank={decode_rank} edge_order={edge_index} edge_id={edge['id']} "
                f"source={edge['source']} target={edge['target']} edge_type={edge['type']}{endpoint_text}> "
                f"{edge['text']} </edge_token>"
            )
        lines.append("</graph>")
        return "\n".join(lines), policy

    @classmethod
    def _graphify_row(cls, row: dict[str, Any], record_id: str) -> tuple[str, str]:
        fallback_parts = []
        for key in cls.TEXT_COLUMNS:
            value = row.get(key)
            if value not in (None, ""):
                fallback_parts.append(f"{key}: {cls._payload_text(value, max_chars=2400)}")
        fallback_text = "\n".join(part for part in fallback_parts if part)
        graph_raw = row.get("graph_json")
        if graph_raw not in (None, ""):
            try:
                payload = json.loads(graph_raw) if isinstance(graph_raw, str) else graph_raw
            except Exception:
                payload = None
            if payload is not None:
                nodes, edges = cls._normalize_nodes_edges(payload, fallback_text)
                return cls._serialize_graph(nodes, edges, record_id)
        raw_bio_graph = cls._raw_bio_nodes_edges(row, record_id)
        if raw_bio_graph is not None:
            nodes, edges = raw_bio_graph
            return cls._serialize_graph(nodes, edges, record_id)
        chunks = cls._chunk_text(fallback_text)
        nodes = [{"id": f"n{idx}", "type": "text_span", "text": chunk} for idx, chunk in enumerate(chunks)]
        edges = [
            {"id": f"e{idx}", "source": f"n{idx}", "target": f"n{idx + 1}", "type": "next_text_span", "text": "next"}
            for idx in range(max(0, len(nodes) - 1))
        ]
        return cls._serialize_graph(nodes, edges, record_id, force_linear=True)

    def describe(self) -> str:
        policies = ",".join(f"{key}:{value}" for key, value in sorted(self.policy_counts.items()))
        return (
            f"files:{len(self.files)} current_rows:{len(self.rows)} graphified_rows:{self.rows_graphified} "
            f"skipped_files:{len(self.skipped_files)} policies:{policies or 'none'}"
        )

    def _load_file(self) -> None:
        import pyarrow.parquet as pq

        attempts = 0
        while attempts < len(self.files):
            path = self.files[self.file_idx]
            try:
                self.parquet_file = pq.ParquetFile(path)
                schema_names = set(self.parquet_file.schema_arrow.names)
            except Exception as exc:
                self.skipped_files.append(f"{path}:{exc}")
                self.file_idx = (self.file_idx + 1) % len(self.files)
                attempts += 1
                continue
            columns = [name for name in self.TEXT_COLUMNS if name in schema_names]
            if not columns:
                self.skipped_files.append(f"{path}:no_graphifiable_columns")
                self.file_idx = (self.file_idx + 1) % len(self.files)
                attempts += 1
                continue
            self.active_columns = columns
            self.batch_iter = self.parquet_file.iter_batches(batch_size=self.batch_size_rows, columns=columns)
            self.rows = []
            self.row_idx = 0
            self._load_next_batch()
            return
        raise ValueError(f"No readable graph Parquet shards with usable columns for pattern: {self.pattern}")

    def _load_next_batch(self) -> None:
        import pyarrow as pa

        if self.batch_iter is None:
            self._load_file()
            return
        path = self.files[self.file_idx]
        while True:
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                self._advance_file()
                return
            table = pa.Table.from_batches([batch])
            rows: list[str] = []
            offset_base = self.rows_graphified
            for idx, row in enumerate(table.to_pylist()):
                text, policy = self._graphify_row(row, f"{path.name}:{offset_base + idx}")
                if text.strip():
                    rows.append(text)
                    self.rows_graphified += 1
                    self.policy_counts[policy] = self.policy_counts.get(policy, 0) + 1
            if rows:
                self.rows = rows
                self.row_idx = 0
                return

    def _advance_file(self) -> None:
        self.refresh_files()
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self._load_file()

    def _append_next_row(self) -> None:
        if self.row_idx >= len(self.rows):
            self._load_next_batch()
        text = self.rows[self.row_idx]
        self.row_idx += 1
        ids = self._encode(text)
        sep = self._encode("\n\n")
        self.token_buffer.extend(ids + sep)

    def next_batch(self, device: torch.device, step: int | None = None) -> tuple[Tensor, Tensor]:
        needed = self.batch_size * self.seq_len + 1
        while len(self.token_buffer) < needed:
            self._append_next_row()
        chunk = self.token_buffer[:needed]
        del self.token_buffer[: self.batch_size * self.seq_len]
        tokens = torch.tensor(chunk, dtype=torch.int64)
        x = tokens[:-1].reshape(self.batch_size, self.seq_len)
        y = tokens[1:].reshape(self.batch_size, self.seq_len)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class ScheduledGraphParquetTokenStream:
    """Switch graph LM/sidecar rows to a late-stage train-only stream.

    The base stream remains the default curated graph corpus.  Once
    ``late_start_step`` is reached, a deterministic ratio gate draws rows from
    the late stream.  This is used for train-only Tree/Forest-of-Thought data:
    the validation BPB stream is not affected.
    """

    def __init__(
        self,
        base_pattern: str,
        late_pattern: str,
        tokenizer: Any,
        seq_len: int,
        batch_size: int,
        *,
        late_start_step: int,
        late_mix_ratio: float = 1.0,
    ) -> None:
        self.base = GraphParquetTokenStream(base_pattern, tokenizer, seq_len, batch_size)
        self.late = GraphParquetTokenStream(late_pattern, tokenizer, seq_len, batch_size) if late_pattern else None
        self.late_start_step = max(0, int(late_start_step))
        self.late_mix_ratio = max(0.0, min(1.0, float(late_mix_ratio)))
        self.counter = 0
        self.last_stream = "base"

    def _use_late(self, step: int | None) -> bool:
        if self.late is None:
            return False
        if step is None or int(step) < self.late_start_step:
            return False
        if self.late_mix_ratio >= 1.0:
            return True
        if self.late_mix_ratio <= 0.0:
            return False
        bucket = self.counter % 100
        return bucket < int(round(100.0 * self.late_mix_ratio))

    def describe(self) -> str:
        late_desc = self.late.describe() if self.late is not None else "disabled"
        return (
            f"scheduled base=({self.base.describe()}) "
            f"late=({late_desc}) late_start_step:{self.late_start_step} "
            f"late_mix_ratio:{self.late_mix_ratio:.3f} last_stream:{self.last_stream}"
        )

    def next_batch(self, device: torch.device, step: int | None = None) -> tuple[Tensor, Tensor]:
        self.counter += 1
        if self._use_late(step):
            self.last_stream = "late"
            assert self.late is not None
            return self.late.next_batch(device, step=step)
        self.last_stream = "base"
        return self.base.next_batch(device, step=step)


class FullEntryGraphParquetStream:
    """Stream complete graphified rows for segmented long-entry training.

    This loader preserves row boundaries.  The ordinary graph LM stream is
    intentionally chunk-oriented for stable BPB training; this stream is the
    complementary objective for complete records such as UniProt/FoT/structure
    entries whose tokenized lengths are often thousands to tens of thousands of
    tokens.  Rows are graphified with the same TokenGT-style serialization as
    :class:`GraphParquetTokenStream`, then emitted as full token vectors capped
    by ``max_tokens``.
    """

    def __init__(
        self,
        pattern: str,
        tokenizer: Any,
        *,
        batch_size: int = 1,
        max_tokens: int = 24576,
    ) -> None:
        self.pattern = pattern
        self.files = GraphParquetTokenStream._expand_patterns(pattern)
        if not self.files:
            raise FileNotFoundError(f"No full-entry graph Parquet files found for pattern: {pattern}")
        self.tokenizer = tokenizer
        self.batch_size = max(1, int(batch_size))
        self.max_tokens = max(32, int(max_tokens))
        self.file_idx = 0
        self.row_idx = 0
        self.rows: list[tuple[str, str]] = []
        self.rows_graphified = 0
        self.policy_counts: dict[str, int] = {}
        self.truncated_rows = 0
        self.total_rows_emitted = 0
        self.parquet_file: Any = None
        self.batch_iter: Any = None
        self.active_columns: list[str] = []
        self.batch_size_rows = 128
        self.skipped_files: list[str] = []
        self._load_file()

    def refresh_files(self) -> None:
        refreshed = GraphParquetTokenStream._expand_patterns(self.pattern)
        if refreshed:
            current = self.files[self.file_idx] if self.files and self.file_idx < len(self.files) else None
            self.files = refreshed
            if current in self.files:
                self.file_idx = self.files.index(current)
            else:
                self.file_idx = min(self.file_idx, max(0, len(self.files) - 1))

    def _encode(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer.encode(text, out_type=int)
        except TypeError:
            ids = self.tokenizer.encode(text)
        return [int(token_id) for token_id in ids]

    def describe(self) -> str:
        policies = ",".join(f"{key}:{value}" for key, value in sorted(self.policy_counts.items()))
        return (
            f"files:{len(self.files)} current_rows:{len(self.rows)} graphified_rows:{self.rows_graphified} "
            f"emitted:{self.total_rows_emitted} truncated:{self.truncated_rows} "
            f"max_tokens:{self.max_tokens} skipped_files:{len(self.skipped_files)} policies:{policies or 'none'}"
        )

    def _load_file(self) -> None:
        import pyarrow.parquet as pq

        attempts = 0
        while attempts < len(self.files):
            path = self.files[self.file_idx]
            try:
                self.parquet_file = pq.ParquetFile(path)
                schema_names = set(self.parquet_file.schema_arrow.names)
            except Exception as exc:
                self.skipped_files.append(f"{path}:{exc}")
                self.file_idx = (self.file_idx + 1) % len(self.files)
                attempts += 1
                continue
            candidate_columns = tuple(dict.fromkeys(GraphParquetTokenStream.TEXT_COLUMNS + ("forest_json", "thought_forest_json")))
            columns = [name for name in candidate_columns if name in schema_names]
            if not columns:
                self.skipped_files.append(f"{path}:no_graphifiable_columns")
                self.file_idx = (self.file_idx + 1) % len(self.files)
                attempts += 1
                continue
            self.active_columns = columns
            self.batch_iter = self.parquet_file.iter_batches(batch_size=self.batch_size_rows, columns=columns)
            self.rows = []
            self.row_idx = 0
            self._load_next_batch()
            return
        raise ValueError(f"No readable full-entry graph Parquet shards with usable columns for pattern: {self.pattern}")

    def _load_next_batch(self) -> None:
        import pyarrow as pa

        if self.batch_iter is None:
            self._load_file()
            return
        path = self.files[self.file_idx]
        while True:
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                self._advance_file()
                return
            table = pa.Table.from_batches([batch])
            rows: list[tuple[str, str]] = []
            offset_base = self.rows_graphified
            for idx, row in enumerate(table.to_pylist()):
                text, policy = GraphParquetTokenStream._graphify_row(row, f"{path.name}:{offset_base + idx}")
                forest = row.get("forest_json") or row.get("thought_forest_json")
                if forest not in (None, ""):
                    text = (
                        text
                        + "\n<forest_of_thought>"
                        + GraphParquetTokenStream._payload_text(forest, max_chars=2400)
                        + "</forest_of_thought>"
                    )
                if text.strip():
                    rows.append((text, policy))
                    self.rows_graphified += 1
                    self.policy_counts[policy] = self.policy_counts.get(policy, 0) + 1
            if rows:
                self.rows = rows
                self.row_idx = 0
                return

    def _load_file_legacy_unused(self) -> None:
        import pyarrow.parquet as pq

        path = self.files[self.file_idx]
        schema_names = set(pq.read_schema(path).names)
        candidate_columns = tuple(dict.fromkeys(GraphParquetTokenStream.TEXT_COLUMNS + ("forest_json", "thought_forest_json")))
        columns = [name for name in candidate_columns if name in schema_names]
        if not columns:
            raise ValueError(f"Full-entry graph Parquet shard has no usable text/graph columns: {path}")
        table = pq.read_table(path, columns=columns)
        rows: list[tuple[str, str]] = []
        for idx in range(table.num_rows):
            row = {name: table[name][idx].as_py() for name in columns}
            text, policy = GraphParquetTokenStream._graphify_row(row, f"{path.name}:{idx}")
            forest = row.get("forest_json") or row.get("thought_forest_json")
            if forest not in (None, ""):
                text = (
                    text
                    + "\n<forest_of_thought>"
                    + GraphParquetTokenStream._payload_text(forest, max_chars=2400)
                    + "</forest_of_thought>"
                )
            if text.strip():
                rows.append((text, policy))
                self.rows_graphified += 1
                self.policy_counts[policy] = self.policy_counts.get(policy, 0) + 1
        if not rows:
            raise ValueError(f"Full-entry graph Parquet shard produced no rows: {path}")
        self.rows = rows
        self.row_idx = 0

    def _advance_file(self) -> None:
        self.refresh_files()
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self._load_file()

    def _next_row_text(self) -> tuple[str, str]:
        if self.row_idx >= len(self.rows):
            self._load_next_batch()
        row = self.rows[self.row_idx]
        self.row_idx += 1
        return row

    def next_entries(self, device: torch.device, step: int | None = None) -> dict[str, Any]:
        del step
        entries: list[Tensor] = []
        lengths: list[int] = []
        original_lengths: list[int] = []
        policies: list[str] = []
        truncated: list[bool] = []
        sep = self._encode("\n\n") or [0]
        for _ in range(self.batch_size):
            text, policy = self._next_row_text()
            ids = self._encode(text) + sep
            original_len = len(ids)
            was_truncated = original_len > self.max_tokens + 1
            if was_truncated:
                ids = ids[: self.max_tokens + 1]
                self.truncated_rows += 1
            if len(ids) < 2:
                ids = ids + sep
            tensor = torch.tensor(ids, dtype=torch.int64, device=device)
            entries.append(tensor)
            lengths.append(int(tensor.numel()))
            original_lengths.append(int(original_len))
            policies.append(policy)
            truncated.append(bool(was_truncated))
            self.total_rows_emitted += 1
        return {
            "entries": entries,
            "lengths": lengths,
            "original_lengths": original_lengths,
            "policies": policies,
            "truncated": truncated,
        }


class StructureCoordinateParquetStream:
    """Stream coordinate-bearing graph rows for structure-flow training.

    Unlike :class:`GraphParquetTokenStream`, this keeps row boundaries intact so
    each text/graph context remains aligned with its coordinate target.  Rows
    without explicit coordinate columns are skipped instead of replaced by
    synthetic coordinates.
    """

    TEXT_COLUMNS = tuple(dict.fromkeys(GraphParquetTokenStream.TEXT_COLUMNS + ("forest_json",)))
    COORD_COLUMNS = ("structure_coordinates", "ca_coordinates")

    def __init__(
        self,
        pattern: str,
        tokenizer: Any,
        seq_len: int,
        batch_size: int,
        *,
        max_atoms: int = 128,
    ) -> None:
        self.pattern = pattern
        self.files = self._expand_patterns(pattern)
        if not self.files:
            raise FileNotFoundError(f"No structure Parquet files found for pattern: {pattern}")
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.max_atoms = max(8, int(max_atoms))
        self.file_idx = 0
        self.row_idx = 0
        self.rows: list[tuple[str, list[list[float]], list[float] | None]] = []
        self.file_row_counts: dict[str, int] = {}
        self.coordinate_file_count = 0
        self.total_coordinate_candidate_rows = 0
        self._scan_file_counts()
        self.rows_loaded = 0
        self.rows_coordinate_bearing = 0
        self.rows_skipped = 0
        self._load_file()

    @staticmethod
    def _expand_patterns(pattern: str) -> list[Path]:
        pieces = [piece.strip() for piece in re.split(r"[,;]", pattern) if piece.strip()]
        files: list[Path] = []
        seen: set[str] = set()
        for piece in pieces or [pattern]:
            for item in sorted(glob.glob(piece)):
                path = Path(item)
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    files.append(path)
        return files

    def refresh_files(self) -> None:
        refreshed = self._expand_patterns(self.pattern)
        if refreshed:
            current = self.files[self.file_idx] if self.files and self.file_idx < len(self.files) else None
            self.files = refreshed
            self._scan_file_counts()
            if current in self.files:
                self.file_idx = self.files.index(current)
            else:
                self.file_idx = min(self.file_idx, max(0, len(self.files) - 1))

    def _scan_file_counts(self) -> None:
        """Count configured coordinate-bearing rows from Parquet metadata.

        This is intentionally metadata-only.  The actual loader still validates
        row-level coordinate tensors and skips malformed rows rather than using
        these counts as ground truth.
        """
        import pyarrow.parquet as pq

        self.file_row_counts = {}
        self.coordinate_file_count = 0
        self.total_coordinate_candidate_rows = 0
        for path in self.files:
            try:
                schema_names = set(pq.read_schema(path).names)
                if not any(name in schema_names for name in self.COORD_COLUMNS):
                    self.file_row_counts[str(path)] = 0
                    continue
                rows = int(pq.ParquetFile(path).metadata.num_rows)
            except Exception:
                self.file_row_counts[str(path)] = 0
                continue
            self.file_row_counts[str(path)] = rows
            self.coordinate_file_count += 1
            self.total_coordinate_candidate_rows += rows

    def _encode(self, text: str) -> list[int]:
        try:
            ids = self.tokenizer.encode(text, out_type=int)
        except TypeError:
            ids = self.tokenizer.encode(text)
        return [int(token_id) for token_id in ids]

    @staticmethod
    def _normalize_coords(value: Any, max_atoms: int) -> list[list[float]] | None:
        if value in (None, ""):
            return None
        coords: list[list[float]] = []
        try:
            iterable = value.tolist() if hasattr(value, "tolist") else value
        except Exception:
            iterable = value
        if not isinstance(iterable, list):
            return None
        for item in iterable[:max_atoms]:
            if item is None:
                continue
            try:
                xyz = item.tolist() if hasattr(item, "tolist") else item
            except Exception:
                xyz = item
            if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
                continue
            try:
                coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
            except Exception:
                continue
        return coords if len(coords) >= 8 else None

    @staticmethod
    def _normalize_plddt(value: Any, n: int) -> list[float] | None:
        if value in (None, ""):
            return None
        try:
            vals = value.tolist() if hasattr(value, "tolist") else value
        except Exception:
            vals = value
        if not isinstance(vals, list):
            return None
        out: list[float] = []
        for raw in vals[:n]:
            try:
                out.append(float(raw))
            except Exception:
                out.append(0.0)
        return out if out else None

    @staticmethod
    def _row_text(row: dict[str, Any], record_id: str) -> str:
        text, policy = GraphParquetTokenStream._graphify_row(row, record_id)
        forest = row.get("forest_json")
        if forest not in (None, ""):
            text = text + "\n<forest_of_thought>" + GraphParquetTokenStream._payload_text(forest, max_chars=1600) + "</forest_of_thought>"
        metadata = row.get("metadata_json")
        if metadata not in (None, ""):
            text = text + "\n<structure_metadata>" + GraphParquetTokenStream._payload_text(metadata, max_chars=1200) + "</structure_metadata>"
        return text

    def describe(self) -> str:
        current = self.files[self.file_idx] if self.files and self.file_idx < len(self.files) else None
        current_name = current.name if current is not None else "none"
        return (
            f"files:{len(self.files)} coord_files:{self.coordinate_file_count} "
            f"total_candidate_rows:{self.total_coordinate_candidate_rows} "
            f"current_file:{self.file_idx + 1}/{len(self.files)}:{current_name} "
            f"current_shard_rows:{len(self.rows)} "
            f"loaded_coordinate_rows_seen:{self.rows_coordinate_bearing} skipped:{self.rows_skipped} "
            f"max_atoms:{self.max_atoms}"
        )

    def _load_file(self) -> None:
        import pyarrow.parquet as pq

        path = self.files[self.file_idx]
        schema_names = set(pq.read_schema(path).names)
        coord_col = next((name for name in self.COORD_COLUMNS if name in schema_names), None)
        if coord_col is None:
            self.rows = []
            self.rows_skipped += 1
            return
        columns = [name for name in self.TEXT_COLUMNS if name in schema_names]
        for name in (coord_col, "plddt", "record_id"):
            if name in schema_names and name not in columns:
                columns.append(name)
        table = pq.read_table(path, columns=columns)
        rows: list[tuple[str, list[list[float]], list[float] | None]] = []
        for idx in range(table.num_rows):
            row = {name: table[name][idx].as_py() for name in columns}
            coords = self._normalize_coords(row.get(coord_col), self.max_atoms)
            if coords is None:
                self.rows_skipped += 1
                continue
            record_id = str(row.get("record_id") or f"{path.name}:{idx}")
            rows.append((self._row_text(row, record_id), coords, self._normalize_plddt(row.get("plddt"), len(coords))))
            self.rows_coordinate_bearing += 1
        if not rows:
            raise ValueError(f"Structure Parquet shard has no coordinate-bearing rows: {path}")
        self.rows = rows
        self.rows_loaded += len(rows)
        self.row_idx = 0

    def _advance_file(self) -> None:
        self.refresh_files()
        attempts = 0
        while attempts < len(self.files):
            self.file_idx = (self.file_idx + 1) % len(self.files)
            attempts += 1
            self._load_file()
            if self.rows:
                return
        raise ValueError("No coordinate-bearing structure rows available in configured files")

    def _next_row(self) -> tuple[str, list[list[float]], list[float] | None]:
        if self.row_idx >= len(self.rows):
            self._advance_file()
        row = self.rows[self.row_idx]
        self.row_idx += 1
        return row

    def next_batch(self, device: torch.device, step: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x_rows: list[Tensor] = []
        y_rows: list[Tensor] = []
        coord_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        plddt_rows: list[Tensor] = []
        sep = self._encode("\n\n") or [0]
        for _ in range(self.batch_size):
            text, coords, plddt = self._next_row()
            ids = self._encode(text)
            if len(ids) < 2:
                ids = ids + sep
            ids = (ids + sep * (self.seq_len + 1))[: self.seq_len + 1]
            if len(ids) < self.seq_len + 1:
                ids = ids + [0] * (self.seq_len + 1 - len(ids))
            token_tensor = torch.tensor(ids, dtype=torch.int64)
            x_rows.append(token_tensor[:-1])
            y_rows.append(token_tensor[1:])
            n = min(len(coords), self.max_atoms)
            coord_tensor = torch.zeros(self.max_atoms, 3, dtype=torch.float32)
            mask_tensor = torch.zeros(self.max_atoms, dtype=torch.bool)
            plddt_tensor = torch.zeros(self.max_atoms, dtype=torch.float32)
            if n:
                coord_tensor[:n] = torch.tensor(coords[:n], dtype=torch.float32)
                mask_tensor[:n] = True
                if plddt:
                    plddt_tensor[: min(n, len(plddt))] = torch.tensor(plddt[:n], dtype=torch.float32)
            coord_rows.append(coord_tensor)
            mask_rows.append(mask_tensor)
            plddt_rows.append(plddt_tensor)
        return (
            torch.stack(x_rows).to(device, non_blocking=True),
            torch.stack(y_rows).to(device, non_blocking=True),
            torch.stack(coord_rows).to(device, non_blocking=True),
            torch.stack(mask_rows).to(device, non_blocking=True),
            torch.stack(plddt_rows).to(device, non_blocking=True),
        )


class ToricGTSidecar(nn.Module):
    """Train graph, geometry, topology, category, and memory-retrieval heads."""

    LAGRANGIAN_FAMILIES = (
        "graphcg",
        "analogy",
        "tokengt_graph",
        "trajectory_memory",
        "toric_geometry",
        "toric_vector_bundle_1d_cone_ce",
        "toric_bgg",
        "koszul_persistence",
        "combinatorial_toric",
        "derived_signature",
    )

    def __init__(self, dim: int, args: Any):
        super().__init__()
        self.graphcg_basis = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.graphcg_basis)
        self.graph_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.edge_head = nn.Linear(dim, 1, bias=False)
        self.memory = TrajectoryRetrievalHead(
            dim,
            TrajectoryMemoryConfig(
                projection_dim=min(160, max(64, dim // 3)),
                teacher_temperature=0.20,
                retrieval_temperature=0.20,
                distill_weight=0.25,
                quality_weight=0.10,
                topology_weight=0.20,
                graphcg_weight=0.30,
                toric_weight=0.20,
                dag_weight=0.20,
                derived_weight=0.20,
                persistence_weight=0.20,
                persistence_max_points=48,
                persistence_landscape_layers=3,
                persistence_landscape_resolution=24,
                persistence_image_resolution=12,
                sheaf_gate_min=float(getattr(args, "memory_sheaf_gate_min", 0.15)),
                sheaf_gate_threshold=float(getattr(args, "memory_sheaf_gate_threshold", 0.12)),
                sheaf_gate_softness=float(getattr(args, "memory_sheaf_gate_softness", 0.18)),
                sheaf_ce_weight=float(getattr(args, "memory_sheaf_ce_weight", 1.0)),
            ),
        )
        self.derived_signature_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, 11),
        )
        self.derived_cfg = DerivedCategoryConfig(
            max_vertices=int(getattr(args, "derived_signature_max_vertices", 8)),
            max_edges=int(getattr(args, "derived_signature_max_edges", 64)),
        )
        self.graphcg_loss_weight = float(args.graphcg_loss_weight)
        self.analogy_loss_weight = float(args.analogy_loss_weight)
        self.tokengt_graph_loss_weight = float(args.tokengt_graph_loss_weight)
        self.trajectory_memory_loss_weight = float(args.trajectory_memory_loss_weight)
        self.toric_geometry_loss_weight = float(getattr(args, "toric_geometry_loss_weight", 0.0))
        self.toric_vector_bundle_loss_weight = float(getattr(args, "toric_vector_bundle_loss_weight", 0.0))
        self.toric_bgg_loss_weight = float(getattr(args, "toric_bgg_loss_weight", 0.0))
        self.koszul_persistence_loss_weight = float(getattr(args, "koszul_persistence_loss_weight", 0.0))
        self.combinatorial_toric_loss_weight = float(getattr(args, "combinatorial_toric_loss_weight", 0.0))
        self.derived_signature_loss_weight = float(getattr(args, "derived_signature_loss_weight", 0.0))
        self.compute_all_metrics = bool(int(getattr(args, "sidecar_compute_all_metrics", 1)))
        self.retrieval_conditioned_aux = bool(int(getattr(args, "retrieval_conditioned_aux", 1)))
        self.retrieval_gate_min = float(getattr(args, "retrieval_gate_min", 0.20))
        self.retrieval_gate_softness = max(float(getattr(args, "retrieval_gate_softness", 0.08)), 1e-6)
        self.retrieval_gate_center = float(getattr(args, "retrieval_gate_center", 0.20))
        self.uncertainty_weighting = bool(int(getattr(args, "sidecar_uncertainty_weighting", 1)))
        self.uncertainty_alpha = float(getattr(args, "sidecar_uncertainty_alpha", 0.35))
        self.uncertainty_center = float(getattr(args, "sidecar_uncertainty_center", 2.6))
        self.uncertainty_scale = max(float(getattr(args, "sidecar_uncertainty_scale", 1.0)), 1e-6)
        self.uncertainty_max = max(float(getattr(args, "sidecar_uncertainty_max", 2.0)), 0.0)
        self.graphcg_bpb_orthogonal_weight = float(getattr(args, "graphcg_bpb_orthogonal_weight", 0.02))
        self.graphcg_covariance_conflict_damping = float(getattr(args, "graphcg_covariance_conflict_damping", 0.50))
        self.advanced_lagrangian_controller = bool(int(getattr(args, "advanced_lagrangian_controller", 1)))
        self.lagrangian_dual_lr = float(getattr(args, "lagrangian_dual_lr", 0.025))
        self.lagrangian_decay = float(getattr(args, "lagrangian_decay", 0.985))
        self.lagrangian_min_multiplier = float(getattr(args, "lagrangian_min_multiplier", 0.25))
        self.lagrangian_max_multiplier = float(getattr(args, "lagrangian_max_multiplier", 2.25))
        self.lagrangian_bpb_ceiling = float(getattr(args, "lagrangian_bpb_ceiling", 3.05))
        self.lagrangian_bpb_softness = max(float(getattr(args, "lagrangian_bpb_softness", 0.35)), 1e-6)
        self.toric_fan_curriculum = bool(int(getattr(args, "toric_fan_curriculum", 1)))
        self.toric_fan_coarse_steps = int(getattr(args, "toric_fan_coarse_steps", 450))
        self.toric_fan_intermediate_steps = int(getattr(args, "toric_fan_intermediate_steps", 950))
        targets = torch.tensor(
            [
                float(getattr(args, "lagrangian_graphcg_target", 0.010)),
                float(getattr(args, "lagrangian_analogy_target", 0.045)),
                float(getattr(args, "lagrangian_tokengt_graph_target", 0.75)),
                float(getattr(args, "lagrangian_memory_target", 1.15)),
                float(getattr(args, "lagrangian_toric_target", 2.0)),
                float(getattr(args, "lagrangian_vector_bundle_target", 0.45)),
                float(getattr(args, "lagrangian_bgg_target", 0.06)),
                float(getattr(args, "lagrangian_koszul_target", 0.035)),
                float(getattr(args, "lagrangian_cca_target", 1.0)),
                float(getattr(args, "lagrangian_derived_signature_target", 0.025)),
            ],
            dtype=torch.float32,
        )
        self.register_buffer("lagrangian_dual", torch.zeros(len(self.LAGRANGIAN_FAMILIES), dtype=torch.float32))
        self.register_buffer("lagrangian_targets", targets)
        self.register_buffer("sidecar_forward_count", torch.zeros((), dtype=torch.long))

        self.toric_geometry = LowRankToricGeometryProbe(
            dim,
            ToricGeometryConfig(
                enabled=True,
                num_exponents=int(getattr(args, "toric_geometry_num_exponents", 16)),
                exponent_dim=int(getattr(args, "toric_geometry_exponent_dim", 4)),
                probe_rank=int(getattr(args, "toric_geometry_probe_rank", 12)),
                quant_bits=int(getattr(args, "toric_geometry_quant_bits", 6)),
                max_positions=int(getattr(args, "toric_geometry_max_positions", 128)),
                fan_weight=float(getattr(args, "toric_geometry_fan_weight", 1.0)),
                bend_weight=float(getattr(args, "toric_geometry_bend_weight", 0.3)),
                binom_weight=float(getattr(args, "toric_geometry_binom_weight", 0.3)),
                moment_weight=float(getattr(args, "toric_geometry_moment_weight", 0.5)),
                coxeter_weight=float(getattr(args, "toric_geometry_coxeter_weight", 0.2)),
                braid_weight=float(getattr(args, "toric_geometry_braid_weight", 0.1)),
                leaf_weight=float(getattr(args, "toric_geometry_leaf_weight", 0.25)),
                cas_toric_ideal_certificate_path=str(getattr(args, "cas_toric_ideal_certificate_path", "")),
            ),
        )
        self.vector_bundle = ToricVectorBundleProbe(
            dim,
            ToricVectorBundleConfig(
                rank=int(getattr(args, "toric_vector_bundle_rank", 8)),
                num_one_dimensional_cones=int(
                    getattr(args, "toric_vector_bundle_num_one_dimensional_cones", 8)
                ),
                num_cones=int(getattr(args, "toric_vector_bundle_num_cones", 8)),
                filtration_levels=int(getattr(args, "toric_vector_bundle_filtration_levels", 3)),
                max_positions=int(getattr(args, "toric_vector_bundle_max_positions", 96)),
            ),
        )
        self.toric_bgg = ToricBGGProbe(
            dim,
            ToricBGGConfig(
                num_standard_tokens=int(getattr(args, "toric_bgg_num_standard_tokens", 8)),
                probe_rank=int(getattr(args, "toric_bgg_probe_rank", 8)),
                signature_dim=int(getattr(args, "toric_bgg_signature_dim", 16)),
                max_positions=int(getattr(args, "toric_bgg_max_positions", 64)),
            ),
        )
        self.koszul_cfg = KoszulPersistenceConfig(
            max_points=int(getattr(args, "koszul_max_points", 16)),
            max_windows=int(getattr(args, "koszul_max_windows", 2)),
            window_size=int(getattr(args, "koszul_window_size", 32)),
            step_stride=int(getattr(args, "koszul_step_stride", 16)),
            num_parameters=int(getattr(args, "koszul_num_parameters", 3)),
            temperature=float(getattr(args, "koszul_temperature", 0.12)),
            chart_exponents=int(getattr(args, "koszul_chart_exponents", 10)),
        )
        self.combinatorial_cfg = CombinatorialToricConfig(
            max_points=int(getattr(args, "combinatorial_toric_max_points", 16)),
            max_windows=int(getattr(args, "combinatorial_toric_max_windows", 2)),
            window_size=int(getattr(args, "combinatorial_toric_window_size", 32)),
            step_stride=int(getattr(args, "combinatorial_toric_step_stride", 16)),
            num_chambers=int(getattr(args, "combinatorial_toric_num_chambers", 8)),
            exponent_dim=int(getattr(args, "combinatorial_toric_exponent_dim", 4)),
        )

    def _fan_curriculum_stage(self, step: int | None) -> tuple[int, str]:
        if not self.toric_fan_curriculum:
            return 2, "full"
        if step is None:
            step = int(self.sidecar_forward_count.detach().cpu().item())
        if step < self.toric_fan_coarse_steps:
            return 0, "coarse"
        if step < self.toric_fan_intermediate_steps:
            return 1, "intermediate"
        return 2, "full"

    def _lagrangian_multiplier(
        self,
        family: str,
        loss: Tensor,
        *,
        safety_gate: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = loss.device
        if family not in self.LAGRANGIAN_FAMILIES:
            return loss.new_tensor(1.0), {}
        idx = self.LAGRANGIAN_FAMILIES.index(family)
        target = self.lagrangian_targets[idx].to(device=device, dtype=loss.dtype)
        if self.advanced_lagrangian_controller:
            with torch.no_grad():
                observed = loss.detach().float().clamp_min(0.0)
                violation = (observed - target.detach().float()).clamp_min(0.0)
                update = float(self.lagrangian_dual_lr) * violation * safety_gate.detach().float().clamp(0.0, 1.0)
                self.lagrangian_dual[idx].mul_(float(self.lagrangian_decay)).add_(
                    update.to(device=self.lagrangian_dual.device)
                )
                self.lagrangian_dual[idx].clamp_(0.0, max(0.0, self.lagrangian_max_multiplier - 1.0))
            multiplier = (1.0 + self.lagrangian_dual[idx].to(device=device, dtype=loss.dtype)) * (
                0.25 + 0.75 * safety_gate.to(dtype=loss.dtype).clamp(0.0, 1.0)
            )
            multiplier = multiplier.clamp(float(self.lagrangian_min_multiplier), float(self.lagrangian_max_multiplier))
        else:
            multiplier = loss.new_tensor(1.0)
        prefix = f"lagrangian_{family}"
        return multiplier, {
            f"{prefix}_target": target.detach(),
            f"{prefix}_dual": self.lagrangian_dual[idx].to(device=device, dtype=loss.dtype).detach(),
            f"{prefix}_multiplier": multiplier.detach(),
            f"{prefix}_violation": (loss.detach().float() - target.detach().float()).clamp_min(0.0),
        }

    @staticmethod
    def _byte_class_ids(tokens: Tensor) -> Tensor:
        cls = torch.full_like(tokens, 8)
        cls = torch.where((tokens >= 4) & (tokens < 128), torch.ones_like(cls), cls)
        cls = torch.where((tokens >= 128) & (tokens < 256), torch.full_like(cls, 2), cls)
        return torch.where(tokens >= 256, torch.full_like(cls, 3), cls)

    def _graphcg_losses(self, hidden: Tensor, per_token_nll: Tensor | None = None) -> dict[str, Tensor]:
        flat_hidden = hidden.float().reshape(-1, hidden.shape[-1])
        flat_nll = per_token_nll.float().reshape(-1) if per_token_nll is not None else None
        h = F.normalize(flat_hidden, dim=-1)
        nll_sample = flat_nll
        if h.shape[0] > 256:
            idx = torch.linspace(0, h.shape[0] - 1, steps=256, device=h.device).long()
            h = h.index_select(0, idx)
            if nll_sample is not None:
                nll_sample = nll_sample.index_select(0, idx)
        basis = F.normalize(self.graphcg_basis.float(), dim=-1)
        coords = h @ basis.T
        eye = torch.eye(basis.shape[0], device=h.device, dtype=basis.dtype)
        gram = basis @ basis.T
        offdiag = max(1, basis.shape[0] * (basis.shape[0] - 1))
        orthogonal = (gram - eye).pow(2).sum() / offdiag
        centered = coords - coords.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        diag = cov.diagonal().clamp_min(1e-6)
        corr = cov / torch.sqrt(diag[:, None] * diag[None, :])
        covariance = (corr - eye).pow(2).sum() / offdiag
        axis_probs = torch.softmax(coords.abs() / 0.20, dim=-1)
        entropy = -(axis_probs * axis_probs.clamp_min(1e-8).log()).sum(dim=-1).mean() / math.log(max(2, basis.shape[0]))
        bpb_alignment = h.new_zeros(())
        bpb_conflict = h.new_zeros(())
        covariance_scale = h.new_tensor(1.0)
        if nll_sample is not None and nll_sample.numel() == h.shape[0] and h.shape[0] > 1:
            centered_nll = nll_sample.detach() - nll_sample.detach().mean()
            weights = torch.softmax(centered_nll / 0.50, dim=0).to(h.dtype)
            nll_direction = F.normalize((weights[:, None] * h.detach()).sum(dim=0), dim=0)
            bpb_alignment = (basis @ nll_direction).pow(2).mean()
            bpb_conflict = bpb_alignment.detach().clamp(0.0, 1.0)
            covariance_scale = 1.0 / (1.0 + float(self.graphcg_covariance_conflict_damping) * bpb_conflict)
        loss = (
            orthogonal
            + 0.05 * covariance_scale * covariance
            + 0.01 * entropy
            + float(self.graphcg_bpb_orthogonal_weight) * bpb_alignment
        )
        return {
            "graphcg_loss": loss,
            "graphcg_orthogonal_loss": orthogonal.detach(),
            "graphcg_covariance_loss": covariance.detach(),
            "graphcg_covariance_effective_scale": covariance_scale.detach(),
            "graphcg_axis_entropy": entropy.detach(),
            "graphcg_bpb_alignment_loss": bpb_alignment.detach(),
            "graphcg_bpb_conflict": bpb_conflict.detach(),
            "graphcg_chart_dim": hidden.new_tensor(float(basis.shape[0])).detach(),
        }

    def _analogy_losses(self, hidden: Tensor, targets: Tensor) -> dict[str, Tensor]:
        zero = hidden.float().sum() * 0.0
        if hidden.shape[1] < 3:
            return {"analogy_lattice_loss": zero, "analogy_relation_groups": zero.detach()}
        relation = F.normalize((hidden[:, 1:, :].float() - hidden[:, :-1, :].float()).reshape(-1, hidden.shape[-1]), dim=-1)
        classes = self._byte_class_ids(targets)
        keys = (classes[:, :-1] * 8 + classes[:, 1:]).reshape(-1)
        if relation.shape[0] > 512:
            idx = torch.linspace(0, relation.shape[0] - 1, steps=512, device=relation.device).long()
            relation = relation.index_select(0, idx)
            keys = keys.index_select(0, idx)
        unique, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
        repeated = counts[inverse] > 1
        if bool(repeated.any()):
            sums = relation.new_zeros(unique.shape[0], relation.shape[-1])
            sums.index_add_(0, inverse[repeated], relation[repeated])
            means = F.normalize(sums / counts.clamp_min(1).to(relation.dtype).unsqueeze(-1), dim=-1)
            functor = (relation[repeated] - means[inverse[repeated]]).pow(2).mean()
            groups = (counts > 1).to(relation.dtype).sum()
        else:
            functor = zero
            groups = zero
        basis = F.normalize(self.graphcg_basis.float(), dim=-1)
        coords = relation @ basis.T
        probs = torch.softmax(coords.abs() / 0.20, dim=-1)
        reconstructed = F.normalize((probs * coords.sign()) @ basis, dim=-1)
        basis_loss = (1.0 - (relation * reconstructed).sum(dim=-1)).mean()
        margin = (coords.abs().topk(2, dim=-1).values.diff(dim=-1).abs().mean() if coords.shape[-1] > 1 else coords.abs().mean())
        return {
            "analogy_lattice_loss": functor + 0.5 * basis_loss,
            "analogy_functor_loss": functor.detach(),
            "analogy_basis_loss": basis_loss.detach(),
            "analogy_lattice_margin": margin.detach(),
            "analogy_relation_groups": groups.detach(),
        }

    def _tokengt_graph_losses(self, hidden: Tensor, targets: Tensor) -> dict[str, Tensor]:
        h = self.graph_proj(hidden).float()
        bsz, seqlen, _ = h.shape
        n = min(seqlen, 128)
        if n < 4:
            zero = h.sum() * 0.0
            return {"tokengt_graph_loss": zero, "tokengt_graph_edge_density": zero.detach()}
        idx = torch.linspace(0, seqlen - 1, steps=n, device=h.device).round().long()
        nodes = F.normalize(h.index_select(1, idx), dim=-1)
        tgt = targets.index_select(1, idx)
        pos = idx.float()
        dist = (pos[:, None] - pos[None, :]).abs()
        eye = torch.eye(n, device=h.device, dtype=torch.bool)
        causal = (dist <= 2.0) & (pos[None, :] <= pos[:, None]) & ~eye
        same_class = self._byte_class_ids(tgt)[:, :, None] == self._byte_class_ids(tgt)[:, None, :]
        logits = torch.matmul(nodes, nodes.transpose(1, 2)) / 0.25
        mask = (~eye).to(logits.dtype)[None, :, :]
        edge_target = causal.unsqueeze(0).expand(bsz, -1, -1).to(logits.dtype)
        byte_target = same_class.to(logits.dtype) * mask
        edge_bce = F.binary_cross_entropy_with_logits(logits, edge_target, weight=mask, reduction="sum") / mask.sum().clamp_min(1.0) / bsz
        byte_bce = F.binary_cross_entropy_with_logits(logits, byte_target, weight=mask, reduction="sum") / mask.sum().clamp_min(1.0) / bsz
        projected_edges = torch.sigmoid(self.edge_head(nodes[:, 1:, :] - nodes[:, :-1, :])).mean()
        loss = edge_bce + 0.25 * byte_bce + 0.05 * (1.0 - projected_edges).pow(2)
        return {
            "tokengt_graph_loss": loss,
            "tokengt_graph_edge_bce": edge_bce.detach(),
            "tokengt_graph_byte_class_loss": byte_bce.detach(),
            "tokengt_graph_edge_density": edge_target.mean().detach(),
            "tokengt_graph_causal_edge_fraction": edge_target.mean().detach(),
        }

    def _derived_signature_losses(self, hidden: Tensor) -> dict[str, Tensor]:
        max_vertices = min(hidden.shape[1], int(self.derived_cfg.max_vertices))
        if max_vertices < 2:
            zero = hidden.float().sum() * 0.0
            return {
                "derived_signature_loss": zero,
                "derived_signature_cosine": zero.detach(),
                "derived_signature_target_norm": zero.detach(),
            }
        window = hidden[:, :max_vertices, :].float()
        target = derived_category_feature_summary(window, config=self.derived_cfg).detach().float()
        pred = self.derived_signature_head(window.mean(dim=1).to(hidden.dtype)).float()
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        mse = F.mse_loss(pred, target)
        cosine = (pred_norm * target_norm).sum(dim=-1).mean()
        return {
            "derived_signature_loss": mse,
            "derived_signature_cosine": cosine.detach(),
            "derived_signature_target_norm": target.norm(dim=-1).mean().detach(),
        }

    def forward(
        self,
        hidden: Tensor,
        targets: Tensor,
        positions: Tensor,
        per_token_nll: Tensor,
        *,
        step: int | None = None,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            self.sidecar_forward_count.add_(1)
        out: dict[str, Tensor] = {}
        out.update(self._graphcg_losses(hidden, per_token_nll))
        out.update(self._analogy_losses(hidden, targets))
        out.update(self._tokengt_graph_losses(hidden, targets))
        out.update(self.memory(hidden, positions, per_token_nll, graphcg_basis=self.graphcg_basis))
        out.update(self._derived_signature_losses(hidden))
        run_all = bool(self.compute_all_metrics)
        if run_all or self.toric_geometry_loss_weight != 0.0:
            out.update(self.toric_geometry(hidden, positions, targets))
        if run_all or self.toric_vector_bundle_loss_weight != 0.0:
            out.update(self.vector_bundle(hidden, positions, targets))
        if run_all or self.toric_bgg_loss_weight != 0.0:
            out.update(self.toric_bgg(hidden, positions, targets))
        if run_all or self.koszul_persistence_loss_weight != 0.0:
            out.update(koszul_persistence_loss(hidden, positions, config=self.koszul_cfg))
        if run_all or self.combinatorial_toric_loss_weight != 0.0:
            out.update(combinatorial_toric_cca_topology_loss(hidden, positions, config=self.combinatorial_cfg))
        nll_mean = per_token_nll.detach().float().mean()
        nll_p90 = per_token_nll.detach().float().flatten().quantile(0.90) if per_token_nll.numel() else nll_mean
        bpb_safety_gate = torch.sigmoid(
            (hidden.new_tensor(float(self.lagrangian_bpb_ceiling)) - nll_mean.to(device=hidden.device))
            / float(self.lagrangian_bpb_softness)
        )
        lagrangian_metrics: dict[str, Tensor] = {
            "lagrangian_bpb_safety_gate": bpb_safety_gate.detach(),
            "lagrangian_bpb_ceiling": hidden.new_tensor(float(self.lagrangian_bpb_ceiling)).detach(),
            "lagrangian_controller_active": hidden.new_tensor(float(self.advanced_lagrangian_controller)).detach(),
        }
        stage, stage_name = self._fan_curriculum_stage(step)
        toric_loss_key = (
            "toric_geometry_loss_coarse"
            if stage == 0
            else "toric_geometry_loss_intermediate"
            if stage == 1
            else "toric_geometry_loss_full"
        )
        zero = hidden.new_zeros(())
        uncertainty_raw = ((nll_p90 - self.uncertainty_center) / self.uncertainty_scale).clamp(0.0, self.uncertainty_max)
        uncertainty_weight = 1.0 + float(self.uncertainty_alpha) * uncertainty_raw if self.uncertainty_weighting else hidden.new_tensor(1.0)
        memory_gap = out.get("trajectory_memory_score_gap", zero.detach()).detach()
        memory_recall = out.get("trajectory_memory_recall1", zero.detach()).detach()
        memory_signal = 0.5 * torch.sigmoid((memory_gap - self.retrieval_gate_center) / self.retrieval_gate_softness) + 0.5 * memory_recall.clamp(0.0, 1.0)
        retrieval_gate = self.retrieval_gate_min + (1.0 - self.retrieval_gate_min) * memory_signal
        if not self.retrieval_conditioned_aux:
            retrieval_gate = hidden.new_tensor(1.0)
        total = zero
        graphcg_mult, metrics = self._lagrangian_multiplier("graphcg", out["graphcg_loss"], safety_gate=bpb_safety_gate)
        lagrangian_metrics.update(metrics)
        total = total + self.graphcg_loss_weight * graphcg_mult * out["graphcg_loss"]
        analogy_mult, metrics = self._lagrangian_multiplier(
            "analogy", out["analogy_lattice_loss"], safety_gate=bpb_safety_gate
        )
        lagrangian_metrics.update(metrics)
        total = total + self.analogy_loss_weight * analogy_mult * retrieval_gate * out["analogy_lattice_loss"]
        tokengt_mult, metrics = self._lagrangian_multiplier(
            "tokengt_graph", out["tokengt_graph_loss"], safety_gate=bpb_safety_gate
        )
        lagrangian_metrics.update(metrics)
        total = total + self.tokengt_graph_loss_weight * tokengt_mult * out["tokengt_graph_loss"]
        memory_mult, metrics = self._lagrangian_multiplier(
            "trajectory_memory", out["trajectory_memory_loss"], safety_gate=bpb_safety_gate
        )
        lagrangian_metrics.update(metrics)
        total = total + self.trajectory_memory_loss_weight * memory_mult * retrieval_gate * out["trajectory_memory_loss"]
        toric_loss = out.get(toric_loss_key, out.get("toric_geometry_loss", total.new_zeros(())))
        toric_mult, metrics = self._lagrangian_multiplier("toric_geometry", toric_loss, safety_gate=bpb_safety_gate)
        lagrangian_metrics.update(metrics)
        total = total + self.toric_geometry_loss_weight * toric_mult * uncertainty_weight * toric_loss
        vector_loss = out.get("toric_vector_bundle_1d_cone_ce_loss", total.new_zeros(()))
        vector_mult, metrics = self._lagrangian_multiplier(
            "toric_vector_bundle_1d_cone_ce", vector_loss, safety_gate=bpb_safety_gate
        )
        lagrangian_metrics.update(metrics)
        total = total + self.toric_vector_bundle_loss_weight * vector_mult * uncertainty_weight * vector_loss
        bgg_loss = out.get("toric_bgg_loss", total.new_zeros(()))
        bgg_mult, metrics = self._lagrangian_multiplier("toric_bgg", bgg_loss, safety_gate=bpb_safety_gate)
        lagrangian_metrics.update(metrics)
        total = total + self.toric_bgg_loss_weight * bgg_mult * uncertainty_weight * bgg_loss
        koszul_loss = out.get("koszul_persistence_loss", total.new_zeros(()))
        koszul_mult, metrics = self._lagrangian_multiplier("koszul_persistence", koszul_loss, safety_gate=bpb_safety_gate)
        lagrangian_metrics.update(metrics)
        total = total + self.koszul_persistence_loss_weight * koszul_mult * uncertainty_weight * koszul_loss
        cca_loss = out.get("toric_cca_topology_loss", total.new_zeros(()))
        cca_mult, metrics = self._lagrangian_multiplier("combinatorial_toric", cca_loss, safety_gate=bpb_safety_gate)
        lagrangian_metrics.update(metrics)
        total = total + self.combinatorial_toric_loss_weight * cca_mult * uncertainty_weight * cca_loss
        derived_loss = out.get("derived_signature_loss", total.new_zeros(()))
        derived_mult, metrics = self._lagrangian_multiplier(
            "derived_signature", derived_loss, safety_gate=bpb_safety_gate
        )
        lagrangian_metrics.update(metrics)
        total = total + self.derived_signature_loss_weight * derived_mult * uncertainty_weight * derived_loss
        out["sidecar_nll_mean"] = nll_mean.detach()
        out["sidecar_nll_p90"] = nll_p90.detach()
        out["sidecar_uncertainty_weight"] = uncertainty_weight.detach()
        out["sidecar_retrieval_gate"] = retrieval_gate.detach()
        out["sidecar_analogy_effective_weight"] = (hidden.new_tensor(self.analogy_loss_weight) * retrieval_gate).detach()
        out["sidecar_memory_effective_weight"] = (hidden.new_tensor(self.trajectory_memory_loss_weight) * retrieval_gate).detach()
        out["sidecar_advanced_effective_multiplier"] = uncertainty_weight.detach()
        out["toric_fan_curriculum_stage"] = total.new_tensor(float(stage)).detach()
        out["toric_fan_curriculum_active"] = total.new_tensor(float(self.toric_fan_curriculum)).detach()
        out["toric_geometry_curriculum_loss"] = toric_loss.detach()
        out["derived_signature_active"] = total.new_tensor(float(self.derived_signature_loss_weight != 0.0)).detach()
        out.update(lagrangian_metrics)
        out["toric_geometry_active"] = total.new_tensor(float(self.toric_geometry_loss_weight != 0.0))
        out["toric_vector_bundle_1d_cone_ce_active"] = total.new_tensor(
            float(self.toric_vector_bundle_loss_weight != 0.0)
        )
        out["toric_bgg_active"] = total.new_tensor(float(self.toric_bgg_loss_weight != 0.0))
        out["koszul_persistence_active"] = total.new_tensor(float(self.koszul_persistence_loss_weight != 0.0))
        out["combinatorial_toric_active"] = total.new_tensor(float(self.combinatorial_toric_loss_weight != 0.0))
        out["sidecar_compute_all_metrics"] = total.new_tensor(float(self.compute_all_metrics))
        out["toricgt_sidecar_loss"] = total
        return out
