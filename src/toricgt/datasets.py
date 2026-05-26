"""Dataset manifests, normalization, and deterministic split helpers."""

from __future__ import annotations

import hashlib
import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import DataConfig
from .hebrew import hebrew_niqqud_stats, prefer_pointed_variant


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: str
    splits: tuple[str, ...]
    loader: str
    task_family: str
    role: str
    license: str
    language: str
    url: str
    expected_rows: int | None = None
    file_name: str | None = None
    raw_url: str | None = None
    forced_split: str | None = None

    @property
    def key(self) -> str:
        return f"{self.name}::{self.config}"


MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="AI-MO/NuminaMath-CoT",
        config="default",
        splits=("train", "test"),
        loader="datasets",
        task_family="cot_math",
        role="large math chain-of-thought",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/AI-MO/NuminaMath-CoT",
        expected_rows=859_594,
    ),
    DatasetSpec(
        name="open-r1/OpenR1-Math-220k",
        config="all",
        splits=("train",),
        loader="datasets",
        task_family="cot_math",
        role="multi-trace math reasoning",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/open-r1/OpenR1-Math-220k",
        expected_rows=225_129,
    ),
    DatasetSpec(
        name="open-r1/codeforces-cots",
        config="solutions_w_editorials_py_decontaminated",
        splits=("train",),
        loader="datasets",
        task_family="algorithmic_code",
        role="decontaminated Codeforces Python CoT with editorials",
        license="cc-by-4.0",
        language="en",
        url="https://hf.co/datasets/open-r1/codeforces-cots",
        expected_rows=9_796,
    ),
    DatasetSpec(
        name="openai/gsm8k",
        config="main",
        splits=("train", "test"),
        loader="datasets",
        task_family="openai_math",
        role="OpenAI grade-school math",
        license="mit",
        language="en",
        url="https://hf.co/datasets/openai/gsm8k",
        expected_rows=8_792,
    ),
    DatasetSpec(
        name="openai/frontierscience",
        config="default",
        splits=("test",),
        loader="datasets",
        task_family="openai_frontierscience_eval",
        role="FrontierScience expert-level scientific benchmark; evaluation-only, never training",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/openai/frontierscience",
        expected_rows=160,
        forced_split="test",
    ),
    DatasetSpec(
        name="openai/healthbench",
        config="default",
        splits=("test",),
        loader="datasets",
        task_family="openai_healthbench_eval",
        role="HealthBench rubric-based health evaluation; evaluation-only by default",
        license="mit",
        language="multilingual",
        url="https://hf.co/datasets/openai/healthbench",
        expected_rows=5_000,
        forced_split="test",
    ),
    DatasetSpec(
        name="openai/healthbench-professional",
        config="default",
        splits=("test",),
        loader="datasets",
        task_family="openai_healthbench_professional_eval",
        role="HealthBench Professional physician-response rubric evaluation; evaluation-only by default",
        license="mit",
        language="multilingual",
        url="https://hf.co/datasets/openai/healthbench-professional",
        expected_rows=525,
        forced_split="test",
    ),
    DatasetSpec(
        name="openai/graphwalks",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="openai_graphwalks",
        role="OpenAI GraphWalks multi-hop directed graph reasoning benchmark and augmentation source",
        license="mit",
        language="en",
        url="https://hf.co/datasets/openai/graphwalks",
        expected_rows=1_150,
    ),
    *(
        DatasetSpec(
            name="EleutherAI/hendrycks_math",
            config=config,
            splits=("train", "test"),
            loader="datasets",
            task_family="competition_math",
            role=f"MATH competition subject: {config}",
            license="mit",
            language="en",
            url="https://hf.co/datasets/EleutherAI/hendrycks_math",
        )
        for config in MATH_CONFIGS
    ),
    DatasetSpec(
        name="HuggingFaceH4/MATH-500",
        config="default",
        splits=("test",),
        loader="datasets",
        task_family="competition_math",
        role="OpenAI verifier subset benchmark",
        license="see dataset card",
        language="en",
        url="https://hf.co/datasets/HuggingFaceH4/MATH-500",
        expected_rows=500,
    ),
    DatasetSpec(
        name="terrycraddock/Tree_Of_Thoughts_BASE_24k",
        config="default",
        splits=("train",),
        loader="hf_json",
        task_family="tot",
        role="explicit tree-of-thought QA",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/terrycraddock/Tree_Of_Thoughts_BASE_24k",
        expected_rows=24_731,
        file_name="Tree_Of_Thoughs_qa_base_24k.json",
    ),
    DatasetSpec(
        name="gss1147/Got_Math_500K",
        config="default",
        splits=("train",),
        loader="hf_jsonl",
        task_family="got_math",
        role="graph-of-thought / CoT math mixture",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/gss1147/Got_Math_500K",
        expected_rows=500_000,
        file_name="Got_Math_500K.jsonl",
    ),
    DatasetSpec(
        name="unimorph/universal_morphologies",
        config="heb",
        splits=("train",),
        loader="raw_tsv_url",
        task_family="hebrew_morphology",
        role="Hebrew UniMorph lemma-inflection morphology",
        license="cc-by-sa-3.0",
        language="he",
        url="https://hf.co/datasets/unimorph/universal_morphologies",
        raw_url="https://raw.githubusercontent.com/unimorph/heb/master/heb",
    ),
    DatasetSpec(
        name="Sefaria/Rabbinic-Hebrew-English-Pairs",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="rabbinic_parallel",
        role="Sefaria rabbinic Hebrew/Aramaic-English parallel corpus",
        license="cc-by-4.0",
        language="he,arc,en",
        url="https://hf.co/datasets/Sefaria/Rabbinic-Hebrew-English-Pairs",
        expected_rows=3_708,
    ),
    DatasetSpec(
        name="Sefaria/hebrew_library",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="jewish_hebrew_text",
        role="Sefaria Hebrew Jewish text library",
        license="gpl-3.0",
        language="he",
        url="https://hf.co/datasets/Sefaria/hebrew_library",
        expected_rows=3_549_020,
    ),
    DatasetSpec(
        name="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        config="stage1",
        splits=("train",),
        loader="datasets",
        task_family="frontier_gptoss_reasoning",
        role="gpt-oss-120b stage-1 reasoning traces across math, code, science, and instruction following",
        license="cc-by-4.0",
        language="en",
        url="https://hf.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        expected_rows=105_000,
    ),
    DatasetSpec(
        name="Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        config="stage2",
        splits=("train",),
        loader="datasets",
        task_family="frontier_gptoss_reasoning",
        role="gpt-oss-120b stage-2 reasoning traces across math, code, science, and instruction following",
        license="cc-by-4.0",
        language="en",
        url="https://hf.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
        expected_rows=330_000,
    ),
    DatasetSpec(
        name="Jackrong/GPT-OSS-120B-Distilled-Reasoning-math",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="frontier_gptoss_math",
        role="explicit gpt-oss-120b native reasoning math distillation",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/Jackrong/GPT-OSS-120B-Distilled-Reasoning-math",
    ),
    DatasetSpec(
        name="reasoning-degeneration-dev/algorithmic-sft-training-data-v1",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="procedural_algorithmic_reasoning",
        role="deterministic step-by-step traces for countdown, formal logic, arithmetic, morphology, and cellular automata",
        license="mit",
        language="en",
        url="https://hf.co/datasets/reasoning-degeneration-dev/algorithmic-sft-training-data-v1",
        expected_rows=63_000,
    ),
    DatasetSpec(
        name="lamm-mit/graph-reasoning-messages-11K",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="graph_reasoning_messages",
        role="graph-native structured reasoning conversations",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/lamm-mit/graph-reasoning-messages-11K",
        expected_rows=11_000,
    ),
    DatasetSpec(
        name="sequelbox/DAG-Reasoning-DeepSeek-R1-0528",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="dag_reasoning",
        role="directed-acyclic-graph reasoning traces distilled from DeepSeek-R1-0528",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/sequelbox/DAG-Reasoning-DeepSeek-R1-0528",
    ),
    DatasetSpec(
        name="Gryphe/Opus-4.6-Reasoning-24k",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="frontier_opus_reasoning",
        role="Apache-licensed aggregate of Opus 4.6 reasoning traces for math, code, logic, and science",
        license="apache-2.0",
        language="en",
        url="https://hf.co/datasets/Gryphe/Opus-4.6-Reasoning-24k",
        expected_rows=24_000,
    ),
    DatasetSpec(
        name="nvidia/Nemotron-RL-ReasoningGym-v1",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="nvidia_nemotron_reasoninggym",
        role="Nemotron RL procedural reasoning tasks across algebra, computation, cognition, geometry, graph theory, logic, and games",
        license="cc-by-4.0",
        language="en",
        url="https://hf.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1",
        expected_rows=15_000,
    ),
    DatasetSpec(
        name="nvidia/Nemotron-Content-Safety-Reasoning-Dataset",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="nvidia_nemotron_safety_reasoning",
        role="Nemotron content-safety reasoning traces with explicit label justifications",
        license="cc-by-4.0",
        language="en",
        url="https://hf.co/datasets/nvidia/Nemotron-Content-Safety-Reasoning-Dataset",
    ),
    DatasetSpec(
        name="nvidia/PhysicalAI-Traffic-Anomaly-Reasoning",
        config="default",
        splits=("train",),
        loader="datasets",
        task_family="nvidia_physicalai_traffic_reasoning",
        role="PhysicalAI traffic anomaly video QA, temporal reasoning, and chain-of-thought annotations",
        license="cc-by-4.0",
        language="en",
        url="https://hf.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning",
        expected_rows=44_040,
    ),
)


DATASET_MANIFEST = [asdict(spec) for spec in DATASET_SPECS]

_SPACE_RE = re.compile(r"\s+")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def safe_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return safe.strip("_") or "dataset"


def canonical_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_RE.sub(" ", text)
    return text.strip().lower()


def compact_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def first_string(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def messages_to_text(value: Any, role: str | None = None) -> str:
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        value = value["messages"]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return stripped
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_role = str(item.get("role", "")).strip().lower()
        content = item.get("content")
        if role is not None and item_role != role:
            continue
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n\n".join(parts)


def rubric_items_to_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion") or item.get("criterion_text") or "").strip()
        points = item.get("points")
        if criterion:
            parts.append(f"rubric_{idx}: {criterion} ({points} pts)")
    return "\n".join(parts)


def extract_answer_from_solution(solution: str) -> str:
    match = _BOXED_RE.search(solution or "")
    if match:
        return match.group(1).strip()
    marker = "answer is"
    low = solution.lower()
    idx = low.rfind(marker)
    if idx >= 0:
        return solution[idx + len(marker) :].strip().strip(".: ")
    return ""


def split_reasoning_steps(text: str, max_steps: int = 32) -> list[str]:
    if not text:
        return []
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        lines = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=(?:\d+\.|[A-Z]))", text) if part.strip()]
    return lines[:max_steps]


def make_graph_json(question: str, reasoning: str, answer: str, task_family: str) -> str:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    nodes.append({"id": "problem", "type": "problem", "text": question[:4000]})
    previous = "problem"
    for idx, step in enumerate(split_reasoning_steps(reasoning)):
        node_id = f"step_{idx:03d}"
        nodes.append({"id": node_id, "type": "reasoning_step", "text": step[:4000]})
        edges.append({"source": previous, "target": node_id, "type": "depends_on"})
        previous = node_id
    nodes.append({"id": "answer", "type": "answer", "text": answer[:4000]})
    edges.append({"source": previous, "target": "answer", "type": "supports_answer"})
    return compact_json({"task_family": task_family, "nodes": nodes, "edges": edges})


def make_graphwalks_graph_json(prompt: str, answer_nodes: Any, problem_type: str) -> str:
    edge_matches = re.findall(r"^([A-Za-z0-9_.:-]+)\s*->\s*([A-Za-z0-9_.:-]+)\s*$", prompt, flags=re.MULTILINE)
    operation_match = re.search(r"Operation:\s*(.*?)\s*(?:Final Answer:|$)", prompt, flags=re.DOTALL)
    operation = operation_match.group(1).strip() if operation_match else problem_type
    answer_list = [str(item) for item in answer_nodes] if isinstance(answer_nodes, list) else []
    node_names = sorted({node for edge in edge_matches for node in edge} | set(answer_list))
    nodes: list[dict[str, Any]] = [
        {"id": f"node_{idx:04d}", "type": "graph_node", "text": name}
        for idx, name in enumerate(node_names)
    ]
    id_by_name = {name: f"node_{idx:04d}" for idx, name in enumerate(node_names)}
    edges: list[dict[str, Any]] = []
    for idx, (src, dst) in enumerate(edge_matches):
        edges.append(
            {
                "source": id_by_name[src],
                "target": id_by_name[dst],
                "type": "directed_edge",
                "edge_index": idx,
            }
        )
    nodes.append({"id": "operation", "type": "operation", "text": operation[:4000]})
    for answer_idx, name in enumerate(answer_list):
        node_id = id_by_name.get(name)
        if node_id is not None:
            edges.append({"source": "operation", "target": node_id, "type": "returns_answer_node", "rank": answer_idx})
    return compact_json(
        {
            "task_family": "openai_graphwalks",
            "problem_type": problem_type,
            "nodes": nodes,
            "edges": edges,
        }
    )


def simhash_prefix(text: str, bits: int = 64, prefix_bits: int = 16) -> str:
    words = canonical_text(text).split()
    if not words:
        return "0" * (prefix_bits // 4)
    vector = [0] * bits
    shingles = words if len(words) < 5 else [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
    for token in shingles[:2048]:
        digest = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(bits):
            vector[bit] += 1 if digest & (1 << bit) else -1
    value = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            value |= 1 << bit
    return f"{value >> (bits - prefix_bits):0{prefix_bits // 4}x}"


def assign_split(group_hash: str, train_frac: float = 0.8, val_frac: float = 0.1) -> str:
    value = int(stable_hash(group_hash)[:12], 16) / float(16**12)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "validation"
    return "test"


def family_key(spec: DatasetSpec, record: dict[str, Any], question: str, answer: str) -> str:
    if spec.name == "open-r1/codeforces-cots":
        return str(record.get("id") or record.get("contest_id") or question)
    if spec.name == "open-r1/OpenR1-Math-220k":
        return str(record.get("uuid") or question)
    if spec.name == "openai/frontierscience":
        return str(record.get("task_group_id") or question or answer)
    if spec.name == "openai/healthbench":
        return str(record.get("prompt_id") or question or answer)
    if spec.name == "openai/healthbench-professional":
        return str(record.get("id") or question or answer)
    if spec.name == "openai/graphwalks":
        return stable_hash(str(record.get("prompt") or "") + str(record.get("problem_type") or ""))
    if spec.name == "unimorph/universal_morphologies":
        return f"{record.get('lemma','')}::{record.get('form','')}"
    if spec.name == "Sefaria/Rabbinic-Hebrew-English-Pairs":
        return str(record.get("ref") or question or answer)
    if spec.name == "Sefaria/hebrew_library":
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        return str(metadata.get("ref") or question or answer)
    if spec.name == "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b":
        return str(record.get("uuid") or question or answer)
    if spec.name == "Jackrong/GPT-OSS-120B-Distilled-Reasoning-math":
        return str(record.get("Input") or question or answer)
    if spec.name == "reasoning-degeneration-dev/algorithmic-sft-training-data-v1":
        return f"{record.get('task','')}::{record.get('algorithm','')}::{question}"
    if spec.name == "lamm-mit/graph-reasoning-messages-11K":
        return stable_hash(messages_to_text(record.get("messages")) or question or answer)
    if spec.name == "sequelbox/DAG-Reasoning-DeepSeek-R1-0528":
        return str(record.get("id") or record.get("uuid") or question or answer)
    if spec.name == "Gryphe/Opus-4.6-Reasoning-24k":
        return stable_hash(messages_to_text(record.get("messages")) or question or answer)
    if spec.name == "nvidia/Nemotron-RL-ReasoningGym-v1":
        return str(record.get("id") or record.get("uuid") or record.get("task") or question or answer)
    if spec.name == "nvidia/Nemotron-Content-Safety-Reasoning-Dataset":
        return stable_hash(str(record.get("prompt") or "") + str(record.get("response") or "") + question + answer)
    if spec.name == "nvidia/PhysicalAI-Traffic-Anomaly-Reasoning":
        return str(record.get("video_id") or record.get("clip_id") or record.get("id") or question or answer)
    return question or answer


def normalize_record(
    spec: DatasetSpec,
    record: dict[str, Any],
    source_split: str,
    source_index: int,
    seed: int = 17,
) -> dict[str, Any]:
    question = first_string(
        record,
        (
            "problem",
            "question",
            "Question",
            "instruction",
            "prompt",
            "input",
            "Input",
            "description",
            "english",
            "lemma",
            "ref",
            "category",
            "task",
            "messages",
        ),
    )
    solution = first_string(
        record,
        (
            "solution",
            "generation",
            "output",
            "Output",
            "Answer",
            "answer",
            "Anwser",
            "answer_content",
            "response",
            "completion",
            "sft_trace",
            "hebrew",
            "form",
            "text",
            "he",
            "en",
        ),
    )
    answer = first_string(record, ("answer", "Answer", "Anwser", "answer_content", "output", "Output", "he", "text"))
    reasoning = first_string(
        record,
        (
            "CoT_Native_Reasoning",
            "CoT_content",
            "sft_trace",
            "reasoning",
            "rationale",
            "solution",
            "generation",
            "output",
            "Output",
            "Answer",
            "en",
            "text",
        ),
    )
    if not answer:
        answer = extract_answer_from_solution(solution)
    if spec.name == "unimorph/universal_morphologies":
        question = f"lemma={record.get('lemma','')} features={record.get('features','')}"
        answer = prefer_pointed_variant(str(record.get("form", "")), str(record.get("form", "")))
        solution = compact_json(record)
        reasoning = str(record.get("features", ""))
    if spec.name == "Sefaria/Rabbinic-Hebrew-English-Pairs":
        question = str(record.get("ref") or record.get("category") or "")
        answer = prefer_pointed_variant(str(record.get("he") or ""), str(record.get("hebrew") or ""))
        solution = compact_json({"he": record.get("he"), "en": record.get("en"), "category": record.get("category")})
        reasoning = str(record.get("en") or "")
    if spec.name == "Sefaria/hebrew_library":
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        question = str(metadata.get("ref") or metadata.get("docCategory") or "")
        answer = prefer_pointed_variant(str(record.get("text") or ""), str(record.get("he") or ""))
        solution = str(record.get("text") or "")
        reasoning = solution
    if spec.name == "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b":
        question = str(record.get("input") or "")
        answer = str(record.get("output") or "")
        solution = answer
        reasoning = answer
    if spec.name == "Jackrong/GPT-OSS-120B-Distilled-Reasoning-math":
        question = str(record.get("Input") or "")
        answer = str(record.get("Anwser") or "")
        solution = answer
        reasoning = str(record.get("CoT_Native_Reasoning") or answer)
    if spec.name == "Jackrong/GPT-OSS-20B-Distilled-Reasoning-Mini":
        question = str(record.get("input") or "")
        answer = str(record.get("answer_content") or "")
        solution = answer
        reasoning = str(record.get("CoT_content") or answer)
    if spec.name == "reasoning-degeneration-dev/algorithmic-sft-training-data-v1":
        question = str(record.get("question") or "")
        answer = str(record.get("answer") or "")
        solution = str(record.get("sft_trace") or answer)
        reasoning = solution
    if spec.name == "openai/frontierscience":
        question = str(record.get("problem") or "")
        answer = str(record.get("answer") or "")
        solution = answer
        reasoning = str(record.get("subject") or "")
    if spec.name == "openai/healthbench":
        prompt_text = messages_to_text(record.get("prompt"))
        rubrics = rubric_items_to_text(record.get("rubrics"))
        question = prompt_text
        answer = rubrics
        solution = rubrics
        reasoning = rubrics
    if spec.name == "openai/healthbench-professional":
        question = messages_to_text(record.get("conversation"))
        answer = str(record.get("physician_response") or "")
        rubrics = rubric_items_to_text(record.get("rubric_items"))
        solution = "\n\n".join(part for part in (answer, rubrics) if part)
        reasoning = rubrics
    if spec.name == "openai/graphwalks":
        question = str(record.get("prompt") or "")
        answer_nodes = record.get("answer_nodes")
        answer = compact_json(answer_nodes if isinstance(answer_nodes, list) else [])
        solution = answer
        reasoning = str(record.get("problem_type") or "")
    if spec.name in {"lamm-mit/graph-reasoning-messages-11K", "Gryphe/Opus-4.6-Reasoning-24k"}:
        messages = record.get("messages")
        user_text = messages_to_text(messages, role="user")
        assistant_text = messages_to_text(messages, role="assistant")
        question = user_text or messages_to_text(messages)
        answer = assistant_text
        solution = assistant_text
        reasoning = assistant_text
    if spec.name == "sequelbox/DAG-Reasoning-DeepSeek-R1-0528":
        question = first_string(record, ("input", "prompt", "question", "problem", "messages"))
        answer = first_string(record, ("output", "answer", "response", "completion"))
        solution = first_string(record, ("reasoning", "rationale", "output", "answer", "response", "completion"))
        reasoning = solution
    if spec.name == "nvidia/Nemotron-RL-ReasoningGym-v1":
        question = first_string(record, ("question", "prompt", "problem", "input", "task"))
        answer = first_string(record, ("answer", "solution", "target", "output"))
        solution = first_string(record, ("solution", "reasoning", "rationale", "answer", "output"))
        reasoning = solution
    if spec.name == "nvidia/Nemotron-Content-Safety-Reasoning-Dataset":
        question = first_string(record, ("prompt", "user_prompt", "question", "input"))
        answer = first_string(record, ("label", "answer", "output", "completion", "response"))
        solution = first_string(record, ("reasoning", "rationale", "justification", "explanation", "output"))
        reasoning = solution
    if spec.name == "nvidia/PhysicalAI-Traffic-Anomaly-Reasoning":
        question = first_string(record, ("question", "prompt", "query", "task", "video_id"))
        answer = first_string(record, ("answer", "response", "output", "label"))
        solution = first_string(record, ("reasoning", "rationale", "chain_of_thought", "explanation", "answer"))
        reasoning = solution

    text = "\n\n".join(part for part in (question, reasoning, solution, answer) if part)
    normalized = canonical_text(text)
    content_hash = stable_hash(normalized)
    sim_prefix = simhash_prefix(normalized)
    group_source = f"{spec.task_family}::{family_key(spec, record, question, answer)}::{sim_prefix}"
    group_hash = stable_hash(canonical_text(group_source))
    split = spec.forced_split or assign_split(group_hash)
    record_id = stable_hash(f"{spec.key}::{source_split}::{source_index}::{content_hash}")
    quality_flags = {
        "empty_question": not bool(question.strip()),
        "empty_solution": not bool(solution.strip()),
        "empty_answer": not bool(answer.strip()),
        "simhash_prefix": sim_prefix,
        "seed": seed,
    }
    niqqud_stats = hebrew_niqqud_stats(question, answer, solution, reasoning)
    if niqqud_stats.contains_hebrew:
        quality_flags.update(niqqud_stats.to_dict())
        record = {
            **record,
            "toricgt_hebrew_niqqud": niqqud_stats.to_dict(),
            "toricgt_niqqud_policy": "preserve existing niqqud; prefer same-consonant pointed variants when present; flag unpointed Hebrew",
        }
    return {
        "record_id": record_id,
        "dataset": spec.name,
        "config": spec.config,
        "source_split": source_split,
        "source_index": int(source_index),
        "task_family": spec.task_family,
        "language": spec.language,
        "license": spec.license,
        "role": spec.role,
        "question": question,
        "answer": answer,
        "solution": solution,
        "reasoning": reasoning,
        "metadata_json": compact_json(record),
        "text": text,
        "graph_json": make_graphwalks_graph_json(question, record.get("answer_nodes"), str(record.get("problem_type") or ""))
        if spec.name == "openai/graphwalks"
        else make_graph_json(question, reasoning or solution, answer, spec.task_family),
        "content_hash": content_hash,
        "group_hash": group_hash,
        "split": split,
        "estimated_tokens": len(text.split()),
        "quality_flags_json": compact_json(quality_flags),
    }


def write_manifest(path: str | Path, cfg: DataConfig | None = None) -> None:
    out = {
        "config": asdict(cfg) if cfg is not None else {},
        "datasets": DATASET_MANIFEST,
        "split_policy": "deterministic grouped 80/10/10 by task key, content hash, and simhash prefix",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
