"""Ollama-based type-export clustering for the dependency-graph tool.

Sends a batch of {module_dotted: [type_names]} payloads to a local Ollama
instance and returns a {module_dotted: "types: ..." summary line} dict.
No side effects other than the HTTP call; fully standalone.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b-instruct"

CLUSTER_PROMPT = """For each Python module below, you're given its exported type names (classes, type \
aliases, enums -- not functions). For each module, produce ONE line clustering those \
names into 2-5 human-meaningful groups by shared purpose (not just restating the raw \
list -- actually group by what the types are for, e.g. a shared suffix/pattern, an \
evident small union/variant family, a shared role). Format:

    types: <group name> (<count>) &middot; <group name> (<count>) &middot; ...

A group with only one or two members can just list the bare name(s) instead of a \
count-only label if that reads better. Keep each module's line to exactly one line.

Worked examples of the target style:
- ["CategoryBucket","CategoryContract","CategoryInstance","Slot","FixedSlot","SemanticSlot","ParameterizedSlot","ProseSlot","TableSlot","Parameter","ParamType","Column","RequiredInformation","ProseExample"]
  -> "types: category model (3) &middot; Slot union (6) &middot; parameter model (3) &middot; prose evidence (2)"
- ["DocumentInput","UnsupportedFormatError","IngestedDocument","Node","Section","Paragraph","Table","TableRow","Sentence","SourceAnchor"]
  -> "types: document tree (6), DocumentInput, UnsupportedFormatError, Sentence, SourceAnchor"

Output a JSON object mapping each module's dotted path to its one-line summary string. \
Input:

{payload}
"""


def cluster_types(payload: dict[str, list[str]]) -> dict[str, str]:
    if not payload:
        return {}
    prompt = CLUSTER_PROMPT.format(payload=json.dumps(payload, indent=2))
    body = json.dumps(
        {"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False}
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} (is `ollama serve` running?): {e}"
        ) from e
    text = result.get("response", "")
    try:
        clusters = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama did not return valid JSON: {text!r}") from e
    return {k: v.replace("&amp;middot;", "&middot;") for k, v in clusters.items()}


def cluster_covers_all(cluster_line: str, names: list[str]) -> bool:
    """Cheap completeness check: sum every '(N)' count plus every bare (uncounted)
    comma-separated name, and require it to equal len(names). Catches a model quietly
    dropping names from its clustering rather than trusting the line at face value."""
    total = sum(int(n) for n in re.findall(r"\((\d+)\)", cluster_line))
    for segment in re.split(r"&middot;|·", cluster_line):
        if "(" in segment:
            continue
        bare = [n.strip() for n in segment.replace("types:", "").split(",") if n.strip()]
        total += len(bare)
    return total == len(names)


def cluster_types_validated(payload: dict[str, list[str]]) -> dict[str, str]:
    """cluster_types, but re-checked per domain for completeness, with a single retry
    (smaller, one-domain-at-a-time, easier for the model to get right) and a safe
    ungrouped-but-complete fallback if it still doesn't add up."""
    if not payload:
        return {}
    clusters = cluster_types(payload)
    out = {}
    for dotted, names in payload.items():
        line = clusters.get(dotted)
        if line and cluster_covers_all(line, names):
            out[dotted] = line
            continue
        retry = cluster_types({dotted: names})
        line = retry.get(dotted)
        if line and cluster_covers_all(line, names):
            out[dotted] = line
        else:
            out[dotted] = "types: " + ", ".join(names)
    return out
