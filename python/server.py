#!/usr/bin/env python3
"""MCP server exposing one tool: generate_dependency_graph.

Recursive: processes a target module and every nested domain (submodule) underneath it,
each getting its own <domain>/dependency_graph/{dependency_graph.json,dependency_graph.mermaid,
call_graph.json,call_graph.mermaid}.

Pipeline per domain: extract.py (ast, no model, always self-sufficient/derived from
source) -> a local Ollama call for type-export clustering, skipped when a fresh cached
cluster string already exists for that exact name set -> render.py (string templating,
no model) for the dependency graph, and call_graph.py (ast + string templating, no
model) for the function call graph, reusing the same extract() result. Change detection
is a content hash over each domain's own direct files; regeneration cascades upward
when any nested submodule regenerates, since a parent's rendered graph embeds its
submodules' data. The call graph regenerates on exactly the same trigger as the
dependency graph -- it's a byproduct of the same generate_one() call, not a separately
cached pass.

Fully standalone: no dependency on Claude Code's Agent/subagent mechanism, no network
calls except to the local Ollama daemon.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract import extract, find_domains_in_tree, hash_files, own_direct_files  # noqa: E402
from render import render  # noqa: E402
from call_graph import extract_call_graph, render_call_graph, render_call_graph_html  # noqa: E402

from mcp.server import MCPServer  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b-instruct"
GRAPH_DIRNAME = "dependency_graph"
GRAPH_JSON = "dependency_graph.json"
GRAPH_MERMAID = "dependency_graph.mermaid"
CALL_GRAPH_JSON = "call_graph.json"
CALL_GRAPH_MERMAID = "call_graph.mermaid"
CALL_GRAPH_HTML = "call_graph.html"

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


def cache_path_for(domain_dir: Path) -> Path:
    return domain_dir / GRAPH_DIRNAME / GRAPH_JSON


def load_cached(domain_dir: Path) -> dict | None:
    p = cache_path_for(domain_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def trim_for_persistence(extracted: dict, target_dotted: str) -> dict:
    """The persisted JSON for one domain covers that domain's own data -- not its
    submodules'. Nested in-tree children (other domains under this one's own directory
    tree) are trimmed to a pointer at their own dependency_graph.json; the target's own
    entry, and any external boundary domain (not a submodule -- a dependency), stay in
    full since the render needs them and they aren't duplicated data."""
    trimmed = dict(extracted)
    new_domains = []
    for dom in extracted["domains"]:
        if dom["dotted"] == target_dotted or not dom["in_tree"]:
            new_domains.append(dom)
        else:
            new_domains.append(
                {
                    "dotted": dom["dotted"],
                    "id": dom["id"],
                    "in_tree": True,
                    "see": f"{dom['dotted'].replace('.', '/')}/{GRAPH_DIRNAME}/{GRAPH_JSON}",
                }
            )
    trimmed["domains"] = new_domains
    return trimmed


def generate_one(
    repo_root: Path, domain_dir: Path, clusters_full: dict[str, str], do_reverse_scan: bool = True,
    force: bool = False,
) -> dict:
    """Regenerate one domain's own dependency_graph/ files -- always: extract() is
    cheap/local (no network) and render()/render_call_graph()/render_call_graph_html()
    are pure Python, so this runs on every call, changed or not, and a rendering code
    change always shows up on the next regen without needing `force`. `clusters_full`
    accumulates every cluster string produced or reused this run, keyed by dotted
    path, so later domains in the same run (and future runs, via each domain's own
    cached types_cluster) can reuse rather than re-ask Ollama for an unchanged type
    list -- the one thing here that's actually expensive. `force` bypasses that reuse
    too, asking Ollama again even when a domain's types_raw is byte-identical to what's
    cached; use it if the clustering itself needs redoing (e.g. CLUSTER_PROMPT
    changed), not for an ordinary "did anything change" regen. `do_reverse_scan` is
    False for a domain pulled in as a boundary dependency or importer -- only the
    originally-requested target and its own submodules get "who imports me" treatment,
    so the reverse scan doesn't recurse upward without bound."""
    extracted = extract(repo_root, domain_dir, do_reverse_scan=do_reverse_scan)
    target_dotted = extracted["target"]

    payload = {}
    for dom in extracted["domains"]:
        if not dom["types_raw"]:
            continue
        if dom["dotted"] in clusters_full:
            continue  # already have a cluster string for this exact domain this run -- reused
            # regardless of `force`, since re-asking Ollama twice for the identical
            # input within the same run has no benefit
        cached = None if force else load_cached(Path(repo_root, *dom["dotted"].split(".")))
        if cached and cached.get("types_raw") == dom["types_raw"] and cached.get("types_cluster"):
            clusters_full[dom["dotted"]] = cached["types_cluster"]
            continue
        payload[dom["dotted"]] = dom["types_raw"]

    if payload:
        clusters_full.update(cluster_types_validated(payload))

    mermaid = render(extracted, clusters_full)
    persisted = trim_for_persistence(extracted, target_dotted)
    own_files = own_direct_files(domain_dir)
    persisted["own_content_hash"] = hash_files(own_files)
    persisted["reverse_scan_applied"] = do_reverse_scan
    own_dom = next(d for d in extracted["domains"] if d["dotted"] == target_dotted)
    persisted["types_cluster"] = clusters_full.get(target_dotted)

    call_graph = extract_call_graph(repo_root, extracted)
    call_graph_mermaid = render_call_graph(call_graph)
    call_graph_html = render_call_graph_html(call_graph)

    out_dir = domain_dir / GRAPH_DIRNAME
    out_dir.mkdir(exist_ok=True)
    (out_dir / GRAPH_JSON).write_text(json.dumps(persisted, indent=2))
    (out_dir / GRAPH_MERMAID).write_text(mermaid)
    (out_dir / CALL_GRAPH_JSON).write_text(json.dumps(call_graph, indent=2))
    (out_dir / CALL_GRAPH_MERMAID).write_text(call_graph_mermaid)
    (out_dir / CALL_GRAPH_HTML).write_text(call_graph_html)

    return {
        "dotted": target_dotted,
        "domain_count": len(extracted["domains"]),
        "domains": extracted["domains"],
        "broken": extracted["broken"],
        "violations": extracted["violations"],
    }


def external_boundary_roots(domains_list: list[dict]) -> list[str]:
    """The root dotted path of each distinct external (non-submodule) boundary domain
    referenced -- e.g. just "services", not also "services.storage" underneath it, since
    walking "services" as its own recursive target rediscovers that nested tree anyway."""
    ext = [d for d in domains_list if not d.get("in_tree", True)]
    roots = []
    for d in ext:
        if not any(
            d["dotted"] != e["dotted"] and d["dotted"].startswith(e["dotted"] + ".")
            for e in ext
        ):
            roots.append(d["dotted"])
    return roots


def _generate_dependency_graph(target: str, repo_root: str = "be", force: bool = False) -> str:
    """The actual recursive regeneration walk -- pulled out of the `@server.tool()`
    function so a plain CLI entrypoint (main(), below) can call it too, without going
    through MCP. See generate_dependency_graph's docstring for the full behavior."""
    repo_root_path = Path(repo_root).resolve()
    target_dir = (repo_root_path / target).resolve()

    if not (target_dir / "__init__.py").is_file():
        return f"Error: {target_dir} has no __init__.py -- not a module/domain."

    # A "root" here is any domain directory whose own recursive tree gets walked --
    # the requested target, and (queued as they're discovered) every external boundary
    # domain any processed domain turned out to depend on. Same walk, just re-run with
    # a different root, so a boundary domain like `services` ends up with its own
    # dependency_graph/ tree exactly like a real submodule would.
    #
    # do_reverse_scan is True only for the originally-requested target (and, since it's
    # applied per-root and a root's own find_domains_in_tree walk covers its submodules
    # too, for those submodules as well) -- a boundary/importer root discovered along
    # the way is queued with it False, so "who imports me" doesn't itself recurse
    # upward without bound.
    to_process = [(target_dir, True)]
    processed_roots: set[Path] = set()
    processed_domains: set[Path] = set()
    regenerated: set[Path] = set()
    clusters_full: dict[str, str] = {}
    report_lines = []

    while to_process:
        root_dir, root_do_reverse = to_process.pop(0)
        if root_dir in processed_roots:
            continue
        processed_roots.add(root_dir)

        try:
            all_domains = find_domains_in_tree(repo_root_path, root_dir)
        except Exception as e:
            report_lines.append(f"- {root_dir}: ERROR discovering submodules: {e}")
            continue

        # deepest first, so a submodule is always regenerated (or confirmed unchanged)
        # before the domain that embeds it needs to read its dependency_graph.json
        ordered = sorted(all_domains.values(), key=lambda d: len(d.dir_path.parts), reverse=True)

        for dom in ordered:
            if dom.dir_path in processed_domains:
                continue
            processed_domains.add(dom.dir_path)

            nested = find_domains_in_tree(repo_root_path, dom.dir_path)
            has_regenerated_child = any(
                d.dir_path in regenerated for dt, d in nested.items() if d.dir_path != dom.dir_path
            )
            own_hash = hash_files(own_direct_files(dom.dir_path))
            cached = load_cached(dom.dir_path)
            unchanged = (
                cached is not None
                and cached.get("own_content_hash") == own_hash
                and not has_regenerated_child
                and cached.get("reverse_scan_applied") == root_do_reverse
            )
            # generate_one() always runs, changed or not -- extract() is cheap/local
            # (no network), and re-rendering .mermaid/.html from it means a rendering
            # code change (call_graph.py, render.py) always shows up on the next
            # regen, without needing --force. --force only still matters for the one
            # thing that's genuinely expensive: it's passed through as `force` here so
            # a truly UNCHANGED domain's types_raw still short-circuits the Ollama
            # clustering call inside generate_one() regardless -- only `force` bypasses
            # that reuse too and asks Ollama again even when types_raw is identical.
            try:
                result = generate_one(
                    repo_root_path, dom.dir_path, clusters_full, do_reverse_scan=root_do_reverse, force=force
                )
            except Exception as e:
                report_lines.append(f"- {dom.dotted}: ERROR: {e}")
                continue
            regenerated.add(dom.dir_path)
            status = "regenerated" if not unchanged else "up to date, re-rendered"
            note = f"- {dom.dotted}: {status} ({result['domain_count']} domains in view)"
            if result["broken"]:
                note += "; broken: " + "; ".join(
                    f"{b['importer']}->{b['dotted']}" for b in result["broken"]
                )
            if result["violations"]:
                note += "; violations: " + "; ".join(
                    f"{v['from']}->{v['to']}" for v in result["violations"]
                )
            report_lines.append(note)
            domains_seen = result["domains"]

            for root_dotted in external_boundary_roots(domains_seen):
                root_path = repo_root_path / Path(*root_dotted.split("."))
                if root_path not in processed_roots and not any(
                    p == root_path for p, _ in to_process
                ):
                    to_process.append((root_path, False))

    header = f"Processed {len(processed_domains)} domain(s), starting from {target}:"
    return header + "\n" + "\n".join(report_lines)


server = MCPServer("dependency-graph")


@server.tool()
def generate_dependency_graph(target: str, repo_root: str = "be", force: bool = False) -> str:
    """Recursively regenerate <domain>/dependency_graph/{dependency_graph.json,
    dependency_graph.mermaid,call_graph.json,call_graph.mermaid} for `target` and every
    nested submodule underneath it, per this repo's established mermaid convention
    (domains as colored subgraphs, exported functions and a clustered type-summary above
    each domain's __init__.py, broken imports and module-boundary violations flagged
    distinctly). call_graph.mermaid is a second graph alongside the dependency graph:
    which tracked functions call which others -- same-file, or across a module boundary
    via `from x import name` (chased through re-export chains), or through a
    dispatch-table pattern (`fn = TABLE[key]; fn(...)` where TABLE is a module-level
    dict literal -- resolved conservatively as a call to every one of the table's
    values, since there's no constant-folding to know which key wins at a given call
    site) -- nested by domain then by file, same visual convention as the dependency
    graph. Internal (leading-underscore) functions never appear as nodes: a chain of
    calls through one or more of them collapses into a single edge between the public
    functions on either end, labeled with which internal functions mediated it (plain
    "returns" when the call was direct). Only public functions that end up part of at
    least one such edge get a node. An edge points callee --> caller (the callee is a
    dependency of its caller, matching this toolset's convention of "the thing depended
    on flows up to the thing depending on it"). A function that resolves a named type
    in its own return annotation gets a type node of its own (function --> type_node)
    only if it's the FIRST such function in its chain (closest to the leaves) -- if it
    also calls something, directly or transitively, that emits the same type, it's just
    relaying a value whose origin is further down, so it gets no node, but its own edge
    to its caller is still colored for that type, continuing the same thread one hop
    further up. One node instance per true first emitter, never shared globally, but
    every instance of the same type shares one color. Purely AST-derived, no type
    checker, no real dataflow tracing beyond the dispatch-table case -- a
    module-qualified call (`x.name(...)`) or a method call on an arbitrary object is
    never resolved, since neither import aliases nor object types are tracked here. A
    domain is skipped and left untouched if its own files' content
    hash is unchanged since it was last generated and none of its submodules were
    regenerated this run; otherwise it's rebuilt, reusing a submodule's already-generated
    dependency_graph.json (including its cached type-cluster summary) rather than
    recomputing it. `target` is a module directory (e.g. "templating" or
    "templating/contract"); `repo_root` is the directory local imports resolve against
    (default "be", i.e. rcg-agents/be). `force`, when true, ignores the content-hash
    cache entirely and regenerates every domain in the walk regardless of whether
    anything changed."""
    return _generate_dependency_graph(target, repo_root, force)


def main() -> None:
    """CLI entrypoint: `python3 server.py <target> [--repo-root be] [--force]` regenerates
    and prints the report, same as the MCP tool -- for callers (e.g. an editor keymap)
    that want to trigger generation without going through Claude Code/MCP at all. With no
    `target` given, falls back to running as the MCP stdio server (the normal path when
    Claude Code launches this file)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="module directory, e.g. templating/contract")
    ap.add_argument("--repo-root", default="be")
    ap.add_argument("--force", action="store_true", help="ignore the content-hash cache, regenerate everything")
    args = ap.parse_args()

    if args.target is None:
        server.run(transport="stdio")
    else:
        print(_generate_dependency_graph(args.target, args.repo_root, args.force))


if __name__ == "__main__":
    main()
