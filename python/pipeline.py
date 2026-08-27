"""Generation pipeline: per-domain dependency-graph and call-graph file production.

Orchestrates extract -> cluster -> render -> persist for one domain at a time,
and drives the recursive multi-root walk for the full target tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clusterer import cluster_types_validated  # noqa: E402
from extract import extract, find_domains_in_tree, hash_files, own_direct_files  # noqa: E402
from render import render  # noqa: E402
from call_graph import extract_call_graph, render_call_graph, render_call_graph_html  # noqa: E402
from models import ExtractedGraph, PersistedGraph  # noqa: E402

GRAPH_DIRNAME = "dependency_graph"
GRAPH_JSON = "dependency_graph.json"
GRAPH_MERMAID = "dependency_graph.mermaid"
CALL_GRAPH_JSON = "call_graph.json"
CALL_GRAPH_MERMAID = "call_graph.mermaid"
CALL_GRAPH_HTML = "call_graph.html"


def cache_path_for(domain_dir: Path) -> Path:
    return domain_dir / GRAPH_DIRNAME / GRAPH_JSON


def load_cached(domain_dir: Path) -> PersistedGraph | None:
    p = cache_path_for(domain_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def trim_for_persistence(extracted: ExtractedGraph, target_dotted: str) -> dict:
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
