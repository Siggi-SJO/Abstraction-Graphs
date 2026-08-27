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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import _generate_dependency_graph  # noqa: E402
from mcp.server import MCPServer  # noqa: E402

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
