#!/usr/bin/env python3
"""Render extract.py's JSON (plus a per-domain type-cluster summary) into the repo's
established mermaid dependency-graph format. Pure string templating — no judgment left
to make here; the one creative step (naming the type clusters) already happened upstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PALETTE = [
    ("blue", "#2a78d6", "#184f95", "#ffffff"),
    ("orange", "#eb6834", "#a84e1f", "#1a1a1a"),
    ("aqua", "#1baf7a", "#128257", "#1a1a1a"),
    ("yellow", "#eda100", "#a86c00", "#1a1a1a"),
    ("magenta", "#e87ba4", "#a84f6f", "#1a1a1a"),
    ("green", "#008300", "#004d00", "#ffffff"),
    ("violet", "#4a3aa7", "#2f2569", "#ffffff"),
    ("red", "#e34948", "#a3201f", "#ffffff"),
]

BOUNDARY = "classDef boundary fill:#f5f5f5,stroke:#999999,color:#4d4d4d,stroke-width:1px;"
BROKEN = "classDef broken fill:#d03b3b,stroke:#8a1f1f,stroke-dasharray: 2 2,color:#ffffff;"
EXPORT = "classDef export fill:#4a3aa7,stroke:#2f2569,color:#ffffff,stroke-width:1px;"


def light_tint(hex_fill: str) -> str:
    r, g, b = int(hex_fill[1:3], 16), int(hex_fill[3:5], 16), int(hex_fill[5:7], 16)
    r, g, b = [int(c + (255 - c) * 0.9) for c in (r, g, b)]
    return f"#{r:02x}{g:02x}{b:02x}"


def build_forest(domain_list: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Nest by dotted-path containment within `domain_list` only -- used both for
    the target's own in-tree domains and, separately, for each external boundary
    domain's own sub-tree (e.g. services.storage.backends inside services.storage
    inside services), which may itself be several levels deep. Shared with
    call_graph.py's own nesting, since "nest this domain list by dotted-path
    containment" is the same problem regardless of what's rendered inside each one."""

    def most_specific_parent(dom: dict) -> dict | None:
        best = None
        for cand in domain_list:
            if cand["id"] == dom["id"]:
                continue
            if dom["dotted"].startswith(cand["dotted"] + "."):
                if best is None or len(cand["dotted"]) > len(best["dotted"]):
                    best = cand
        return best

    children = {d["id"]: [] for d in domain_list}
    roots = []
    for d in domain_list:
        parent = most_specific_parent(d)
        if parent is None:
            roots.append(d)
        else:
            children[parent["id"]].append(d)
    return roots, children


def render(extracted: dict, clusters: dict[str, str]) -> str:
    lines = ["---", "config:", "  layout: elk", "---", "graph BT"]
    palette_for = {
        dom["id"]: PALETTE[i % len(PALETTE)] for i, dom in enumerate(extracted["domains"])
    }

    def emit_domain_body(dom: dict, pad: str) -> list[str]:
        out = []
        for f in dom["files"]:
            out.append(f'{pad}{f["id"]}[{f["filename"]}]')
        init_id = f'{dom["id"]}___init__'
        for fn in dom["functions"]:
            out.append(f'{pad}{dom["id"]}_export_{fn}([{fn}])')
        if dom["types_raw"]:
            summary = clusters.get(dom["dotted"], f'types: {", ".join(dom["types_raw"])}')
            out.append(f'{pad}{dom["id"]}_export_types[["{summary}"]]')
        for fn in dom["functions"]:
            out.append(f'{pad}{init_id} --> {dom["id"]}_export_{fn}')
        if dom["types_raw"]:
            out.append(f'{pad}{init_id} --> {dom["id"]}_export_types')
        return out

    def emit_nested(dom: dict, children: dict[str, list[dict]], indent: str) -> list[str]:
        out = [f'{indent}subgraph {dom["id"]}["{dom["dotted"]}"]']
        pad = indent + "    "
        out.extend(emit_domain_body(dom, pad))
        for child in children[dom["id"]]:
            out.append("")
            out.extend(emit_nested(child, children, pad))
        out.append(f"{indent}end")
        return out

    in_tree = [d for d in extracted["domains"] if d["in_tree"]]
    external = [d for d in extracted["domains"] if not d["in_tree"]]

    in_tree_roots, in_tree_children = build_forest(in_tree)
    for root in in_tree_roots:  # color_order puts the target first, so this is just the one
        lines.extend(emit_nested(root, in_tree_children, "    "))

    ext_roots, ext_children = build_forest(external)
    for root in ext_roots:
        lines.append("")
        lines.extend(emit_nested(root, ext_children, "    "))
    lines.append("")

    for sf in extracted["standalone_files"]:
        lines.append(f'    {sf["id"]}[{sf["filename"]}]')
    for b in extracted["broken"]:
        lines.append(f'    {b["id"]}["{b["dotted"]} ⚠ missing"]')
    lines.append("")

    for e in extracted["edges"]:
        if e["kind"] == "missing":
            lines.append(f'    {e["from"]} -.->|missing| {e["to"]}')
        elif e["kind"] == "violation":
            lines.append(f'    {e["from"]} -.->|boundary violation| {e["to"]}')
        else:
            lines.append(f'    {e["from"]} --> {e["to"]}')
    lines.append("")

    for dom in extracted["domains"]:
        name, fill, stroke, color = palette_for[dom["id"]]
        lines.append(
            f'    classDef {dom["id"]}Domain fill:{fill},stroke:{stroke},color:{color},stroke-width:1px;'
        )
    lines.append(f"    {BOUNDARY}")
    lines.append(f"    {BROKEN}")
    lines.append(f"    {EXPORT}")
    lines.append("")

    for dom in extracted["domains"]:
        ids = [f'{dom["id"]}___init__'] + [f["id"] for f in dom["files"] if f["filename"] != "__init__.py"]
        lines.append(f'    class {",".join(ids)} {dom["id"]}Domain')
    if extracted["standalone_files"]:
        ids = [s["id"] for s in extracted["standalone_files"]]
        lines.append(f'    class {",".join(ids)} boundary')
    if extracted["broken"]:
        ids = [b["id"] for b in extracted["broken"]]
        lines.append(f'    class {",".join(ids)} broken')
    export_ids = []
    for dom in extracted["domains"]:
        export_ids += [f'{dom["id"]}_export_{fn}' for fn in dom["functions"]]
        if dom["types_raw"]:
            export_ids.append(f'{dom["id"]}_export_types')
    if export_ids:
        lines.append(f'    class {",".join(export_ids)} export')
    lines.append("")

    for dom in extracted["domains"]:
        _, fill, _, _ = palette_for[dom["id"]]
        tint = light_tint(fill)
        lines.append(f'    style {dom["id"]} fill:{tint},stroke:{fill},stroke-width:2px;')

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extracted_json")
    ap.add_argument("--clusters", help="JSON file: {domain_dotted_path: 'types: ...' summary line}")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    extracted = json.loads(Path(args.extracted_json).read_text())
    clusters = json.loads(Path(args.clusters).read_text()) if args.clusters else {}
    Path(args.output).write_text(render(extracted, clusters))


if __name__ == "__main__":
    main()
