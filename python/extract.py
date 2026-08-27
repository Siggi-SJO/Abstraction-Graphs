#!/usr/bin/env python3
"""Static extraction of a module dependency graph, per the repo's mermaid convention
(see rcg-agents/.claude/commands/dependency-graph.md for the full spec this implements).

Pure `ast` parsing — the target code is never imported/executed. Output is a JSON
description of domains, files, exports (raw, unclustered), edges, broken imports, and
module-boundary violations. The one thing this deliberately leaves undone is naming/
grouping each domain's raw type-export list into human-readable clusters — that's the
one step handed to a model (see cluster.py / the dependency-graph slash command).
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

EXCLUDE_DIR_NAMES = {
    "tests",
    "test_scripts",
    "_retired",
    "__pycache__",
    ".venv",
    "venv",
    ".git",
    "node_modules",
    "site-packages",
    ".tox",
    "build",
    "dist",
    ".eggs",
}


def hash_files(files: list[Path]) -> str:
    """Stable content hash over a domain's own direct files -- order-independent
    (sorted by path), content-based (not mtime, so it survives checkouts/touches)."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted(set(files), key=str):
        h.update(str(f).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def iter_py_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        rel_parts = p.relative_to(root).parts[:-1]
        if any(part in EXCLUDE_DIR_NAMES for part in rel_parts):
            continue
        yield p


def dotted_path(repo_root: Path, file_path: Path) -> str:
    """The dotted *import* path. For an __init__.py this is the package's own dotted
    path (same as its domain) — that's correct for import resolution, but callers that
    need a distinct id for the __init__.py *file itself* must use file_node_id below."""
    parts = list(file_path.relative_to(repo_root).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def node_id(dotted: str) -> str:
    return dotted.replace(".", "_")


def file_node_id(repo_root: Path, file_path: Path) -> str:
    dotted = dotted_path(repo_root, file_path)
    if file_path.name == "__init__.py":
        return node_id(dotted) + "___init__"
    return node_id(dotted)


def is_domain_dir(dir_path: Path) -> bool:
    return (dir_path / "__init__.py").is_file()


def resolve_target(repo_root: Path, module: str) -> tuple[str, Path | None]:
    """Classify a dotted module path against the repo. Returns (kind, path):
    'package' (a domain's __init__.py), 'file' (a plain .py file), 'missing' (doesn't
    resolve), 'external' (not a repo-local top-level package at all)."""
    if not module:
        return "missing", None
    top = module.split(".")[0]
    if not (repo_root / top).is_dir() and not (repo_root / f"{top}.py").is_file():
        return "external", None
    rel = Path(*module.split("."))
    file_candidate = repo_root / f"{rel}.py"
    pkg_candidate = repo_root / rel / "__init__.py"
    if pkg_candidate.is_file():
        return "package", pkg_candidate
    if file_candidate.is_file():
        return "file", file_candidate
    return "missing", None


def sibling_dotted(repo_root: Path, file_path: Path, module: str) -> str | None:
    """If `module` resolves to a .py file or package that is a sibling of `file_path`
    (and lies within repo_root), return its repo-root-relative dotted path. Catches
    the sys.path.insert(0, script_dir) pattern where scripts import sibling modules
    by bare name rather than as a dotted package path."""
    top = module.split(".")[0]
    file_dir = file_path.parent
    sibling_py = file_dir / f"{top}.py"
    sibling_pkg = file_dir / top / "__init__.py"
    if sibling_py.is_file():
        candidate = sibling_py
    elif sibling_pkg.is_file():
        candidate = sibling_pkg
    else:
        return None
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        return None
    base = dotted_path(repo_root, candidate)
    rest = module[len(top):]
    return (base + rest) if rest else base


@dataclass
class ImportRef:
    module: str
    names: list[str]


def _resolve_relative(own_package: str, level: int, module: str | None) -> str:
    parts = own_package.split(".") if own_package else []
    if level > 1:
        parts = parts[: -(level - 1)] if len(parts) >= level - 1 else []
    base = ".".join(parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def parse_imports(file_path: Path, file_dotted: str) -> list[ImportRef]:
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError, OSError):
        return []
    own_package = file_dotted if file_path.name == "__init__.py" else (
        file_dotted.rsplit(".", 1)[0] if "." in file_dotted else ""
    )
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (
                _resolve_relative(own_package, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            out.append(ImportRef(module, [a.name for a in node.names]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append(ImportRef(alias.name, []))
    return out


def read_all_list(init_path: Path) -> list[str]:
    tree = ast.parse(init_path.read_text(), filename=str(init_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "__all__" in targets and isinstance(node.value, (ast.List, ast.Tuple)):
                return [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
    return []


def _defined_directly(file_path: Path, name: str) -> str | None:
    tree = ast.parse(file_path.read_text(), filename=str(file_path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "function"
        if isinstance(node, ast.ClassDef) and node.name == name:
            return "type"
    return None


def resolve_symbol_kind(repo_root: Path, module_dotted: str, name: str, cache: dict) -> str:
    """'function' if `name` (imported from `module_dotted`) ultimately traces back to a
    module-level `def`, else 'type'. Follows re-export chains (a name may be imported
    into a package's __init__.py from an internal file, or re-exported again by a
    parent package's __init__.py) with memoization, independent of processing order."""
    key = (module_dotted, name)
    if key in cache:
        return cache[key]
    cache[key] = "type"  # break any accidental cycle defensively
    kind, path = resolve_target(repo_root, module_dotted)
    result = "type"
    if kind == "file":
        result = _defined_directly(path, name) or "type"
    elif kind == "package":
        found = None
        for imp in parse_imports(path, module_dotted):
            if name in imp.names:
                found = imp.module
                break
        if found:
            result = resolve_symbol_kind(repo_root, found, name, cache)
        else:
            result = _defined_directly(path, name) or "type"
    cache[key] = result
    return result


@dataclass
class Domain:
    dotted: str
    dir_path: Path
    in_tree: bool  # True if under the target's own directory tree
    files: list[Path] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    types_raw: list[str] = field(default_factory=list)
    init_internal_edges: list[str] = field(default_factory=list)  # dotted paths of files __init__ imports


def own_direct_files(domain_dir: Path) -> list[Path]:
    """Files directly inside `domain_dir`, excluding anything that belongs to a nested
    domain (a subdirectory with its own __init__.py) -- exactly the file set that ends
    up in that domain's *own* persisted JSON, not its submodules'."""
    nested_dirs = [
        d
        for d in domain_dir.rglob("*")
        if d.is_dir()
        and d != domain_dir
        and is_domain_dir(d)
        and not any(part in EXCLUDE_DIR_NAMES for part in d.relative_to(domain_dir).parts)
    ]
    out = []
    for f in iter_py_files(domain_dir):
        if any(nd in f.parents for nd in nested_dirs):
            continue
        out.append(f)
    return out


def find_domains_in_tree(repo_root: Path, target_dir: Path) -> dict[str, Domain]:
    domains: dict[str, Domain] = {}
    nested = [
        d
        for d in sorted(target_dir.rglob("*"))
        if d.is_dir()
        and is_domain_dir(d)
        and not any(part in EXCLUDE_DIR_NAMES for part in d.relative_to(target_dir).parts)
    ]
    for d in [target_dir] + nested:
        dotted = dotted_path(repo_root, d / "__init__.py")
        domains[dotted] = Domain(dotted=dotted, dir_path=d, in_tree=True)

    def owning_domain(file_path: Path) -> Domain:
        best = None
        for dom in domains.values():
            try:
                file_path.relative_to(dom.dir_path)
            except ValueError:
                continue
            if best is None or len(dom.dir_path.parts) > len(best.dir_path.parts):
                best = dom
        return best

    for f in iter_py_files(target_dir):
        owning_domain(f).files.append(f)
    return domains


def topmost_domain_dir(repo_root: Path, file_path: Path) -> Path | None:
    """The shallowest ancestor directory of `file_path` that has an __init__.py -- i.e.
    the root of whatever package this file belongs to, however deep the file itself
    sits. Used so a reverse-importer found deep inside some other package (e.g.
    services/storage/backends/x.py) gets registered as that package's own top-level
    domain, not an orphaned, unnested fragment of it."""
    ancestors = []
    p = file_path.parent
    while p != repo_root and repo_root in p.parents:
        ancestors.append(p)
        p = p.parent
    for d in reversed(ancestors):
        if is_domain_dir(d):
            return d
    return None


def most_specific_domain(domains: dict[str, Domain], dotted: str) -> Domain | None:
    best = None
    for dom in domains.values():
        if dotted == dom.dotted or dotted.startswith(dom.dotted + "."):
            if best is None or len(dom.dotted) > len(best.dotted):
                best = dom
    return best


def process_domain_exports(repo_root: Path, domain: Domain, symbol_cache: dict) -> None:
    init_path = domain.dir_path / "__init__.py"
    all_names = read_all_list(init_path)
    for name in all_names:
        kind = resolve_symbol_kind(repo_root, domain.dotted, name, symbol_cache)
        (domain.functions if kind == "function" else domain.types_raw).append(name)
    for imp in parse_imports(init_path, domain.dotted):
        candidates = [imp.module] if imp.module != domain.dotted else [
            f"{domain.dotted}.{n}" for n in imp.names
        ]
        for candidate in candidates:
            if not candidate.startswith(domain.dotted + "."):
                continue
            kind, _ = resolve_target(repo_root, candidate)
            if kind == "file":  # a real file, not a further-nested package/domain
                domain.init_internal_edges.append(candidate)


def extract(repo_root: Path, target_dir: Path, do_reverse_scan: bool = True) -> dict:
    target_dotted = dotted_path(repo_root, target_dir / "__init__.py")
    assert is_domain_dir(target_dir), f"{target_dir} has no __init__.py — not a domain"

    domains = find_domains_in_tree(repo_root, target_dir)
    symbol_cache: dict = {}
    for dom in domains.values():
        process_domain_exports(repo_root, dom, symbol_cache)

    standalone_files: dict[str, Path] = {}
    broken: dict[str, str] = {}
    edges = []
    violations = []

    def export_target_for(dom: Domain, name: str) -> str:
        if name in dom.functions:
            return f"{node_id(dom.dotted)}_export_{name}"
        return f"{node_id(dom.dotted)}_export_types"

    def register_external_domain(root_dotted: str, root_init_path: Path) -> Domain:
        """A boundary domain reached via import may itself contain nested domains
        (e.g. `services.storage.backends`) -- discover that whole sub-tree the same way
        `find_domains_in_tree` does for the target's own tree, not a flat file dump."""
        sub_domains = find_domains_in_tree(repo_root, root_init_path.parent)
        for sub_dotted, sub_dom in sub_domains.items():
            sub_dom.in_tree = False
            domains[sub_dotted] = sub_dom
            process_domain_exports(repo_root, sub_dom, symbol_cache)
        for sub_dotted, sub_dom in sub_domains.items():
            for f in sub_dom.files:
                f_dotted = dotted_path(repo_root, f)
                process_file_imports(f, f_dotted, sub_dom, scope_prefix=root_dotted)
            for target_dotted_ in sub_dom.init_internal_edges:
                _, path = resolve_target(repo_root, target_dotted_)
                if path:
                    edges.append({
                        "from": file_node_id(repo_root, path),
                        "to": file_node_id(repo_root, sub_dom.dir_path / "__init__.py"),
                        "kind": "internal",
                    })
        return domains[root_dotted]

    def process_file_imports(
        f: Path,
        f_dotted: str,
        f_domain: Domain | None,
        scope_prefix: str | None = None,
        restrict_to: set[str] | None = None,
    ):
        """`scope_prefix`, when set, is the one-hop boundary: this file already lives
        inside an external domain we pulled in, so a further import is only allowed to
        register yet another new domain if it stays within `scope_prefix`'s own tree or
        is already known -- otherwise it's silently out of scope, same as the original
        single-level external-domain handling.

        `restrict_to`, when set, only processes imports resolving into one of these
        dotted paths (or a submodule of one) -- used for the reverse pass, where a file
        elsewhere in the repo might import lots of things and only the ones touching
        this graph's own target/submodules are relevant here."""
        f_id = file_node_id(repo_root, f)
        for imp in parse_imports(f, f_dotted):
            if restrict_to is not None and not (
                imp.module in restrict_to
                or any(imp.module.startswith(t + ".") for t in restrict_to)
            ):
                continue
            kind, path = resolve_target(repo_root, imp.module)
            if kind == "external":
                resolved = sibling_dotted(repo_root, f, imp.module)
                if resolved is None:
                    continue
                imp = ImportRef(resolved, imp.names)
                kind, path = resolve_target(repo_root, resolved)
                if kind not in ("file", "package"):
                    continue
            if kind == "missing":
                broken.setdefault(imp.module, f_dotted)
                edges.append({"from": f_id, "to": node_id(imp.module), "kind": "missing"})
                continue
            if kind == "package":
                dom = domains.get(imp.module)
                if dom is None:
                    if scope_prefix is not None and not (
                        imp.module == scope_prefix or imp.module.startswith(scope_prefix + ".")
                    ):
                        continue  # one-hop rule: don't chase a second external boundary
                    dom = register_external_domain(imp.module, path)
                seen_targets = set()
                for name in imp.names:
                    export_id = export_target_for(dom, name)
                    if export_id in seen_targets:
                        continue
                    seen_targets.add(export_id)
                    edges.append({"from": export_id, "to": f_id, "kind": "normal"})
                continue
            if kind == "file":
                owning = most_specific_domain(domains, imp.module)
                fname = path.name
                is_internal_name = fname.startswith("_") and fname != "__init__.py"
                if owning is not None and is_internal_name:
                    target_id = file_node_id(repo_root, path)
                    if f_domain is not None and f_domain.dotted == owning.dotted:
                        edges.append({"from": target_id, "to": f_id, "kind": "internal"})
                    else:
                        violations.append({"from": f_id, "to": target_id})
                        edges.append({"from": target_id, "to": f_id, "kind": "violation"})
                    continue
                if owning is not None and path in owning.files:
                    # a plain-named file that already belongs to a known domain's own
                    # file list (already gets a node there) -- just the edge, no
                    # separate standalone node, or it'd be declared twice.
                    target_id = file_node_id(repo_root, path)
                    edge_kind = "internal" if f_domain is not None and f_domain.dotted == owning.dotted else "normal"
                    edges.append({"from": target_id, "to": f_id, "kind": edge_kind})
                    continue
                # a flat file outside any known domain entirely -> its own standalone node
                standalone_files.setdefault(imp.module, path)
                edges.append({"from": node_id(imp.module), "to": f_id, "kind": "normal"})

    in_tree_dotted = [dot for dot, dom in domains.items() if dom.in_tree]
    for dotted in in_tree_dotted:
        dom = domains[dotted]
        for f in dom.files:
            f_dotted = dotted_path(repo_root, f)
            process_file_imports(f, f_dotted, dom)
        for imported_dotted in dom.init_internal_edges:
            _, path = resolve_target(repo_root, imported_dotted)
            edges.append({
                "from": file_node_id(repo_root, path),
                "to": file_node_id(repo_root, dom.dir_path / "__init__.py"),
                "kind": "internal",
            })

    # reverse pass: anything ANYWHERE else in the repo that imports from the target or
    # any of its submodules -- "who depends on me", not just "what do I depend on".
    # Only runs for the originally-requested target and its own submodules -- a domain
    # pulled in as a boundary dependency or as an importer does NOT get its own reverse
    # scan, or this would recurse upward without bound (found empirically: templating's
    # reverse-scan finding `utils`, then `utils`'s own reverse-scan finding everything
    # that imports `utils`, cascading through a big chunk of the repo).
    own_dirs = {domains[d].dir_path for d in in_tree_dotted}
    target_set = set(in_tree_dotted)
    for f in ([] if not do_reverse_scan else iter_py_files(repo_root)):
        if any(f == d or d in f.parents for d in own_dirs):
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in f.relative_to(repo_root).parts[:-1]):
            continue
        f_dotted = dotted_path(repo_root, f)
        # only bother registering this file's home domain (a real, possibly expensive
        # expansion) if it actually imports something from our target -- most files in
        # the repo won't, and shouldn't drag their whole unrelated domain in.
        has_match = any(
            imp.module in target_set or any(imp.module.startswith(t + ".") for t in target_set)
            for imp in parse_imports(f, f_dotted)
        )
        if not has_match:
            continue
        home_dir = topmost_domain_dir(repo_root, f)
        if home_dir is not None:
            # An importer whose home domain is an *ancestor* of our own target isn't a
            # real external boundary -- registering it would pull in its whole subtree
            # (via register_external_domain's own find_domains_in_tree), which includes
            # the target itself and any siblings, ballooning this graph back out to the
            # ancestor's scope. That relationship is already implicit (the target is
            # nested inside it), so skip it rather than register it.
            if any(home_dir in od.parents for od in own_dirs):
                continue
            home_dotted = dotted_path(repo_root, home_dir / "__init__.py")
            importer_domain = domains.get(home_dotted) or register_external_domain(
                home_dotted, home_dir / "__init__.py"
            )
        else:
            importer_domain = None
            standalone_files.setdefault(f_dotted, f)
        process_file_imports(f, f_dotted, importer_domain, restrict_to=target_set)

    def dedupe(items):
        seen = set()
        out = []
        for it in items:
            key = tuple(sorted(it.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    edges = dedupe(edges)

    color_order = [d for d, dom in domains.items() if dom.in_tree] + [
        d for d, dom in domains.items() if not dom.in_tree
    ]

    def domain_json(dom: Domain) -> dict:
        return {
            "dotted": dom.dotted,
            "id": node_id(dom.dotted),
            "in_tree": dom.in_tree,
            "files": [
                {
                    "id": file_node_id(repo_root, f),
                    "filename": f.name,
                    # Relative to the domain's own directory, POSIX-separated -- for a
                    # file directly in the domain dir this is just `filename`, but a
                    # grouping subdirectory (e.g. ingest/adapters/_docx_adapter.py, no
                    # __init__.py of its own) needs the extra path segment to actually
                    # be found on disk again from the domain dir alone.
                    "relpath": f.relative_to(dom.dir_path).as_posix(),
                }
                for f in dom.files
            ],
            "functions": dom.functions,
            "types_raw": dom.types_raw,
        }

    return {
        "target": target_dotted,
        "color_order": color_order,
        "domains": [domain_json(domains[d]) for d in color_order],
        "standalone_files": [
            {"dotted": d, "id": node_id(d), "filename": p.name}
            for d, p in sorted(standalone_files.items())
        ],
        "broken": [
            {"dotted": d, "id": node_id(d), "importer": importer}
            for d, importer in sorted(broken.items())
        ],
        "violations": violations,
        "edges": edges,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="module directory to graph, e.g. templating/contract")
    ap.add_argument("--repo-root", required=True, help="the directory local imports resolve against, e.g. be/")
    ap.add_argument("-o", "--output", help="write JSON here (default: stdout)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    target_dir = (repo_root / args.target).resolve()
    result = extract(repo_root, target_dir)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
