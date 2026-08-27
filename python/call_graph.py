#!/usr/bin/env python3
"""Static extraction and rendering of a function call graph: which tracked functions
call which others, same-file or across a module boundary, nested by domain then by
file -- the same visual convention as the dependency graph, just at function
granularity instead of file/domain granularity. Pure `ast` parsing, no type checker, no
execution, no dataflow tracing.

Internal (leading-underscore) functions never become nodes -- a chain of calls through
one or more of them is collapsed into a single edge between the two PUBLIC functions on
either end, carrying the internal functions it passed through as that edge's `via`
list. Only public functions that participate in at least one such edge (direct or
collapsed), AND are called from a different file than the one they're defined in, are
kept -- a function nobody calls, that calls nothing else, or that's only ever called
from within its own file adds no cross-module information to a call graph and is
dropped, the same "don't show what isn't connected to anything" spirit the rest of this
toolset already follows.

Operates on an already-computed extract.extract() result (its `domains` list already
covers the target's own in-tree subtree plus any one-hop external boundary domain).
"""
from __future__ import annotations

import ast
import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from extract import dotted_path, file_node_id, parse_imports, resolve_target, sibling_dotted
from render import PALETTE, build_forest, light_tint
from models import CallDomain, CallEdge, CallGraph, ExtractedGraph, FunctionInfo, TypeHighlight


def _is_internal(qualname: str) -> bool:
    """A function/method is internal if its own name (the last dotted segment -- a
    method's class name doesn't count) starts with an underscore. Dunders like
    __init__ count as internal too; they're rarely a meaningful call target to show."""
    return qualname.rsplit(".", 1)[-1].startswith("_")


IGNORED_ANNOTATION_NAMES = {
    "None", "Any", "object", "type", "bool", "int", "float", "complex", "str",
    "bytes", "bytearray", "list", "dict", "set", "frozenset", "tuple",
    "Optional", "Union", "Callable", "Iterable", "Iterator", "Sequence",
    "Mapping", "MutableMapping", "MutableSequence", "Generic", "TypeVar",
    "ClassVar", "Literal", "Final", "Protocol", "Awaitable", "Coroutine",
    "Generator", "AsyncGenerator", "Self", "NoReturn", "Never", "TypeAlias",
    "ParamSpec", "Annotated", "NamedTuple", "TypedDict",
}


def _annotation_names(expr: ast.AST | None) -> list[str]:
    """Every named-type reference inside an annotation expression -- walks into
    Subscript args (list[X], Union[X, Y]) and PEP 604 unions (X | Y), so a compound
    annotation still yields its real type names, filtered to drop builtins/typing
    constructs that aren't a type this tool tracks."""
    if expr is None:
        return []
    out: list[str] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Subscript):
            walk(node.value)
            walk(node.slice)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                walk(elt)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                walk(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                pass

    walk(expr)
    return [n for n in out if n not in IGNORED_ANNOTATION_NAMES]


def _module_level_type_definitions(file_path: Path) -> dict[str, tuple[str, ast.AST]]:
    """Module-level names that count as a type definition, mapped to ("class", <ClassDef
    node>) or ("alias", <RHS expression>). A type alias is recognized as `Slot = A | B |
    C` or `Slot = Union[A, B, C]` (module-level `Assign` whose value is a union shape),
    `Slot: TypeAlias = ...` (explicit marker, any RHS shape), or the PEP 695 `type Slot =
    ...` statement (Python 3.12+)."""
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError, OSError):
        return {}
    out: dict[str, tuple[str, ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = ("class", node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.annotation, ast.Name) and node.annotation.id == "TypeAlias" and node.value is not None:
                out[node.target.id] = ("alias", node.value)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            is_union = isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr)
            if not is_union and isinstance(value, ast.Subscript):
                base = value.value
                base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                is_union = base_name in ("Union", "Optional")
            if is_union:
                out[node.targets[0].id] = ("alias", value)
        elif isinstance(node, getattr(ast, "TypeAlias", ())) and isinstance(node.name, ast.Name):
            out[node.name.id] = ("alias", node.value)
    return out


def _field_reference_names(class_node: ast.ClassDef) -> list[str]:
    """Every named-type reference in this class's own annotated fields (module-level
    `AnnAssign` items in its body) -- pydantic-model/dataclass style. Only the class's
    own declared attributes; inherited fields and method signatures aren't considered
    -- this is composition (has-a), not inheritance (is-a)."""
    out: list[str] = []
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            out.extend(_annotation_names(item.annotation))
    return out


def resolve_type_definition(
    repo_root: Path, file_dotted: str, file_path: Path, name: str, cache: dict
) -> tuple[str, Path] | None:
    """Chase `name` back to the file that actually defines it as a class or named type
    alias -- following `from x import name` re-export chains. Uses its own cache,
    separate from resolve_function_definition's: a (file_dotted, name) pair unambiguously
    resolves to one real thing, but a cached "not a function" result would wrongly
    short-circuit this lookup for the exact names it exists to resolve, and vice versa."""
    key = (file_dotted, name)
    if key in cache:
        return cache[key]
    cache[key] = None
    if name in _module_level_type_definitions(file_path):
        result = (file_dotted, file_path)
        cache[key] = result
        return result
    for imp in parse_imports(file_path, file_dotted):
        if name not in imp.names:
            continue
        kind, path = resolve_target(repo_root, imp.module)
        imp_module = imp.module
        if kind == "external":
            resolved = sibling_dotted(repo_root, file_path, imp.module)
            if resolved is not None:
                imp_module = resolved
                kind, path = resolve_target(repo_root, resolved)
        if kind in ("file", "package"):
            result = resolve_type_definition(repo_root, imp_module, path, name, cache)
            if result is not None:
                cache[key] = result
                return result
    return None


def _bare_constructor_calls(fn_node: ast.AST) -> list[str]:
    """Every bare `Name(...)` call anywhere in this function's body -- candidates for a
    literal construction of some tracked type, resolved by the caller. Deliberately
    whole-body, not just the return statement: a type is often built and immediately
    nested inside another type's own constructor (`return Category(slots=[Slot(...) for
    row in rows])`), or built up across a loop before being wrapped (`slots.append(Slot(...))`
    ... `return Category(slots=slots)`) -- restricting the search to the return
    expression alone would miss both of these common patterns, and there's no variable
    tracking here to follow a value from an intermediate assignment to its eventual use
    -- whole-body is the cheap approximation that still catches them."""
    names: list[str] = []
    for node in ast.walk(fn_node):
        if node is fn_node or not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return list(dict.fromkeys(names))


@dataclass
class SigFunction:
    qualname: str
    node: ast.AST
    class_name: str | None = None


def _functions_in_file(file_path: Path) -> list[SigFunction]:
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError, OSError):
        return []
    out: list[SigFunction] = []
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(SigFunction(item.name, item))
        elif isinstance(item, ast.ClassDef):
            for sub in item.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(SigFunction(f"{item.name}.{sub.name}", sub, item.name))
    return out


def _module_level_dispatch_tables(file_path: Path) -> dict[str, list[ast.expr]]:
    """Module-level `NAME = {...}` dict literals, mapped to their value expressions --
    used to resolve a dispatch-table call (`fn = TABLE[key]; fn(...)`) as a call to
    every one of the table's possible values. There's no constant-folding here to know
    which key actually wins at a given call site, so this is deliberately conservative:
    all of a dispatch table's values are treated as possible targets of any call
    through a variable assigned from it."""
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError, OSError):
        return {}
    out: dict[str, list[ast.expr]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
        ):
            out[node.targets[0].id] = list(node.value.values)
    return out


def _local_dispatch_vars(fn_node: ast.AST, dispatch_tables: dict[str, list[ast.expr]]) -> dict[str, str]:
    """Local variables assigned as `var = TABLE[...]` where TABLE is a known
    module-level dispatch dict -- `var`'s possible values are exactly the dict's own
    values, so a later `var(...)` call can be resolved through them."""
    found: dict[str, str] = {}
    for node in ast.walk(fn_node):
        if node is fn_node:
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in dispatch_tables
        ):
            found[node.targets[0].id] = node.value.value.id
    return found


def _call_targets(
    fn_node: ast.AST,
    class_name: str | None,
    bare_names: set[str],
    class_methods: dict[str, set[str]],
    dispatch_tables: dict[str, list[ast.expr]],
) -> list[tuple[str, str]]:
    """This function's body's direct calls, as an ordered, deduped (first occurrence
    wins) list of (kind, name) in source order -- so a function that calls several
    things ends up with its outgoing edges in the same order those calls actually
    happen in its body, not whatever order a set would iterate them in. `kind` is
    "same_file" for a bare name call matching a module-level function in this file, or
    a self./cls. call matching a method on the function's own class; "bare" for a plain
    `name(...)` call that isn't a same-file function, left for the caller to chase
    across module boundaries via resolve_function_definition. A call through a local
    variable assigned from a known module-level dispatch dict (`fn = TABLE[key];
    fn(...)`) expands, in the table's own declared order, into one entry per
    resolvable value, all sharing the call site's own position. Purely
    textual/structural otherwise: no import or type tracing, so an attribute call on
    anything other than self/cls (e.g. some_obj.method()) is never resolved -- real
    type info would be needed to know what some_obj even is."""
    local_dispatch = _local_dispatch_vars(fn_node, dispatch_tables)
    # (lineno, col_offset, subindex-within-a-dispatch-expansion, kind, name)
    raw: list[tuple[int, int, int, str, str]] = []
    for node in ast.walk(fn_node):
        if node is fn_node or not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in bare_names:
                raw.append((node.lineno, node.col_offset, 0, "same_file", func.id))
            elif func.id in local_dispatch:
                for i, value in enumerate(dispatch_tables[local_dispatch[func.id]]):
                    if not isinstance(value, ast.Name):
                        continue
                    kind = "same_file" if value.id in bare_names else "bare"
                    raw.append((node.lineno, node.col_offset, i, kind, value.id))
            else:
                raw.append((node.lineno, node.col_offset, 0, "bare", func.id))
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("self", "cls")
            and class_name is not None
            and func.attr in class_methods.get(class_name, set())
        ):
            raw.append((node.lineno, node.col_offset, 0, "same_file", f"{class_name}.{func.attr}"))

    raw.sort(key=lambda r: r[:3])
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for _, _, _, kind, name in raw:
        key = (kind, name)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _module_level_function_names(file_path: Path) -> set[str]:
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError, OSError):
        return set()
    return {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def resolve_function_definition(
    repo_root: Path, file_dotted: str, file_path: Path, name: str, cache: dict
) -> tuple[str, Path] | None:
    """Chase `name` (referenced somewhere in `file_path`) back to the file that
    actually defines it as a module-level function -- following `from x import name`
    re-export chains the same way extract.resolve_symbol_kind does. Deliberately
    narrow: only a bare `name(...)` call reached via `from x import name` is resolved
    -- a module-qualified call (`x.name(...)`) isn't, since parse_imports doesn't track
    import aliases, and neither is a method call on an arbitrary object, since there's
    no type inference here to know what that object is."""
    key = (file_dotted, name)
    if key in cache:
        return cache[key]
    cache[key] = None  # break cycles defensively
    if name in _module_level_function_names(file_path):
        result = (file_dotted, file_path)
        cache[key] = result
        return result
    for imp in parse_imports(file_path, file_dotted):
        if name not in imp.names:
            continue
        kind, path = resolve_target(repo_root, imp.module)
        imp_module = imp.module
        if kind == "external":
            resolved = sibling_dotted(repo_root, file_path, imp.module)
            if resolved is not None:
                imp_module = resolved
                kind, path = resolve_target(repo_root, resolved)
        if kind in ("file", "package"):
            result = resolve_function_definition(repo_root, imp_module, path, name, cache)
            if result is not None:
                cache[key] = result
                return result
    return None


def extract_call_graph(repo_root: Path, extracted: ExtractedGraph) -> CallGraph:
    dom_dirs = [(dom, Path(repo_root, *dom["dotted"].split("."))) for dom in extracted["domains"]]

    all_functions: dict[str, dict] = {}
    # Each function's own AST node plus the file context it was found in -- kept
    # alongside all_functions (not folded into it) since it's only needed for the
    # type-construction scan below, not for anything JSON-serializable.
    function_context: dict[str, tuple[ast.AST, str, Path]] = {}
    for dom, dom_dir in dom_dirs:
        for f in dom["files"]:
            file_path = dom_dir / f["relpath"]
            file_dotted = dotted_path(repo_root, file_path)
            for sig in _functions_in_file(file_path):
                function_id = f'{f["id"]}_fn_{sig.qualname.replace(".", "_")}'
                all_functions[function_id] = {
                    "id": function_id,
                    "qualname": sig.qualname,
                    "domain": dom["dotted"],
                    "file_id": f["id"],
                    "filename": f["filename"],
                }
                function_context[function_id] = (sig.node, file_dotted, file_path)

    # -- Main-type classification -------------------------------------------------
    #
    # A type is "main" -- worth tracking through the call graph at all -- if it's
    # something an external caller of this domain actually interacts with, as
    # opposed to a utility type exported only incidentally (type guards, sentinels,
    # standalone aliases). Two ways in:
    #
    #   1. It's named directly in an exported function's own signature (return type
    #      or a non-primitive param type) -- the caller hands it in or gets it back.
    #   2. It's the root of the composition hierarchy among this domain's exported
    #      types -- zero out-degree (never itself a field of another exported type)
    #      but nonzero in-degree (things ARE composed into it). Catches a pure-types
    #      domain with no exported functions to seed from.
    #
    # From either kind of seed, everything transitively composed INTO it (its own
    # fields, their fields, ...) is main too -- if you get a Category back, you will
    # encounter its Slots and their Columns, even though neither ever appears
    # directly in a function signature or is itself a root.
    #
    # Composition edges point field-type --> containing-type (same "dependency flows
    # up to dependent" convention as everything else here); union/alias membership
    # counts as composition too (a Slot value might really be a FixedSlot). Only
    # considered among types this tool can resolve to an in-tree definition -- a
    # field typed as some external/stdlib class contributes no edge. Self-loops
    # (recursive fields, e.g. Category.subcategories: list[Category]) are dropped
    # rather than counted -- a type composed of itself isn't "contained" by
    # anything new. Non-self cycles between two DIFFERENT types aren't specially
    # unwound (no strongly-connected-component collapsing) -- deliberately deferred
    # until an actual cycle is found to matter in practice; today, a cycle like that
    # just means the DAG-root search finds nothing inside that cluster.
    type_resolve_cache: dict = {}

    all_type_defs: dict[str, dict] = {}
    for dom, dom_dir in dom_dirs:
        for f in dom["files"]:
            file_path = dom_dir / f["relpath"]
            file_dotted = dotted_path(repo_root, file_path)
            for name, (kind, node) in _module_level_type_definitions(file_path).items():
                type_key = f"{file_dotted}.{name}"
                all_type_defs[type_key] = {
                    "kind": kind,
                    "node": node,
                    "file_dotted": file_dotted,
                    "file_path": file_path,
                    "name": name,
                    "domain": dom["dotted"],
                    "file_id": f["id"],
                    "filename": f["filename"],
                }

    composition_edges: set[tuple[str, str]] = set()
    for type_key, td in all_type_defs.items():
        referenced_names = (
            _field_reference_names(td["node"]) if td["kind"] == "class" else _annotation_names(td["node"])
        )
        for name in referenced_names:
            resolved = resolve_type_definition(repo_root, td["file_dotted"], td["file_path"], name, type_resolve_cache)
            if resolved is None:
                continue
            ref_key = f"{resolved[0]}.{name}"
            if ref_key == type_key or ref_key not in all_type_defs:
                continue
            composition_edges.add((ref_key, type_key))

    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    predecessors: dict[str, list[str]] = {}
    successors: dict[str, list[str]] = {}
    for a, b in composition_edges:
        out_degree[a] = out_degree.get(a, 0) + 1
        in_degree[b] = in_degree.get(b, 0) + 1
        predecessors.setdefault(b, []).append(a)
        successors.setdefault(a, []).append(b)

    exported_type_keys: set[str] = set()
    seed_types: set[str] = set()
    for dom, dom_dir in dom_dirs:
        init_entry = next((f for f in dom["files"] if f["filename"] == "__init__.py"), None)
        if init_entry is None:
            continue
        init_path = dom_dir / init_entry["relpath"]

        for name in dom.get("types_raw", []):
            resolved = resolve_type_definition(repo_root, dom["dotted"], init_path, name, type_resolve_cache)
            if resolved is not None:
                exported_type_keys.add(f"{resolved[0]}.{name}")

        for name in dom.get("functions", []):
            resolved = resolve_function_definition(repo_root, dom["dotted"], init_path, name, {})
            if resolved is None:
                continue
            def_dotted, def_path = resolved
            sig = next((s for s in _functions_in_file(def_path) if s.qualname == name), None)
            if sig is None:
                continue
            fn_node = sig.node
            all_args = fn_node.args.posonlyargs + fn_node.args.args + fn_node.args.kwonlyargs
            for ann in [fn_node.returns] + [a.annotation for a in all_args if a.annotation is not None]:
                for ann_name in _annotation_names(ann):
                    resolved_ann = resolve_type_definition(repo_root, def_dotted, def_path, ann_name, type_resolve_cache)
                    if resolved_ann is not None:
                        seed_types.add(f"{resolved_ann[0]}.{ann_name}")

    root_types = {
        key for key in exported_type_keys
        if out_degree.get(key, 0) == 0 and in_degree.get(key, 0) > 0
    }

    def closure_from(seed: str) -> set[str]:
        seen: set[str] = set()
        stack = [seed]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(predecessors.get(cur, []))
        return seen

    main_type_keys: set[str] = set()
    for seed in seed_types | root_types:
        main_type_keys |= closure_from(seed)

    # Ordered by caller: each caller's callees are collected in the same order those
    # calls actually happen in its body (see _call_targets), deduped so a callee called
    # more than once keeps only its first position -- so a function with several
    # outgoing calls renders/lays out in that same, real call order downstream.
    call_order: dict[str, list[str]] = {}
    resolve_cache: dict = {}

    def record_call(caller_id: str, callee_id: str) -> None:
        callees = call_order.setdefault(caller_id, [])
        if callee_id not in callees:
            callees.append(callee_id)

    for dom, dom_dir in dom_dirs:
        for f in dom["files"]:
            file_path = dom_dir / f["relpath"]
            file_dotted = dotted_path(repo_root, file_path)
            sigs = _functions_in_file(file_path)
            function_id_by_qualname = {
                s.qualname: f'{f["id"]}_fn_{s.qualname.replace(".", "_")}' for s in sigs
            }
            bare_names = {s.qualname for s in sigs if "." not in s.qualname}
            class_methods: dict[str, set[str]] = {}
            for s in sigs:
                if "." in s.qualname:
                    cls_name, _, meth_name = s.qualname.partition(".")
                    class_methods.setdefault(cls_name, set()).add(meth_name)
            dispatch_tables = _module_level_dispatch_tables(file_path)

            for sig in sigs:
                caller_id = function_id_by_qualname[sig.qualname]
                for kind, name in _call_targets(
                    sig.node, sig.class_name, bare_names, class_methods, dispatch_tables
                ):
                    if kind == "same_file":
                        if name == sig.qualname:
                            continue
                        record_call(caller_id, function_id_by_qualname[name])
                        continue
                    resolved = resolve_function_definition(
                        repo_root, file_dotted, file_path, name, resolve_cache
                    )
                    if resolved is None:
                        continue
                    _, callee_path = resolved
                    callee_id = f"{file_node_id(repo_root, callee_path)}_fn_{name}"
                    if callee_id == caller_id or callee_id not in all_functions:
                        continue
                    record_call(caller_id, callee_id)

    # Construction pseudo-edges: for every function whose body literally calls a main
    # type's own constructor (anywhere -- see _bare_constructor_calls), record a call
    # into a synthetic "type:<type_key>" node, injected into the same adjacency used
    # for real calls above. This lets the internal-function collapsing below (walk())
    # attribute a construction found inside a private helper up to whichever public
    # function actually reaches it, exactly the way it already attributes ordinary
    # calls -- no separate mechanism needed.
    for function_id, (fn_node, file_dotted, file_path) in function_context.items():
        for name in _bare_constructor_calls(fn_node):
            resolved = resolve_type_definition(repo_root, file_dotted, file_path, name, type_resolve_cache)
            if resolved is None:
                continue
            type_key = f"{resolved[0]}.{name}"
            if type_key in main_type_keys:
                record_call(function_id, f"type:{type_key}")

    adjacency = call_order

    public_ids = {fid for fid, fn in all_functions.items() if not _is_internal(fn["qualname"])}
    internal_ids = set(all_functions) - public_ids

    # Collapse chains through internal (leading-underscore) functions: a public
    # function's call into one or more internal functions in a row folds into a single
    # edge to whichever public function the chain eventually reaches, carrying the
    # internal functions it passed through as `via`. An internal function never becomes
    # a node of its own -- it only survives as a name listed on the edge(s) it
    # mediates. A chain that never reaches another public function (a true dead end)
    # produces no edge at all, same as any other unconnected function. A "type:<key>"
    # pseudo node (see above) is terminal the same way a public function is -- it never
    # has outgoing edges of its own, so it's never a candidate to recurse into.
    collapsed: dict[tuple[str, str], set[str]] = {}

    def walk(node: str, path_internals: frozenset[str], origin: str) -> None:
        for callee in adjacency.get(node, []):
            if callee == origin:
                continue  # a chain looping back to its own origin isn't a real edge
            if callee in public_ids or callee.startswith("type:"):
                collapsed.setdefault((origin, callee), set()).update(path_internals)
            elif callee in internal_ids and callee not in path_internals:
                walk(callee, path_internals | {callee}, origin)

    for p in sorted(public_ids):
        walk(p, frozenset(), p)

    function_calls = {(a, b): mediators for (a, b), mediators in collapsed.items() if not b.startswith("type:")}
    construct_edges = {(a, b): mediators for (a, b), mediators in collapsed.items() if b.startswith("type:")}
    construct_touches: dict[str, set[str]] = {}
    for a, b in construct_edges:
        construct_touches.setdefault(a, set()).add(b[len("type:"):])

    # Signature touches: does a function's own return/param annotations directly name
    # a main type? Purely local per-function facts -- unlike construction (a body-level
    # detail that can be buried in a private helper), a function's own signature is
    # already right there on whichever node represents it, public or not; no
    # internal-collapsing needed, since what matters is what the INCLUDED function's
    # own annotations say, not what some uncredited helper's annotations say.
    return_touches: dict[str, set[str]] = {}
    accept_touches: dict[str, set[str]] = {}
    for function_id, (fn_node, file_dotted, file_path) in function_context.items():
        for ret_name in _annotation_names(fn_node.returns):
            resolved = resolve_type_definition(repo_root, file_dotted, file_path, ret_name, type_resolve_cache)
            if resolved is not None:
                type_key = f"{resolved[0]}.{ret_name}"
                if type_key in main_type_keys:
                    return_touches.setdefault(function_id, set()).add(type_key)
        all_args = fn_node.args.posonlyargs + fn_node.args.args + fn_node.args.kwonlyargs
        for arg in all_args:
            for arg_name in _annotation_names(arg.annotation):
                resolved = resolve_type_definition(repo_root, file_dotted, file_path, arg_name, type_resolve_cache)
                if resolved is not None:
                    type_key = f"{resolved[0]}.{arg_name}"
                    if type_key in main_type_keys:
                        accept_touches.setdefault(function_id, set()).add(type_key)

    # A public function gets a node only if it's called from a DIFFERENT file than the
    # one it's defined in -- "called from outside the module" in the literal Python
    # sense (a .py file is a module). A function called only from within its own file
    # doesn't qualify: it's still fully internal to that file's own story, same spirit
    # as collapsing away leading-underscore functions, just for the "nobody outside
    # this file needed to know this exists" case instead of the naming convention.
    # Both ends of a qualifying cross-file edge are included, not just the callee --
    # the caller has to be a node too, or the very edge that justifies the callee's
    # inclusion would have nothing to attach to. Touching a main type (constructing,
    # returning, or accepting one) earns a node on its own, independent of the
    # cross-file rule.
    used_ids: set[str] = set()
    for a, b in function_calls:
        if all_functions[a]["file_id"] != all_functions[b]["file_id"]:
            used_ids.add(a)
            used_ids.add(b)
    for fid in set(construct_touches) | set(return_touches) | set(accept_touches):
        used_ids.add(fid)
    functions = {fid: all_functions[fid] for fid in used_ids}

    # Per-main-type highlight sets, precomputed here rather than shipped as raw graph
    # structure for the frontend to re-derive: for each main type T, every included
    # function that touches T directly (constructs/returns/accepts it), PLUS every
    # function that only touches something T is transitively a component of (T is a
    # field of U, U of V, ... -- walking `successors`, i.e. "what T flows into") gets
    # a lower-priority "component" tag instead. A function touching T both directly
    # and via some ancestor keeps the stronger, more specific tag.
    KIND_PRIORITY = {"constructs": 0, "returns": 1, "accepts": 2, "component": 3}

    def ancestors_of(key: str) -> set[str]:
        seen: set[str] = set()
        stack = [key]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(successors.get(cur, []))
        return seen

    main_types_out = []
    for type_key in sorted(main_type_keys):
        relevant = ancestors_of(type_key) - {type_key}
        highlights: dict[str, str] = {}

        def note(fid: str, kind: str) -> None:
            if fid not in used_ids:
                return
            current = highlights.get(fid)
            if current is None or KIND_PRIORITY[kind] < KIND_PRIORITY[current]:
                highlights[fid] = kind

        for touches, kind in ((construct_touches, "constructs"), (return_touches, "returns"), (accept_touches, "accepts")):
            for fid, keys in touches.items():
                if type_key in keys:
                    note(fid, kind)
                elif keys & relevant:
                    note(fid, "component")

        # produces/consumes: unlike `highlights` (one priority-picked kind per
        # function, for node border styling), these are plain membership sets used to
        # tag the two edges between a caller and callee -- the "returns" edge (value
        # flows callee -> caller) is relevant to T if the callee PRODUCES T (or
        # something T is a component of); the "calls" edge (caller -> callee) is
        # relevant if the callee CONSUMES T the same way. A function that both accepts
        # and returns T ends up in both sets -- `highlights` would have collapsed that
        # to a single kind, which is fine for a node's one border color but would
        # silently drop one of the two edge relationships if reused here.
        relevant_incl = relevant | {type_key}
        produces = sorted(
            {fid for fid, keys in construct_touches.items() if fid in used_ids and keys & relevant_incl}
            | {fid for fid, keys in return_touches.items() if fid in used_ids and keys & relevant_incl}
        )
        consumes = sorted(
            {fid for fid, keys in accept_touches.items() if fid in used_ids and keys & relevant_incl}
        )

        td = all_type_defs[type_key]
        main_types_out.append({
            "type_key": type_key, "name": td["name"], "highlights": highlights,
            "produces": produces, "consumes": consumes,
        })

    return {
        "target": extracted["target"],
        "domains": [
            {"dotted": dom["dotted"], "id": dom["id"], "in_tree": dom["in_tree"]}
            for dom, _ in dom_dirs
        ],
        "functions": [functions[k] for k in sorted(functions)],
        "calls": [
            {
                "from": a,
                "to": b,
                "via": sorted({all_functions[m]["qualname"] for m in mediators}),
            }
            for (a, b), mediators in function_calls.items()
            if a in used_ids and b in used_ids
        ],
        "main_types": main_types_out,
    }


def render_call_graph(call_graph: CallGraph) -> str:
    functions = {f["id"]: f for f in call_graph["functions"]}

    file_entries: dict[tuple[str, str], dict] = {}

    def file_slot(domain_dotted: str, file_id: str, filename: str) -> dict:
        return file_entries.setdefault(
            (domain_dotted, file_id), {"filename": filename, "function_ids": []}
        )

    for fn in call_graph["functions"]:
        file_slot(fn["domain"], fn["file_id"], fn["filename"])["function_ids"].append(fn["id"])

    files_by_domain: dict[str, list[tuple[str, dict]]] = {}
    for (domain_dotted, file_id), entry in file_entries.items():
        files_by_domain.setdefault(domain_dotted, []).append((file_id, entry))
    for lst in files_by_domain.values():
        lst.sort(key=lambda pair: pair[1]["filename"])

    domains_by_dotted = {d["dotted"]: d for d in call_graph["domains"]}
    touched_domains = [domains_by_dotted[d] for d in sorted(files_by_domain)]

    lines = ["---", "config:", "  layout: elk", "---", "graph BT"]

    def escape(text: str) -> str:
        return text.replace('"', "&quot;")

    def emit_domain(dom: dict, children: dict[str, list[dict]], indent: str) -> list[str]:
        out = [f'{indent}subgraph {dom["id"]}["{dom["dotted"]}"]']
        pad = indent + "    "
        for file_id, entry in files_by_domain.get(dom["dotted"], []):
            out.append(f'{pad}subgraph {file_id}["{entry["filename"]}"]')
            fpad = pad + "    "
            for fn_id in entry["function_ids"]:
                fn = functions[fn_id]
                out.append(f'{fpad}{fn_id}(["{escape(fn["qualname"])}"])')
            out.append(f"{pad}end")
        for child in children.get(dom["id"], []):
            out.append("")
            out.extend(emit_domain(child, children, pad))
        out.append(f"{indent}end")
        return out

    roots, children = build_forest(touched_domains)
    for root in roots:
        lines.extend(emit_domain(root, children, "    "))
        lines.append("")

    # One bidirectional edge per (caller, callee) pair -- <--> puts an arrowhead on
    # both ends, standing in for both "caller calls callee" and "callee's value
    # returns to caller" without needing two separate directed edges (tried once, both
    # as two "-->" edges and with an ELK cycle-breaking config to tame the 2-cycle
    # that created between every pair -- more machinery than the relationship needs;
    # see this function's git history if that's ever worth revisiting).
    #
    # call_graph["calls"] is in true source call order (see extract_call_graph) -- kept
    # that way since it's the semantically correct data. Emitted here in reverse,
    # because mermaid/ELK's BT layout visually places a caller's multiple targets in
    # the opposite order from the order their edges were declared; reversing at this
    # rendering step (not in the data itself) compensates for that specific layout
    # quirk without making the underlying call_graph.json order lie about the source.
    for c in reversed(call_graph["calls"]):
        via = c.get("via") or []
        label = "calls / returns via " + ", ".join(via) if via else "calls / returns"
        lines.append(f'    {c["to"]} <-->|{label}| {c["from"]}')
    lines.append("")

    palette_for_domain = {
        dom["id"]: PALETTE[i % len(PALETTE)] for i, dom in enumerate(touched_domains)
    }
    for dom in touched_domains:
        _, fill, stroke, color = palette_for_domain[dom["id"]]
        lines.append(
            f'    classDef {dom["id"]}Domain fill:{fill},stroke:{stroke},color:{color},stroke-width:1px;'
        )
    lines.append("    classDef fileGroup fill:none,stroke:#999999,stroke-width:1px,stroke-dasharray: 2 2;")
    lines.append("")

    for dom in touched_domains:
        lines.append(f'    class {dom["id"]} {dom["id"]}Domain')
    file_ids_all = [fid for _, fid in file_entries]
    if file_ids_all:
        lines.append(f'    class {",".join(file_ids_all)} fileGroup')
    lines.append("")

    for dom in touched_domains:
        _, fill, _, _ = palette_for_domain[dom["id"]]
        tint = light_tint(fill)
        lines.append(f'    style {dom["id"]} fill:{tint},stroke:{fill},stroke-width:2px;')
    lines.append("")

    return "\n".join(lines) + "\n"


def render_call_graph_html(call_graph: CallGraph) -> str:
    """mermaid + ELK layout (confirmed to render fine from a plain file://-opened page
    -- see render_call_graph_html's prior revision for that test) plus a legend of
    main types (extract_call_graph's main-type classification). Each main type gets
    its own stable color (cycling PALETTE, same convention as domain coloring
    elsewhere in this toolset) -- shown as the legend row's own dot, and used
    uniformly on every node that touches that type once selected (constructs/returns/
    accepts it directly, or only touches something it's transitively a component of --
    see extract_call_graph's `highlights`). One color per type, not per kind: an
    earlier revision colored by kind instead and quietly picked "constructs" as the
    legend dot for nearly every type (whole-body construction scanning means a single
    function assembling a nested literal, e.g. `Category(slots=[Slot(...) for ...])`,
    legitimately constructs BOTH types at once, so "constructs" ends up present on
    almost every type's highlight set) and colored a single selection's nodes
    inconsistently (each touching function painted by ITS OWN kind, not the type's).
    Neither reads as "this is type X, everywhere" at a glance, which is the actual
    goal here -- so kind is no longer a color dimension at all.

    Nothing dims when a type is selected -- the rest of the graph stays fully
    readable as context around whatever's highlighted. Node lookup matches the
    rendered SVG's `.node` elements by an `id*=` (contains) selector against our own
    fn_id, rather than assuming a specific `flowchart-<id>-<n>` id format -- that
    exact wrapping has drifted across mermaid versions, whereas embedding the
    original id as a substring somewhere in the generated one has been stable;
    contains-matching is safe here since a complete fn_id is only ever a substring of
    another when they're the same node.

    Edge direction (parameter vs. return) is drawn as a hand-rolled overlay, not by
    touching mermaid's own edge paths: the base graph keeps its single bidirectional
    `<-->` edge per caller/callee pair (see render_call_graph's docstring for why --
    splitting that into two directed edges was tried once and reverted, it creates a
    2-cycle at every pair that fights ELK's layout). Mermaid's rendered edge/marker
    elements are exactly the kind of version-drifting internals the node-lookup
    comment above already had to work around once; reaching for them a second time
    for something as fiddly as per-direction arrowheads isn't worth it. Instead, on
    selection, a small SVG `<g>` is appended on top of the rendered diagram with one
    arrow per relevant produces/consumes edge (extract_call_graph's `produces` =
    callee's return flows back to caller, `consumes` = caller's argument flows into
    callee), positioned via `el.getBoundingClientRect()` mapped back into svgRoot's
    coordinate system through `svgRoot.getScreenCTM().inverse()` -- real rendered
    screen pixels, not `getBBox()`/`getCTM()`'s "clean" SVG geometry model, which was
    tried first and placed arrows nowhere near the actual nodes (mermaid node labels
    render through `<foreignObject>` HTML content, which that geometry model doesn't
    reliably account for). getBoundingClientRect is immune to that: it reports
    wherever the content actually painted, regardless of what mix of transforms or
    foreignObject reflow produced it. A pair where the
    type flows both ways gets two arrows, offset perpendicular to the line so they
    don't overlap; direction is shown by arrowhead placement alone, same single color
    as the type's node highlight -- no second color axis, consistent with why kind
    stopped being a color dimension for nodes (see above).
    """
    mermaid_source = html.escape(render_call_graph(call_graph))
    target = html.escape(call_graph.get("target", "call graph"))

    sorted_types = sorted(call_graph.get("main_types", []), key=lambda t: t["name"])
    main_types = []
    for i, t in enumerate(sorted_types):
        _, fill, stroke, text_color = PALETTE[i % len(PALETTE)]
        main_types.append({
            "type_key": t["type_key"], "name": t["name"], "highlights": t["highlights"],
            "produces": t.get("produces", []), "consumes": t.get("consumes", []),
            "fill": fill, "stroke": stroke, "text_color": text_color,
        })
    calls_min = [{"from": c["from"], "to": c["to"]} for c in call_graph.get("calls", [])]

    template = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__ -- call graph</title>
<style>
  html, body { margin: 0; height: 100%; font-family: -apple-system, Segoe UI, sans-serif; }
  #app { display: flex; height: 100%; }
  #legend { width: 260px; flex: none; overflow-y: auto; border-right: 1px solid #ccc; padding: 12px; box-sizing: border-box; }
  #legend h2 { font-size: 14px; margin: 0 0 8px 0; }
  #legend .clear-btn { display: block; width: 100%; margin-bottom: 10px; padding: 6px; cursor: pointer; }
  #legend .type-row { padding: 6px 8px; cursor: pointer; border-radius: 4px; font-size: 13px; }
  #legend .type-row:hover { background: #eee; }
  #legend .type-row.selected { background: #dde; font-weight: 600; }
  .type-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  #diagram { flex: 1; overflow: hidden; padding: 12px; }
</style>
</head>
<body>
<div id="app">
  <div id="legend">
    <h2>__TITLE__</h2>
    <button class="clear-btn" onclick="clearHighlight()">Clear selection</button>
    <div id="type-list"></div>
  </div>
  <div id="diagram"></div>
</div>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.esm.min.mjs";
  import elkLayouts from "https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0.2.3/dist/mermaid-layout-elk.esm.min.mjs";
  mermaid.registerLayoutLoaders(elkLayouts);
  mermaid.initialize({ startOnLoad: false, theme: "default" });

  const mainTypes = __MAIN_TYPES_JSON__;
  const calls = __CALLS_JSON__;
  const mermaidSource = document.getElementById('mermaid-source').textContent;

  const diagram = document.getElementById('diagram');
  const { svg, bindFunctions } = await mermaid.render('graph-svg', mermaidSource);
  diagram.innerHTML = svg;
  if (bindFunctions) bindFunctions(diagram);
  const svgRoot = diagram.querySelector('svg');

  // Pan and zoom via viewBox manipulation. mermaid injects inline max-width/height
  // on the SVG; override them so the SVG fills the container, then shift/scale the
  // viewBox on wheel (zoom toward cursor) and mousedown+drag (pan).
  svgRoot.style.maxWidth = 'none';
  svgRoot.style.width = '100%';
  svgRoot.style.height = '100%';
  svgRoot.style.display = 'block';
  svgRoot.style.cursor = 'grab';

  const _vb = (svgRoot.getAttribute('viewBox') || '').trim().split(/[\\s,]+/).map(Number);
  let vbX = _vb[0] || 0, vbY = _vb[1] || 0;
  let vbW = _vb[2] || parseFloat(svgRoot.getAttribute('width') || '800');
  let vbH = _vb[3] || parseFloat(svgRoot.getAttribute('height') || '600');
  function applyVB() { svgRoot.setAttribute('viewBox', `${vbX} ${vbY} ${vbW} ${vbH}`); }

  diagram.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = svgRoot.getBoundingClientRect();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const mx = vbX + (e.clientX - rect.left) / rect.width * vbW;
    const my = vbY + (e.clientY - rect.top) / rect.height * vbH;
    vbW *= factor; vbH *= factor;
    vbX = mx - (e.clientX - rect.left) / rect.width * vbW;
    vbY = my - (e.clientY - rect.top) / rect.height * vbH;
    applyVB();
  }, { passive: false });

  let panning = false, panCX, panCY, panVbX, panVbY;
  svgRoot.addEventListener('mousedown', e => {
    panning = true; panCX = e.clientX; panCY = e.clientY; panVbX = vbX; panVbY = vbY;
    svgRoot.style.cursor = 'grabbing'; e.preventDefault();
  });
  window.addEventListener('mousemove', e => {
    if (!panning) return;
    const rect = svgRoot.getBoundingClientRect();
    vbX = panVbX - (e.clientX - panCX) / rect.width * vbW;
    vbY = panVbY - (e.clientY - panCY) / rect.height * vbH;
    applyVB();
  });
  window.addEventListener('mouseup', () => {
    if (panning) { panning = false; svgRoot.style.cursor = 'grab'; }
  });

  function findNode(id) {
    // See this function's module docstring: contains-match, not a specific
    // flowchart-<id>-<n> format assumption.
    return svgRoot.querySelector('.node[id*="' + CSS.escape(id) + '"]');
  }

  // Shape/text tag names inside a mermaid node group aren't something this module
  // can rely on across versions (rect for a plain box, but a stadium/rounded shape
  // like ours can render as a path instead) -- rather than guess and chase a CSS
  // selector that quietly matches nothing, this queries broadly for anything
  // shape-like or text-like and sets the color via inline style with !important,
  // which beats mermaid's own embedded <style> (inserted into the SVG, appearing
  // AFTER this page's own stylesheet in the DOM, so a same-specificity !important
  // rule there would otherwise win the cascade over an external stylesheet rule).
  const SHAPE_SELECTOR = 'rect, polygon, circle, ellipse, path';
  const TEXT_SELECTOR = '.nodeLabel, .nodeLabel *, tspan, text';

  function paintNode(el, entry) {
    el.querySelectorAll(SHAPE_SELECTOR).forEach(shape => {
      shape.style.setProperty('fill', entry.fill, 'important');
      shape.style.setProperty('stroke', entry.stroke, 'important');
    });
    el.querySelectorAll(TEXT_SELECTOR).forEach(t => {
      t.style.setProperty('color', entry.text_color, 'important');
      t.style.setProperty('fill', entry.text_color, 'important');
    });
  }

  function unpaintNode(el) {
    el.querySelectorAll(SHAPE_SELECTOR).forEach(shape => {
      shape.style.removeProperty('fill');
      shape.style.removeProperty('stroke');
    });
    el.querySelectorAll(TEXT_SELECTOR).forEach(t => {
      t.style.removeProperty('color');
      t.style.removeProperty('fill');
    });
  }

  // Edge direction (parameter vs. return) is shown by directly restyling mermaid's
  // own rendered edge <path>, not by drawing new geometry on top -- two earlier
  // attempts (straight overlay lines, then hand-drawn orthogonal elbows guessing at
  // ELK's routing) both looked visibly foreign next to the real edges. Confirmed by
  // rendering a sample graph and inspecting the live DOM (see git history / session
  // notes) that mermaid gives every edge a stable, matchable id and already carries
  // exactly what's needed: `render_call_graph` emits each call as `{c.to} <-->
  // |label| {c.from}` (reversed order, see that function's own docstring), and
  // mermaid renders that as a <path id="...-L_{c.to}_{c.from}_0" ...> whose `d` is
  // ELK's real computed route, with SEPARATE marker-start (arrowhead at the {c.to}
  // end, i.e. the callee) and marker-end (arrowhead at the {c.from} end, i.e. the
  // caller) attributes already on it. So: asParam (consumes) -> show marker-start
  // (arrow into callee); asReturn (produces) -> show marker-end (arrow into caller);
  // both -> show both, matching the default bidirectional look, just recolored. This
  // reuses the actual route pixel-for-pixel and needs no node-position math at all.
  let highlightedEdges = []; // {el, markerStart, markerEnd} snapshots, for clearHighlight to restore
  const coloredMarkerCache = new Map(); // "baseMarkerId|color" -> cloned marker id

  function findEdgePath(callFrom, callTo) {
    return svgRoot.querySelector('path[id*="L_' + CSS.escape(callTo) + '_' + CSS.escape(callFrom) + '_"]');
  }

  function markerRefId(attrValue) {
    if (!attrValue) return null;
    const m = attrValue.match(/url\\(["']?#([^"')]+)["']?\\)/);
    return m ? m[1] : null;
  }

  // mermaid's default arrowhead markers are shared by every edge (colored via a CSS
  // class rule), so they can't be recolored in place without recoloring every edge's
  // arrowhead. Instead, clone the specific marker this edge already references and
  // recolor only the clone, cached by (base marker, color) since many edges share the
  // same base marker and same type color.
  function ensureColoredMarker(baseId, color) {
    if (!baseId) return null;
    const cacheKey = baseId + '|' + color;
    if (coloredMarkerCache.has(cacheKey)) return coloredMarkerCache.get(cacheKey);
    const base = document.getElementById(baseId);
    if (!base) return null;
    const clone = base.cloneNode(true);
    const cloneId = 'hlmarker-' + coloredMarkerCache.size + '-' + color.replace('#', '');
    clone.setAttribute('id', cloneId);
    clone.querySelectorAll('path, circle, polygon').forEach(shape => {
      shape.style.setProperty('fill', color, 'important');
      shape.style.setProperty('stroke', color, 'important');
    });
    base.parentNode.appendChild(clone);
    coloredMarkerCache.set(cacheKey, cloneId);
    return cloneId;
  }

  function styleTypeFlowEdges(entry) {
    const produces = new Set(entry.produces);
    const consumes = new Set(entry.consumes);
    for (const c of calls) {
      const asParam = consumes.has(c.to);   // caller's argument flows into callee
      const asReturn = produces.has(c.to);  // callee's return flows back to caller
      if (!asParam && !asReturn) continue;
      const edgeEl = findEdgePath(c.from, c.to);
      if (!edgeEl) continue;

      const origMarkerStart = edgeEl.getAttribute('marker-start');
      const origMarkerEnd = edgeEl.getAttribute('marker-end');
      highlightedEdges.push({ el: edgeEl, markerStart: origMarkerStart, markerEnd: origMarkerEnd });

      edgeEl.classList.add('hl-edge');
      edgeEl.style.setProperty('stroke', entry.fill, 'important');
      edgeEl.style.setProperty('stroke-width', '3px', 'important');

      if (asParam) {
        const id = ensureColoredMarker(markerRefId(origMarkerStart), entry.fill);
        edgeEl.setAttribute('marker-start', id ? 'url(#' + id + ')' : origMarkerStart || 'none');
      } else {
        edgeEl.setAttribute('marker-start', 'none');
      }
      if (asReturn) {
        const id = ensureColoredMarker(markerRefId(origMarkerEnd), entry.fill);
        edgeEl.setAttribute('marker-end', id ? 'url(#' + id + ')' : origMarkerEnd || 'none');
      } else {
        edgeEl.setAttribute('marker-end', 'none');
      }
    }
  }

  // Coloring, not dimming: a selected type colors every node that touches it, all in
  // that ONE type's own color (see this function's docstring for why kind used to be
  // a second color dimension and isn't anymore), and leaves everything else at its
  // normal appearance -- the rest of the graph stays fully readable as context.
  function clearHighlight() {
    svgRoot.querySelectorAll('.node.hl').forEach(el => {
      unpaintNode(el);
      el.classList.remove('hl');
    });
    highlightedEdges.forEach(({ el, markerStart, markerEnd }) => {
      el.classList.remove('hl-edge');
      el.style.removeProperty('stroke');
      el.style.removeProperty('stroke-width');
      if (markerStart) el.setAttribute('marker-start', markerStart); else el.removeAttribute('marker-start');
      if (markerEnd) el.setAttribute('marker-end', markerEnd); else el.removeAttribute('marker-end');
    });
    highlightedEdges = [];
    document.querySelectorAll('#type-list .type-row').forEach(el => el.classList.remove('selected'));
  }
  window.clearHighlight = clearHighlight;

  function highlightType(typeKey, rowEl) {
    clearHighlight();
    const entry = mainTypes.find(t => t.type_key === typeKey);
    if (!entry) return;
    rowEl.classList.add('selected');
    for (const fid of Object.keys(entry.highlights)) {
      const el = findNode(fid);
      if (!el) continue;
      el.classList.add('hl');
      paintNode(el, entry);
    }
    styleTypeFlowEdges(entry);
  }

  const listEl = document.getElementById('type-list');
  for (const t of mainTypes) {
    const row = document.createElement('div');
    row.className = 'type-row';
    const dot = document.createElement('span');
    dot.className = 'type-dot';
    dot.style.background = t.fill;
    row.appendChild(dot);
    row.appendChild(document.createTextNode(t.name));
    row.onclick = () => highlightType(t.type_key, row);
    listEl.appendChild(row);
  }
</script>
<pre id="mermaid-source" style="display:none">__MERMAID_SOURCE__</pre>
</body>
</html>
"""
    template = template.replace("__TITLE__", target)
    template = template.replace("__MERMAID_SOURCE__", mermaid_source)
    template = template.replace("__MAIN_TYPES_JSON__", json.dumps(main_types))
    template = template.replace("__CALLS_JSON__", json.dumps(calls_min))
    return template
