#!/usr/bin/env python3
"""Static extraction and rendering of a function call graph: which tracked functions
call which others, same-file or across a module boundary, nested by domain then by
file -- the same visual convention as the dependency graph, just at function
granularity instead of file/domain granularity. Pure `ast` parsing, no type checker, no
execution, no dataflow/type tracing beyond the one targeted exception noted below.

Internal (leading-underscore) functions never become nodes -- a chain of calls through
one or more of them is collapsed into a single edge between the two PUBLIC functions on
either end, carrying the internal functions it passed through as that edge's `via`
list. Only public functions that participate in at least one such edge (direct or
collapsed) are kept -- a function nobody calls and that calls nothing else adds no
information to a call graph and is dropped, the same "don't show what isn't connected
to anything" spirit the rest of this toolset already follows.

Type nodes: a function that resolves a named type in its own return annotation is a
candidate to get a type node, function --> type_node -- but only the FIRST such
function in a chain (closest to the leaves) actually gets one: if it also calls
something, directly or transitively, that emits the same type, it's just relaying a
value whose origin is further down (ingest_document also returns IngestedDocument, but
it calls ingest_docx, which is where one is actually built -- ingest_docx gets the
node, not ingest_document). One node instance per true first emitter -- never shared
globally, so unrelated chains never converge on one hub node -- but every instance of
the same type shares one color (cycling the same PALETTE used for domains), so a
recurring type is still recognizable by color alone. A non-first function that also
emits the same type gets no node, but its own edge to its caller is still colored for
it, so the chain reads as one continuous thread: ingest_docx -> IngestedDocument ->
ingest_document -> (colored, no node) -> ingest_documents.

Operates on an already-computed extract.extract() result (its `domains` list already
covers the target's own in-tree subtree plus any one-hop external boundary domain).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from extract import dotted_path, file_node_id, parse_imports, resolve_target
from render import PALETTE, build_forest, light_tint


def _domain_for_path(dom_dirs: list[tuple[dict, Path]], path: Path) -> dict | None:
    """Which tracked domain a path lives under -- the deepest (most specific) matching
    domain directory, in case domains ever nest."""
    best = None
    best_depth = -1
    for dom, dom_dir in dom_dirs:
        try:
            path.relative_to(dom_dir)
        except ValueError:
            continue
        depth = len(dom_dir.parts)
        if depth > best_depth:
            best = dom
            best_depth = depth
    return best


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
        if kind in ("file", "package"):
            result = resolve_type_definition(repo_root, imp.module, path, name, cache)
            if result is not None:
                cache[key] = result
                return result
    return None


@dataclass
class SigFunction:
    qualname: str
    node: ast.AST
    class_name: str | None = None
    params: list[str] = field(default_factory=list)
    param_annotations: list[ast.AST] = field(default_factory=list)
    return_annotation: ast.AST | None = None


def _param_list(node: ast.FunctionDef | ast.AsyncFunctionDef, skip_first: bool) -> tuple[list[str], list[ast.AST]]:
    """`name: annotation` for each parameter -- self/cls omitted for a method, since
    it's implicit. Positional-only, regular, and keyword-only params are all included
    (in that order, undifferentiated -- this is a diagram label, not a real call
    signature); *args/**kwargs are skipped as visual noise unless something downstream
    ever needs them. Also returns the raw annotation expressions (for type resolution),
    parallel to but separate from the rendered text."""
    all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    if skip_first and all_args and all_args[0].arg in ("self", "cls"):
        all_args = all_args[1:]
    parts = []
    annotations = []
    for a in all_args:
        text = a.arg
        if a.annotation is not None:
            text += f": {ast.unparse(a.annotation)}"
            annotations.append(a.annotation)
        parts.append(text)
    return parts, annotations


def _functions_in_file(file_path: Path) -> list[SigFunction]:
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except (UnicodeDecodeError, SyntaxError, OSError):
        return []
    out: list[SigFunction] = []
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params, param_annotations = _param_list(item, skip_first=False)
            out.append(SigFunction(item.name, item, None, params, param_annotations, item.returns))
        elif isinstance(item, ast.ClassDef):
            for sub in item.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params, param_annotations = _param_list(sub, skip_first=True)
                    out.append(
                        SigFunction(
                            f"{item.name}.{sub.name}", sub, item.name, params, param_annotations, sub.returns
                        )
                    )
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
        if kind in ("file", "package"):
            result = resolve_function_definition(repo_root, imp.module, path, name, cache)
            if result is not None:
                cache[key] = result
                return result
    return None


def extract_call_graph(repo_root: Path, extracted: dict) -> dict:
    dom_dirs = [(dom, Path(repo_root, *dom["dotted"].split("."))) for dom in extracted["domains"]]

    all_functions: dict[str, dict] = {}
    type_resolve_cache: dict = {}
    # Where a type_key is actually DEFINED (as opposed to wherever it's emitted from or
    # re-exported through) -- populated by every successful type resolution below, used
    # later to decide which types are candidates to collapse into one shared group node.
    type_defining_file: dict[str, dict] = {}

    def record_type_location(type_key: str, path: Path) -> None:
        if type_key in type_defining_file:
            return
        dom = _domain_for_path(dom_dirs, path)
        if dom is None:
            return  # defining file isn't under any tracked domain -- not groupable
        type_defining_file[type_key] = {
            "domain": dom["dotted"],
            "file_id": file_node_id(repo_root, path),
            "filename": path.name,
        }

    for dom, dom_dir in dom_dirs:
        for f in dom["files"]:
            file_path = dom_dir / f["relpath"]
            file_dotted = dotted_path(repo_root, file_path)
            for sig in _functions_in_file(file_path):
                function_id = f'{f["id"]}_fn_{sig.qualname.replace(".", "_")}'

                # The first resolvable named type referenced in the function's own
                # return annotation -- this is where that type is first introduced, so
                # it's THIS function that gets the type node (see below), not
                # necessarily whatever's at the top of the call chain.
                emits_type_key = None
                emits_type_name = None
                for name in _annotation_names(sig.return_annotation):
                    resolved = resolve_type_definition(repo_root, file_dotted, file_path, name, type_resolve_cache)
                    if resolved is not None:
                        emits_type_key = f"{resolved[0]}.{name}"
                        emits_type_name = name
                        break

                all_functions[function_id] = {
                    "id": function_id,
                    "qualname": sig.qualname,
                    "params": sig.params,
                    "domain": dom["dotted"],
                    "file_id": f["id"],
                    "filename": f["filename"],
                    "emits_type_key": emits_type_key,
                    "emits_type_name": emits_type_name,
                }

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

    adjacency = call_order

    public_ids = {fid for fid, fn in all_functions.items() if not _is_internal(fn["qualname"])}
    internal_ids = set(all_functions) - public_ids

    # Collapse chains through internal (leading-underscore) functions: a public
    # function's call into one or more internal functions in a row folds into a single
    # edge to whichever public function the chain eventually reaches, carrying the
    # internal functions it passed through as `via`. An internal function never becomes
    # a node of its own -- it only survives as a name listed on the edge(s) it
    # mediates. A chain that never reaches another public function (a true dead end)
    # produces no edge at all, same as any other unconnected function.
    collapsed: dict[tuple[str, str], set[str]] = {}

    def walk(node: str, path_internals: frozenset[str], origin: str) -> None:
        for callee in adjacency.get(node, []):
            if callee == origin:
                continue  # a chain looping back to its own origin isn't a real edge
            if callee in public_ids:
                collapsed.setdefault((origin, callee), set()).update(path_internals)
            elif callee in internal_ids and callee not in path_internals:
                walk(callee, path_internals | {callee}, origin)

    for p in sorted(public_ids):
        walk(p, frozenset(), p)

    # Domain exports: every name in a domain's own __all__ -- both types and functions
    # -- shown regardless of whether it currently participates in any call, since being
    # publicly exported alone earns it a place here. A type always gets its own export
    # node (types have no "regular node" elsewhere in this graph). A function only gets
    # a separate export node if __init__.py is re-exporting it from somewhere else --
    # if __init__.py is where it's actually `def`'d, it's forced into the normal
    # function-node set below instead, so it doesn't appear twice.
    domain_exports: dict[str, dict] = {}
    forced_function_ids: set[str] = set()
    for dom, dom_dir in dom_dirs:
        init_entry = next((f for f in dom["files"] if f["filename"] == "__init__.py"), None)
        if init_entry is None:
            continue
        init_path = dom_dir / init_entry["relpath"]

        for name in dom.get("types_raw", []):
            resolved = resolve_type_definition(repo_root, dom["dotted"], init_path, name, type_resolve_cache)
            if resolved is None:
                continue
            type_key = f"{resolved[0]}.{name}"
            record_type_location(type_key, resolved[1])
            export_id = f'{init_entry["id"]}_export_type_{name}'
            domain_exports[export_id] = {
                "id": export_id,
                "kind": "type",
                "domain": dom["dotted"],
                "file_id": init_entry["id"],
                "filename": init_entry["filename"],
                "name": name,
                "type_key": type_key,
            }

        for name in dom.get("functions", []):
            local_fid = f'{init_entry["id"]}_fn_{name}'
            if local_fid in all_functions:
                forced_function_ids.add(local_fid)
                continue
            # Re-exported from elsewhere -- force its own real node too (not just the
            # marker below), wherever it's actually defined. `real_id` (when resolved)
            # is what lets render_call_graph draw the "re-exports" edge back to it.
            resolved = resolve_function_definition(repo_root, dom["dotted"], init_path, name, resolve_cache)
            real_fid = None
            if resolved is not None:
                candidate_fid = f"{file_node_id(repo_root, resolved[1])}_fn_{name}"
                if candidate_fid in all_functions:
                    forced_function_ids.add(candidate_fid)
                    real_fid = candidate_fid
            export_id = f'{init_entry["id"]}_export_fn_{name}'
            domain_exports[export_id] = {
                "id": export_id,
                "kind": "function",
                "domain": dom["dotted"],
                "file_id": init_entry["id"],
                "filename": init_entry["filename"],
                "name": name,
                "real_id": real_fid,
            }

    # A public function gets a node if it's exported (forced_function_ids, above), OR
    # it's called from a DIFFERENT file than the one it's defined in -- "called from
    # outside the module" in the literal Python sense (a .py file is a module). A
    # function called only from within its own file, and never exported, doesn't
    # qualify on its own: it's still fully internal to that file's own story, same
    # spirit as collapsing away leading-underscore functions, just for the "nobody
    # outside this file needed to know this exists" case instead of the naming
    # convention. Both ends of a qualifying cross-file edge are included, not just the
    # callee -- the caller has to be a node too, or the very edge that justifies the
    # callee's inclusion would have nothing to attach to. `calls` is filtered to match
    # below, so no edge ever references a function that didn't otherwise earn a node.
    cross_file_ids: set[str] = set()
    for a, b in collapsed:
        if all_functions[a]["file_id"] != all_functions[b]["file_id"]:
            cross_file_ids.add(a)
            cross_file_ids.add(b)
    used_ids = forced_function_ids | cross_file_ids
    functions = {fid: all_functions[fid] for fid in used_ids}

    # Type nodes: wherever a tracked function resolves a named type in its own return
    # annotation, that's a candidate for where the type is introduced -- but only the
    # FIRST such function in a chain (the one closest to the leaves) actually gets a
    # node of its own: a function that emits type T but also calls something, directly
    # or transitively, that ALSO emits T is just relaying a value that already has its
    # origin further down (e.g. ingest_document also returns IngestedDocument, but it
    # calls ingest_docx, which is where an IngestedDocument is actually built -- so
    # ingest_docx gets the node, not ingest_document). One node instance per true first
    # emitter, never shared globally, but every instance of the same type shares one
    # color (assigned in render_call_graph) so a type re-emitted at several unrelated
    # first-emission points still reads as the same type. A non-first function that
    # also emits T doesn't get a node, but its edge to its own caller is still colored
    # for T (render_call_graph), so the chain reads as one continuous colored thread:
    # ingest_docx -> IngestedDocument -> ingest_document -> (colored, no node) ->
    # ingest_documents.
    forward: dict[str, list[str]] = {}
    for a, b in collapsed:
        forward.setdefault(a, []).append(b)

    def reachable_from(start: str) -> set[str]:
        seen: set[str] = set()
        stack = list(forward.get(start, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(forward.get(cur, []))
        return seen

    emitted_types = {}
    for fid in sorted(used_ids):
        type_key = all_functions[fid]["emits_type_key"]
        if not type_key:
            continue
        if any(all_functions[d]["emits_type_key"] == type_key for d in reachable_from(fid)):
            continue  # a descendant already emits this same type -- not the first
        emitted_types[f"{fid}_emits"] = {
            "function": fid,
            "type_key": type_key,
            "name": all_functions[fid]["emits_type_name"],
        }

    # Type groups: when 2+ distinct type_keys exported from the same __init__.py share
    # the same DEFINING file, they collapse into one shared node for that file instead
    # of each getting their own export marker -- the "too many types" clutter case,
    # scoped only to the __init__.py view. A file contributing only one exported type
    # isn't grouped; it keeps its existing individual export marker untouched. This
    # never touches the chain-based first-emitter mechanism (emitted_types) -- those
    # nodes are unrelated to what a domain's __init__.py happens to export.
    used_type_keys = {e["type_key"] for e in domain_exports.values() if e["kind"] == "type"}

    by_defining_file: dict[tuple[str, str], set[str]] = {}
    for key in used_type_keys:
        loc = type_defining_file.get(key)
        if loc is None:
            continue
        by_defining_file.setdefault((loc["domain"], loc["file_id"]), set()).add(key)

    type_groups: dict[str, dict] = {}
    type_key_to_group: dict[str, str] = {}
    for (domain, file_id), keys in by_defining_file.items():
        if len(keys) < 2:
            continue
        loc = type_defining_file[next(iter(keys))]
        group_id = f"{file_id}_types"
        type_groups[group_id] = {
            "id": group_id,
            "domain": domain,
            "file_id": file_id,
            "filename": loc["filename"],
            "name": f"{Path(loc['filename']).stem}_types",
        }
        for k in keys:
            type_key_to_group[k] = group_id

    for e in domain_exports.values():
        if e["kind"] == "type":
            e["group_id"] = type_key_to_group.get(e["type_key"])

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
            for (a, b), mediators in collapsed.items()
            if a in used_ids and b in used_ids
        ],
        "emitted_types": [{"id": eid, **info} for eid, info in emitted_types.items()],
        "domain_exports": [domain_exports[k] for k in sorted(domain_exports)],
        "type_groups": [type_groups[k] for k in sorted(type_groups)],
    }


def render_call_graph(call_graph: dict) -> str:
    functions = {f["id"]: f for f in call_graph["functions"]}

    file_entries: dict[tuple[str, str], dict] = {}

    def file_slot(domain_dotted: str, file_id: str, filename: str) -> dict:
        return file_entries.setdefault(
            (domain_dotted, file_id), {"filename": filename, "function_ids": []}
        )

    for fn in call_graph["functions"]:
        file_slot(fn["domain"], fn["file_id"], fn["filename"])["function_ids"].append(fn["id"])
    # A domain export's own __init__.py file subgraph must exist even when no function
    # happens to live in that same file.
    exports_by_file: dict[tuple[str, str], list[dict]] = {}
    for de in call_graph.get("domain_exports", []):
        file_slot(de["domain"], de["file_id"], de["filename"])
        exports_by_file.setdefault((de["domain"], de["file_id"]), []).append(de)
    # Same for a type group's defining file -- it must exist even when no function or
    # export happens to live there too.
    groups_by_file: dict[tuple[str, str], list[dict]] = {}
    for g in call_graph.get("type_groups", []):
        file_slot(g["domain"], g["file_id"], g["filename"])
        groups_by_file.setdefault((g["domain"], g["file_id"]), []).append(g)

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

    # A type_key's color is shared across every instance of that type, wherever it
    # occurs -- see the module docstring's note on type nodes never being shared
    # globally: each first-emitting function gets its OWN node instance, but recurring
    # types still read as the same color. Domain-export type nodes share this same
    # color map (keyed by type_key) UNLESS grouped: a grouped export (extract_call_graph's
    # type_groups) has no color of its own -- ckey() resolves it to its group's id
    # instead. Grouping only ever affects the domain-export view, never the chain-based
    # emitted_types mechanism, so ckey() is a no-op for any type_key that only appears
    # there.
    group_id_by_type_key: dict[str, str] = {}
    for e in call_graph.get("domain_exports", []):
        if e["kind"] == "type" and e.get("group_id"):
            group_id_by_type_key[e["type_key"]] = e["group_id"]

    def ckey(type_key: str) -> str:
        return group_id_by_type_key.get(type_key, type_key)

    emitted_type_keys = {e["type_key"] for e in call_graph.get("emitted_types", [])}
    export_color_keys = {ckey(e["type_key"]) for e in call_graph.get("domain_exports", []) if e["kind"] == "type"}
    type_color = {
        key: PALETTE[i % len(PALETTE)]
        for i, key in enumerate(sorted(emitted_type_keys | export_color_keys))
    }
    emit_by_function = {e["function"]: e for e in call_graph.get("emitted_types", [])}

    # A function's own node stays a plain, uncluttered label -- its parameters (if any)
    # get their own node instead, one per function, colored distinctly (see the
    # paramNode classDef below) and feeding into it with a single edge, rather than
    # being crammed into the function's own label. A first-emitting function's type
    # (see extract_call_graph) gets its own node the same way, in the same file
    # subgraph as the function, colored by type_color rather than a shared classDef
    # since each type has its own distinct color.
    param_node_ids: list[str] = []
    param_edges: list[tuple[str, str]] = []
    emit_edges: list[tuple[str, str, str]] = []  # (function_id, emit_id, type_key)

    def emit_domain(dom: dict, children: dict[str, list[dict]], indent: str) -> list[str]:
        out = [f'{indent}subgraph {dom["id"]}["{dom["dotted"]}"]']
        pad = indent + "    "
        for file_id, entry in files_by_domain.get(dom["dotted"], []):
            out.append(f'{pad}subgraph {file_id}["{entry["filename"]}"]')
            fpad = pad + "    "
            for fn_id in entry["function_ids"]:
                fn = functions[fn_id]
                out.append(f'{fpad}{fn_id}(["{escape(fn["qualname"])}"])')
                if fn["params"]:
                    param_id = f"{fn_id}_params"
                    label = escape("<br/>".join(fn["params"]))
                    out.append(f'{fpad}{param_id}["{label}"]')
                    param_node_ids.append(param_id)
                    param_edges.append((param_id, fn_id))
                emitted = emit_by_function.get(fn_id)
                if emitted:
                    out.append(f'{fpad}{emitted["id"]}[["{escape(emitted["name"])}"]]')
                    emit_edges.append((fn_id, emitted["id"], emitted["type_key"]))
            # Domain exports (types and re-exported functions) belonging to this file --
            # see extract_call_graph's domain_exports for what's already been folded
            # into a regular function node instead (locally-defined exported functions).
            # A grouped type export has no marker of its own here -- it's represented by
            # its group node in its defining file's subgraph instead (below).
            # Functions first, then types -- explicit rather than relying on id sort order.
            file_exports = exports_by_file.get((dom["dotted"], file_id), [])
            for de in sorted(file_exports, key=lambda e: e["kind"] != "function"):
                if de["kind"] == "type":
                    if de.get("group_id"):
                        continue
                    out.append(f'{fpad}{de["id"]}[["{escape(de["name"])}"]]')
                else:
                    out.append(f'{fpad}{de["id"]}(["{escape(de["name"])}"])')
            for g in groups_by_file.get((dom["dotted"], file_id), []):
                out.append(f'{fpad}{g["id"]}[["{escape(g["name"])}"]]')
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

    # callee --> caller: every edge in this toolset points from the thing depended on
    # to the thing depending on it, and a callee is a dependency of its caller. When
    # the connection was mediated by one or more collapsed internal functions, they're
    # named right on the edge label instead of appearing as nodes of their own.
    # Labeled "returns" rather than "called by" -- this edge is really about what
    # value flows up into the caller, which reads more naturally alongside the
    # type-node routing below than a call-direction label would.
    #
    # If the callee is a first-emitter of some type (see extract_call_graph), this
    # edge is rerouted through its type node instead of drawn directly -- the node
    # itself already carries the fn-->type hop (below), so this is the type-->caller
    # continuation: emitter -> Type -> caller. If the callee merely echoes a type some
    # deeper descendant already originated (not itself a first-emitter), no node is
    # inserted, but the edge is still colored for that type -- a plain pass-through
    # continuing the same colored thread one hop further up the chain.
    #
    # (A literal reverse "calls" edge was tried here too, so each relationship showed
    # both directions -- dropped again: mermaid's ELK integration routes a back-edge as
    # an ugly loop around the outside of the nodes whenever the pair sits inside nested
    # subgraphs, which every node in this graph always does (domain > file). Confirmed
    # this is a subgraph-specific ELK/mermaid limitation, not fixable by edge order,
    # labels, or the `mergeEdges` option -- see the graph module's history for the
    # dead ends tried.)
    #
    # call_graph["calls"] is in true source call order (see extract_call_graph) -- kept
    # that way since it's the semantically correct data. Emitted here in reverse,
    # because mermaid/ELK's BT layout visually places a caller's multiple targets in
    # the opposite order from the order their edges were declared; reversing at this
    # rendering step (not in the data itself) compensates for that specific layout
    # quirk without making the underlying call_graph.json order lie about the source.
    idx = 0
    type_edge_indices: dict[str, list[int]] = {key: [] for key in type_color}

    for c in reversed(call_graph["calls"]):
        via = c.get("via") or []
        label = "returns via " + ", ".join(via) if via else "returns"
        emitted = emit_by_function.get(c["to"])
        if emitted:
            lines.append(f'    {emitted["id"]} -->|{label}| {c["from"]}')
            type_edge_indices[emitted["type_key"]].append(idx)
        else:
            lines.append(f'    {c["to"]} -->|{label}| {c["from"]}')
            passthrough_key = functions[c["to"]].get("emits_type_key")
            if passthrough_key:
                type_edge_indices[passthrough_key].append(idx)
        idx += 1
    lines.append("")

    # A parameter node feeds into its function, same direction convention as
    # everything else here: the thing depended on (the input) flows up to the thing
    # depending on it (the function that receives it).
    for param_id, fn_id in param_edges:
        lines.append(f"    {param_id} --> {fn_id}")
        idx += 1
    lines.append("")

    # A first-emitting function's own type: an undirected line (---), not an arrow --
    # this is an association (this function's type is this), not a flow direction like
    # every other edge here.
    for fn_id, emit_id, type_key in emit_edges:
        lines.append(f"    {fn_id} --- {emit_id}")
        type_edge_indices[type_key].append(idx)
        idx += 1
    lines.append("")

    # A re-exported function's marker node (in __init__.py's subgraph) traces back to
    # where it's actually defined -- dashed, since this isn't a call relationship at
    # all, just "this name is the same function as that one." Each re-export gets its
    # own PALETTE color (same convention as domains/type groups), purely so multiple
    # dashed lines stay visually distinguishable in a busy graph -- not tied to type.
    reexports = [
        de for de in call_graph.get("domain_exports", [])
        if de["kind"] == "function" and de.get("real_id")
    ]
    reexport_color = {
        de["id"]: PALETTE[i % len(PALETTE)] for i, de in enumerate(sorted(reexports, key=lambda d: d["id"]))
    }
    reexport_edge_indices: dict[str, int] = {}
    for de in reexports:
        lines.append(f'    {de["real_id"]} -.->|re-exports| {de["id"]}')
        reexport_edge_indices[de["id"]] = idx
        idx += 1
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
    lines.append("    classDef paramNode fill:#fff3cd,stroke:#c9a227,color:#5c4a00,stroke-width:1px;")
    lines.append("")

    for dom in touched_domains:
        lines.append(f'    class {dom["id"]} {dom["id"]}Domain')
    file_ids_all = [fid for _, fid in file_entries]
    if file_ids_all:
        lines.append(f'    class {",".join(file_ids_all)} fileGroup')
    if param_node_ids:
        lines.append(f'    class {",".join(param_node_ids)} paramNode')
    lines.append("")

    for dom in touched_domains:
        _, fill, _, _ = palette_for_domain[dom["id"]]
        tint = light_tint(fill)
        lines.append(f'    style {dom["id"]} fill:{tint},stroke:{fill},stroke-width:2px;')
    lines.append("")

    for e in call_graph.get("emitted_types", []):
        _, fill, stroke, color = type_color[e["type_key"]]
        lines.append(f'    style {e["id"]} fill:{fill},stroke:{stroke},color:{color};')
    for de in call_graph.get("domain_exports", []):
        if de["kind"] != "type" or de.get("group_id"):
            continue
        _, fill, stroke, color = type_color[de["type_key"]]
        lines.append(f'    style {de["id"]} fill:{fill},stroke:{stroke},color:{color};')
    for g in call_graph.get("type_groups", []):
        _, fill, stroke, color = type_color[g["id"]]
        lines.append(f'    style {g["id"]} fill:{fill},stroke:{stroke},color:{color};')
    lines.append("")

    for type_key, indices in type_edge_indices.items():
        if not indices:
            continue
        _, fill, _, _ = type_color[type_key]
        lines.append(f'    linkStyle {",".join(str(i) for i in indices)} stroke:{fill},stroke-width:2px;')

    for de_id, i in reexport_edge_indices.items():
        _, fill, _, _ = reexport_color[de_id]
        lines.append(f'    linkStyle {i} stroke:{fill},stroke-width:2px;')

    return "\n".join(lines) + "\n"
