"""Shared TypedDict / type-alias definitions used across more than one module."""
from __future__ import annotations

from typing import TypedDict

# Functional-form TypedDicts: used when keys are Python keywords (e.g. 'from').
Edge = TypedDict('Edge', {'from': str, 'to': str, 'kind': str})
Violation = TypedDict('Violation', {'from': str, 'to': str})
CallEdge = TypedDict('CallEdge', {'from': str, 'to': str, 'via': list[str]})


class FileEntry(TypedDict):
    id: str
    filename: str
    relpath: str


class DomainInfo(TypedDict):
    dotted: str
    id: str
    in_tree: bool
    files: list[FileEntry]
    functions: list[str]
    types_raw: list[str]


class BrokenImport(TypedDict):
    dotted: str
    id: str
    importer: str


class StandaloneFile(TypedDict):
    dotted: str
    id: str
    filename: str


class ExtractedGraph(TypedDict):
    target: str
    color_order: list[str]
    domains: list[DomainInfo]
    standalone_files: list[StandaloneFile]
    broken: list[BrokenImport]
    violations: list[Violation]
    edges: list[Edge]


class FunctionInfo(TypedDict):
    id: str
    qualname: str
    domain: str
    file_id: str
    filename: str


class TypeHighlight(TypedDict):
    type_key: str
    name: str
    highlights: dict[str, str]
    produces: list[str]
    consumes: list[str]


class CallDomain(TypedDict):
    dotted: str
    id: str
    in_tree: bool


class CallGraph(TypedDict):
    target: str
    domains: list[CallDomain]
    functions: list[FunctionInfo]
    calls: list[CallEdge]
    main_types: list[TypeHighlight]


class PersistedGraph(TypedDict, total=False):
    # Core fields (from ExtractedGraph)
    target: str
    color_order: list[str]
    domains: list[DomainInfo]
    standalone_files: list[StandaloneFile]
    broken: list[BrokenImport]
    violations: list[Violation]
    edges: list[Edge]
    # Fields added by trim_for_persistence / generate_one
    own_content_hash: str
    reverse_scan_applied: bool
    types_cluster: str | None
    types_raw: list[str]
    # Field on submodule-pointer domain entries
    see: str
