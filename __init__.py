"""Standalone dependency-graph/call-graph generation tool for rcg-agents, exposed to
Claude Code as an MCP server (server.py) and runnable directly via its own venv. This
package marker exists so tooling that resolves Python modules by walking up directories
for `__init__.py` (test runners, editor integrations) can find this as a proper module
root -- the tool's own scripts still import each other as plain top-level modules
(`from extract import extract`), not through this package name.
"""
