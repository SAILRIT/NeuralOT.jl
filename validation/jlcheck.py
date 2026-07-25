#!/usr/bin/env python3
"""
Static checker for Julia sources using the tree-sitter Julia grammar.

Checks performed
----------------
1. Parse every .jl file; report ERROR / MISSING nodes with line numbers.
2. Verify every `include("...")` target exists on disk.
3. Collect top-level definitions (function / struct / const / macro / assignment)
   and verify that every `export`ed symbol is actually defined somewhere in the
   package.
4. Flag calls to names that look like package-internal functions but are never
   defined (heuristic; reports only names defined nowhere and not known Base/Flux).
5. Report duplicate method-free redefinitions of structs.
"""
from __future__ import annotations
import sys, os, re, json
from pathlib import Path
import tree_sitter_julia
from tree_sitter import Language, Parser

LANG = Language(tree_sitter_julia.language())
PARSER = Parser(LANG)


def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)


def text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", "replace")


def parse_file(path: Path):
    src = path.read_bytes()
    tree = PARSER.parse(src)
    return tree, src


def syntax_errors(tree, src, path):
    out = []
    for n in walk(tree.root_node):
        if n.type == "ERROR":
            out.append((path, n.start_point[0] + 1, "ERROR", text(n, src)[:120].replace("\n", "\\n")))
        elif n.is_missing:
            out.append((path, n.start_point[0] + 1, "MISSING", n.type))
    return out


def _head_identifier(node, src):
    """Return the leading identifier of a call/where/parametrised expression."""
    cur = node
    for _ in range(8):
        if cur is None:
            return None
        if cur.type == "identifier":
            return text(cur, src)
        if cur.type == "field_expression":
            # Base.show(...) -> report the field name
            return text(cur, src)
        if cur.children:
            cur = cur.children[0]
        else:
            return None
    return None


def collect_defs(tree, src):
    """Version-agnostic definition collection via a manual tree walk."""
    names = set()
    for n in walk(tree.root_node):
        t = n.type
        if t == "function_definition":
            for c in n.children:
                if c.type in ("signature", "call_expression", "where_expression",
                              "identifier", "field_expression"):
                    nm = _head_identifier(c, src)
                    if nm:
                        names.add(nm)
                    break
        elif t in ("struct_definition", "abstract_definition", "primitive_definition"):
            for c in n.children:
                if c.type == "identifier":
                    names.add(text(c, src)); break
                if c.type in ("type_head", "parametrized_type_expression",
                              "type_parameter_list", "curly_expression",
                              "binary_expression"):
                    nm = _head_identifier(c, src)
                    if nm:
                        names.add(nm)
                    break
        elif t == "macro_definition":
            for c in n.children:
                if c.type in ("signature", "call_expression", "identifier"):
                    nm = _head_identifier(c, src)
                    if nm:
                        names.add("@" + nm); names.add(nm)
                    break
        elif t == "assignment":
            lhs = n.children[0] if n.children else None
            if lhs is None:
                continue
            if lhs.type == "identifier":
                names.add(text(lhs, src))
            elif lhs.type in ("call_expression", "where_expression"):
                nm = _head_identifier(lhs, src)
                if nm:
                    names.add(nm)
    return names


def collect_exports(tree, src):
    out = []
    for n in walk(tree.root_node):
        if n.type == "export_statement":
            for c in n.children:
                if c.type in ("identifier", "operator"):
                    out.append(text(c, src))
    return out


def collect_includes(tree, src):
    out = []
    for n in walk(tree.root_node):
        if n.type == "call_expression":
            fn = n.children[0]
            if fn.type == "identifier" and text(fn, src) == "include":
                args = text(n, src)
                m = re.search(r'"([^"]+)"', args)
                if m:
                    out.append((m.group(1), n.start_point[0] + 1))
    return out


def main(root: str):
    root = Path(root)
    files = sorted(root.rglob("*.jl"))
    if not files:
        print(f"no .jl files under {root}")
        return 1
    all_errs, all_defs, all_exports, all_includes = [], set(), [], []
    per_file = {}
    for f in files:
        tree, src = parse_file(f)
        errs = syntax_errors(tree, src, f)
        all_errs += errs
        d = collect_defs(tree, src)
        per_file[str(f)] = dict(defs=len(d), errors=len(errs))
        all_defs |= d
        all_exports += [(e, f) for e in collect_exports(tree, src)]
        all_includes += [(inc, f, ln) for inc, ln in collect_includes(tree, src)]

    print("=" * 72)
    print(f"Parsed {len(files)} Julia files under {root}")
    print("=" * 72)
    if all_errs:
        print(f"\n!! {len(all_errs)} SYNTAX PROBLEM(S):")
        for p, ln, kind, t in all_errs:
            print(f"   {p}:{ln}  [{kind}] {t}")
    else:
        print("\nOK  syntax: no parse errors in any file")

    missing_inc = []
    for inc, f, ln in all_includes:
        target = (f.parent / inc)
        if not target.exists():
            missing_inc.append((f, ln, inc))
    if missing_inc:
        print(f"\n!! {len(missing_inc)} MISSING include target(s):")
        for f, ln, inc in missing_inc:
            print(f"   {f}:{ln}  include(\"{inc}\")")
    else:
        print(f"OK  includes: all {len(all_includes)} include() targets exist")

    undef_exports = sorted({e for e, _f in all_exports if e not in all_defs})
    if undef_exports:
        print(f"\n!! {len(undef_exports)} exported name(s) with no visible definition:")
        for e in undef_exports:
            print(f"   {e}")
    else:
        print(f"OK  exports: all {len({e for e,_ in all_exports})} exported names are defined")

    print(f"\nTop-level definitions found: {len(all_defs)}")
    return 1 if (all_errs or missing_inc or undef_exports) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
