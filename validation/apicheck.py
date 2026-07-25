"""Check that every exported symbol appears in docs/src/api.md and has a docstring."""
import re, sys, pathlib
root = pathlib.Path(sys.argv[1])
mod = (root / "src" / "NeuralOT.jl").read_text()
exports = []
for line in mod.splitlines():
    m = re.match(r"\s*export\s+(.*)", line)
    if m:
        exports += [x.strip() for x in m.group(1).split(",") if x.strip()]
api = (root / "docs" / "src" / "api.md").read_text()
documented = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_!]*)$", api, re.M))
missing_api = [e for e in exports if e not in documented]

src = "\n".join((root / "src" / f).read_text() for f in
                [p.name for p in (root / "src").glob("*.jl")])
# a docstring is a """ block immediately preceding a definition mentioning the name
blocks = re.findall(r'"""\s*\n(.*?)"""', src, re.S)
has_doc = set()
for b in blocks:
    first = b.strip().split("\n")[0].strip()
    nm = re.match(r"([A-Za-z_][A-Za-z0-9_!]*)", first)
    if nm:
        has_doc.add(nm.group(1))
missing_doc = [e for e in exports if e not in has_doc]

print(f"exported symbols: {len(exports)}")
print("OK  api.md covers every export" if not missing_api
      else f"!!  missing from api.md: {missing_api}")
print("OK  every export has a docstring" if not missing_doc
      else f"!!  missing docstring: {missing_doc}")
sys.exit(1 if (missing_api or missing_doc) else 0)
