"""Inline assets.json into template.html -> harness.html (one self-contained page)."""
import pathlib
import sys

d = pathlib.Path(__file__).parent
assets = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else d / 'assets.json'
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else d / 'harness.html'
a = assets.read_text()
t = (d / 'template.html').read_text()
out.write_text(t.replace('/*ASSETS*/null', a.replace('</', '<\\/')))
print(out)
