"""Put the core package and the evaluation harness on sys.path.

Both use flat imports, so the tests need their directories on the path
rather than their parents.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.dirname(_HERE)               # foresight/
_ROOT = os.path.dirname(_CORE)               # repo root
for p in (_CORE, _ROOT, os.path.join(_ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
