import os as _os, sys as _sys
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..')))
from ghr_model import GHRModel, RMSNorm, SwiGLU, _build_mlp  # noqa: F401
