from .adgn import ADGN
from .drew_delay import DRew_GCN
from .gnn import GNN
from .graphcon import GraphCON
from .phdgn import PHDGN
from .swan import SWAN
from .ghr_model import GHRModel


models_map = {
    "GNN": GNN,
    "ADGN": ADGN,
    "DRew_GCN": DRew_GCN,
    "GraphCON": GraphCON,
    "PHDGN": PHDGN,
    "SWAN": SWAN,
    "GHR": GHRModel,
}