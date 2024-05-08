from vina import Vina
from typing import List
from enum import Enum
import json
import os
from dataclasses import dataclass


ROOT_CONFIG_PATH = "augmented_memory/beam_enumeration_reproduce_experiments/drug-discovery-experiments/"

NUM_POSES = 1
VINA_SEED = 42
MAXIMUM_ITERATIONS = 600
VINA_EXHAUSTIVENESS = 8  

def get_receptor_file_path(experiment_name) -> str:
    experiment_receptor_map ={
        "drd2": "6cm4-grid.pdbqt",
        "mk2": "3kc3-grid.pdbqt",
        "ache": "1eve-grid.pdbqt",
        }
    return os.path.join(ROOT_CONFIG_PATH, experiment_name,experiment_receptor_map[experiment_name])

@dataclass
class VinaConfig:
    centers : List[float]
    box_size : List[float]
    experiment: str
    transform_sigmoid_low: int
    seed: int = VINA_SEED
    num_poses: int = NUM_POSES
    exhaustiveness: int = VINA_EXHAUSTIVENESS # please see https://github.com/MolecularAI/DockStream/blob/c62e6abd919b5b54d144f5f792d40663c9a43a5b/dockstream/utils/enums/AutodockVina_enums.py#L74 # noqa

    def __post_init__(self):
        self.receptor = get_receptor_file_path(self.experiment)

class VinaConfigEnum(Enum):
    DRD2 = VinaConfig(centers=[9.93, 5.85, -9.58], box_size=[15, 15, 15],experiment="drd2", transform_sigmoid_low = -16)
    ACHE = VinaConfig(centers=[2.78, 64.38, 67.97], box_size=[15, 15, 15],experiment="ache", transform_sigmoid_low = -18)
    MK2 = VinaConfig(centers=[-61.62, 30.31, -21.9], box_size=[15, 15, 15],experiment = "mk2", transform_sigmoid_low = -16)
