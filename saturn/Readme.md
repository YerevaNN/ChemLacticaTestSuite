# The `saturn` module of ChemLacticaTestSuite
This module contains the utilites to run molecule optimization against GEAM Oracle, based on docking scores over specific protein targets.

## Setup
Run `conda install -c conda-forge openbabel` and `conda install morfeus-ml -c conda-forge`

Modify the path to in `saturn/saturn/oracles/docking/geam_oracle.py`, set the
`self.vina_path` to be in format `/auto/home/{username}/ChemLacticaTestSuite/saturn/saturn/oracles/docking/docking_grids/qvina02`
