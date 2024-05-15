#!/bin/bash

#SBATCH --job-name=oracle_server  # Job name
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --mem=8G                   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --time=48:00:00             # Time limit hrs:min:sec
#SBATCH --output=/nfs/ap/mnt/sxtn2/oracle_server_logs/server_run%j.log   # Standard output and error log (%j expands to jobId)




python3 /nfs/ap/mnt/sxtn/phil/ChemLacticaTestSuite/src/oracle_server/oracle_server.py --vina_path /nfs/ap/mnt/sxtn2/chem/chemlm_oracles/vina_bin/autodock_vina_1_1_2_linux_x86/bin/vina
