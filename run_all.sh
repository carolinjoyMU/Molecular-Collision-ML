#!/bin/bash
# Trains all four physical subsets (excitation/quenching x ortho-H2/para-H2)
# and prints the final combined overall + per-U RRMSE table.
set -e
cd "$(dirname "$0")"

if [ ! -f data/excitation_transitions.csv ]; then
    echo "Data not prepared yet, running prepare_data.sh..."
    bash prepare_data.sh
fi

python src/train.py --process excitation --h2_symmetry ortho --out_dir results/exc_ortho
python src/train.py --process excitation --h2_symmetry para  --out_dir results/exc_para
python src/train.py --process quenching  --h2_symmetry ortho --out_dir results/que_ortho
python src/train.py --process quenching  --h2_symmetry para  --out_dir results/que_para

python src/combine_results.py results/exc_ortho results/exc_para results/que_ortho results/que_para
