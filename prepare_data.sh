#!/bin/bash
# Decompress the CSV data files (needed once, before training).
set -e
cd "$(dirname "$0")/data"
gunzip -k excitation_transitions.csv.gz
gunzip -k quenching_transitions.csv.gz
echo "Done. data/excitation_transitions.csv and data/quenching_transitions.csv are ready."
