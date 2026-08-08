# MQCT-ML

Neural network model for predicting H2O + H2 rotational state-to-state transition
cross sections, described in Joy, Occhiogrosso, Mandal & Babikov, *"A Machine Learning
Approach for Computationally Efficient Prediction of State-to-State Transition Cross
Sections for Molecular Collision Processes"* (manuscript under review).

A plain MLP (13 input features -> 4 hidden layers of 128 nodes -> 1 output, GELU
activations) is trained on the available H2O + H2 collision cross section data
(sampled by initial rotational state, across 7 collision energies).

## Setup

```bash
pip install -r requirements.txt
```

```bash
bash prepare_data.sh
```

## Running

Four physically distinct subsets are trained as separate models: excitation/quenching
transitions, crossed with ortho-H2/para-H2 projectile symmetry. Run all four and get
the combined result in one command:

```bash
bash run_all.sh

python src/combine_results.py results/exc_ortho results/exc_para results/que_ortho results/que_para
```
