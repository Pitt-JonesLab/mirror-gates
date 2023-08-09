# MIRAGE (Mirror-decomposition Integrated Routing for quantum Algorithm Gate Efficiency)
![Tests](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/format-check.yml/badge.svg?branch=main)

## Project Overview

This project focuses on optimizing quantum transpilation by combining layout and routing stages with gate decomposition. The mirror gate of $\texttt{U}$, $\texttt{U} \cdot \texttt{SWAP}$, can be used to make routing cheaper without changing the cost of decomposition or in some cases can make decomposition cheaper. Traditionally, transpilation in quantum computing treats layout and routing as separate from gate decomposition. However, our work shows significant improvements in circuit depth and gate count by breaking this convention. We utilize the equivalence of the decomposition of CNOT and CNOT+SWAP operations into iSWAP gates and introduce the concept of "free" SWAP operations within the CNOT+SWAP sequence.

## Key Features

Mirage algorithm defined in `src/mirror_gates/mirage.py`

![image](https://github.com/Pitt-JonesLab/mirror-gates/assets/47376937/08408089-561a-4799-9904-a2637d829edd)

## Getting Started

Run `make init` to setup python environment. The main pass manager is defined in `src/mirror_gates/pass_managers.py`. Experiments are defined in `src/notebooks/results`

## Results & Comparisons

Our experiments demonstrate significant reductions in circuit depth and swap count compared to traditional methods across various topologies.

![image](https://github.com/Pitt-JonesLab/mirror-gates/assets/47376937/81653cab-24c1-4170-ac5a-438c94d2bab3)

## Reference

```bibtex
@misc{mckinney2023mirage,
      title={MIRAGE: Quantum Circuit Decomposition and Routing Collaborative Design using Mirror Gates},
      author={Evan McKinney and Michael Hatridge and Alex K. Jones},
      year={2023},
      eprint={2308.03874},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
