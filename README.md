# mirror-gate routing

#### (virtual-swap transpilation)

This project focuses on optimizing quantum transpilation by combining layout and routing stages with gate decomposition. The mirror gate of $\texttt{U}$ is $\texttt{U} \cdot \texttt{SWAP}$. It can be used to make routing cheaper without changing the cost of decomposition. Or in some cases, can make decomposition cheaper.

![Tests](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/format-check.yml/badge.svg?branch=main)

## Project Overview

Traditionally, transpilation in quantum computing treats layout and routing as separate from gate decomposition. However, our work shows significant improvements in circuit depth and gate count by breaking this convention. We utilize the equivalence of the decomposition of CNOT and CNOT+SWAP operations into iSWAP gates and introduce the concept of "free" SWAP operations within the CNOT+SWAP sequence.

## Key Features

_TODO: Insert the main algorithm or function code snippet here_

## Getting Started

_TODO: Insert instructions on how to install and run the project, including any dependencies here_

## Results & Comparisons

Our experiments demonstrate significant reductions in circuit depth and swap count compared to traditional methods across various topologies.

_TODO: Insert the code snippet showing results or comparison chart here_

## Reference

\_TODO:

## Citing Our Work

_Paper in preparation_

_TODO: Insert how to cite your work here_
