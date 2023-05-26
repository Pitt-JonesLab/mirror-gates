# virtual-swap transpilation
This project focuses on optimizing quantum transpilation by combining layout and routing stages with gate decomposition, primarily using the iSWAP gate and a unique concept of a "virtual swap" gate.

![Tests](https://github.com/Pitt-JonesLab/virtual-swap/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/Pitt-JonesLab/virtual-swap/actions/workflows/format-check.yml/badge.svg?branch=main)

## Project Overview

Traditionally, transpilation in quantum computing treats layout and routing as separate from gate decomposition. However, our work shows significant improvements in circuit depth and gate count by breaking this convention. We utilize the equivalence of the decomposition of CNOT and CNOT+SWAP operations into iSWAP gates and introduce the concept of "free" SWAP operations within the CNOT+SWAP sequence.

## Key Features

1. **iSWAP Gate Recognition**: Our approach hinges on treating the iSWAP as a basis gate and understanding its role in decomposing two-qubit operations.

2. **Virtual SWAP (vSWAP) Gate**: A novel aspect of our work is the introduction of a "virtual swap" gate, which helps to optimize circuit depth and swap count. This takes advantage of the "free" data movement within CNOT+SWAP operations.

3. **Routing Algorithm**: We provide a routing algorithm that leverages the above features to minimize circuit depth and swap count.

_TODO: Insert the main algorithm or function code snippet here_

4. **Circuit Compression**: By integrating gate decomposition into layout and routing, our approach enhances circuit compression during transpilation, resulting in reduced circuit depth and gate count.

_TODO: Insert the code snippet showing how circuit compression is enhanced here_

## Getting Started

_TODO: Insert instructions on how to install and run the project, including any dependencies here_

## Results & Comparisons

Our experiments demonstrate significant reductions in circuit depth and swap count compared to traditional methods across various topologies.

_TODO: Insert the code snippet showing results or comparison chart here_

## Reference

_TODO:

## Citing Our Work
_Paper in preparation_

_TODO: Insert how to cite your work here_

We hope you find this project useful for your quantum computing endeavors. Feel free to reach out with any questions or suggestions.
