# virtual-swap transpilation

![Tests](https://github.com/Pitt-JonesLab/virtual-swap/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/Pitt-JonesLab/virtual-swap/actions/workflows/format-check.yml/badge.svg?branch=main)

This project is exploring SWAP gate that performed via physical-logical qubit remapping.

<img src="images/vswap.png" width="200" />

In particular, we are interested in the following questions:

1. Can we use this vSWAP gate in routing algorithms to reduce the number of SWAP gates?

2. Can we use vSWAP as a decomposition resource?

The main reason this works is due to an identity that:
CX + SWAP = iSWAP.

Considering circuits written in a CX basis, then introducing vSWAPS is a convienient way to decompose into the iSWAP basis, a gate more naturally supported by superocnducting qubits.

<img src="images/decomp.png" width="200" />

Install this to draw the topologies
`sudo apt install graphviz`

#### References:

https://arxiv.org/pdf/quant-ph/0209035.pdf

https://iopscience.iop.org/article/10.1088/0953-4075/44/11/115502/pdf

https://www.proquest.com/docview/2474836120?pq-origsite=gscholar&fromopenview=true

https://journals.aps.org/pra/pdf/10.1103/PhysRevA.67.032301

https://arxiv.org/pdf/2211.03094.pdf

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9116963

https://arxiv.org/pdf/1909.07534.pdf
