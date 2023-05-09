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
