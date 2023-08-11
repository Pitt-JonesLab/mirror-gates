# MIRAGE (Mirror-decomposition Integrated Routing for quantum Algorithm Gate Efficiency)
![Tests](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/format-check.yml/badge.svg?branch=main)

## Project Overview

- This project focuses on optimizing quantum transpilation by combining layout and routing stages with gate decomposition.
- The mirror gate of $\texttt{U}$, $\texttt{U} \cdot \texttt{SWAP}$, can be used to make routing cheaper without changing the cost of decomposition or in some cases can make decomposition cheaper. Traditionally, transpilation in quantum computing treats layout and routing as separate from gate decomposition. However, our work shows significant improvements in circuit depth and gate count by breaking this convention.
- We utilize the equivalence of the decomposition of CNOT and CNOT+SWAP operations into iSWAP gates and introduce the concept of "free" SWAP operations within the CNOT+SWAP sequence.

## Key Features

Mirage algorithm defined in `src/mirror_gates/mirage.py`

![image](https://github.com/Pitt-JonesLab/mirror-gates/assets/47376937/08408089-561a-4799-9904-a2637d829edd)

## Results & Comparisons

- Experiments are defined in `src/notebooks/results`
- Our experiments demonstrate significant reductions in circuit depth and swap count compared to traditional methods across various topologies.

![image](https://github.com/Pitt-JonesLab/mirror-gates/assets/47376937/81653cab-24c1-4170-ac5a-438c94d2bab3)

## Getting Started
Main demo: `src/mirror_gates/notebooks/bench.ipynb`.

### Usage
1. Use Mirage as a complete pass manager.

```python
from qiskit.transpiler import CouplingMap
coupling_map = CouplingMap.from_grid(6, 6)
```
Handles all pre-, post-processing steps used in the paper, only requires a qubit connectivity as an input.
```python
from mirror_gates.pass_managers import Mirage
mirage = Mirage(
            coupling, # coupling map
            name="Mirage-$\sqrt{\texttt{iSWAP}}$", # transpile_benchy and figure labels)
            parallel=True, # run trials in parallel or serial
            cx_basis=False, # turning on sets CNOT as the basis gate,
            # (can take arbitrary basis but parameters are not configured that way yet)
            cost_function="depth", # switch to "basic" for counting SWAPs
            fixed_aggression=None, # force aggression level on all iterations
            layout_trials=None, # how many independent layout trials to run (20)
            fb_iters=None, # how many forward-backward iterations to run (4)
            swap_trials=None, # how many independent routing trials to run (20)
            no_vf2=False, # keep False to use VF2 for finding complete layouts
            logger=None, # from logging moduel
)
mirage_qc = Mirage.run(qc)
```
2. Use as a Qiskit-Plugin to incorporate into your own larger transpilation pipeline.
>❗ coming soon. Some prts of this project are already configured as plugins (`src/mirror_gates/qiskit/legacy_plugin.py`) but the complete Mirage has not yet for various technical reasons.

```python
from qikist import transpile
mirage_qc = transpile(
                  qc, # input circuit
                  optimization_level = 3, # default: Qiskit's highest level
                  coupling_map=coupling_map,
                  basis_gates= ["u", "xx_plus_yy", "id"],
                  # custom routing use a subclass of qiskit.transpiler.passes.SabreSwap
                  routing_method="mirage" if self.python_sabre else None,
                  # custom layout modifies SabreLayout for parallel trials
                  layout_method="sabre_layout_v2" if self.python_sabre else None,
            )
```

### Prerequesites
- Monodromy requires [`lrs`](http://cgm.cs.mcgill.ca/~avis/C/lrs.html).
   Typically, this means downloading the source, building it, and placing the generated executable somewhere in the search path for your python

- ⚠️ There are two other packages of mine that this project depends on.

  🛑 For simplest method to use as a Qiskit-plugin uncomment the following lines in `setup.cfg` so they will be included as dependencies.
 ```
    # monodromy @ git+https://github.com/evmckinney9/monodromy.git
    ##  NOTE, monodromy can be installed via git, but during development
    ## I want to be able to edit the source code and have it reflected
    ## so I install it manually in the Makefile
    ## when I'm done, don't forget to update .github/workflows/test.yml

    # transpile_benchy @ git+https://github.com/evmckinney9/transpile_benchy.git
    ## NOTE, transpile_benchy installed manually in Makefile
    ## unlike monodromy, transpile_benchy cannot be installed via git
    ## this is because of the way I reference its submodules
```
- By default I do not include these as dependencies. For simple editting and debugging practices, these packages already exist on my computer as sibling repostiories since I am developing the interworking parts in parallel. If I include them as dependencies, they will install into the virtualenvironemnt and now you have multiple clones when you would rather have only one (at least for my purposes). The debugger will trace into the version you then can directly edit on and the changes will be applied everywhere since it is installed in editable mode.

- By default, running `make init` will clone each of these repository as siblings to the repo root directory, or if the clone already exists, pulls the latest changes. I used this so I could easily accumulate all changes when needed.

#### [transpile_benchy](https://github.com/evmckinney9/transpile_benchy).
- Handles circuit benchmarks, crunching data, and plotting.

#### [My fork of monodromy](https://github.com/evmckinney9/monodromy)
 - Added features for acting as a Qiskit AnalysisPass.

### Using a a Qiskit-Plugin
>❗ coming soon

### Setting up for development
- Please submit any issue you find. Currently the most unstable part is related to the parallel processing. 😺
- Run `make init` to setup the virtualenvironment and install itself in edittable mode. This also will setup the tools for formatting and testing commands. Part of this script will clone each repostiory and install it into the virtualenvironment as noted above. For more information, visit my [python-template](https://github.com/evmckinney9/python-template)`.

#### 💻🐒
- The main logic of the MIRAGE pass is in `src/mirror_gates/mirage.py` which includes a wrapper `ParallelMirage` for parallelism and statisticis, and the class `Mirage`, itself a subclass of `qiskit.transpiler.passes.SabreSwap` to handle serial passes`.
- The complete main pass manager is defined in `src/mirror_gates/pass_managers.py`.
- Circuit benchmarks are defiend as `.txt` in `src/mirror_gates/circuits/`. These are loaded into a `transpile_benchy.Library` object.
- For more details, see code documentation or contact me.

### Makefile
🛑 This part of `Makefile` handles the sibling project clones. It can be commented out if they are included as python dependencies in `setup.cfg`, which enabled installing both without cloning.
```
	if [ -d "../transpile_benchy" ]; then \
		echo "Repository already exists. Updating with latest changes."; \
		cd ../transpile_benchy && git pull; \
	else \
		cd .. && git clone https://github.com/evmckinney9/transpile_benchy.git --recurse-submodules; \
		cd transpile_benchy; \
	fi
	$(PIP) install -e ../transpile_benchy --quiet
	if [ -d "../monodromy" ]; then \
		echo "Repository already exists. Updating with latest changes."; \
		cd ../monodromy && git pull; \
	else \
		cd .. && git clone https://github.com/evmckinney9/monodromy.git; \
		cd monodromy; \
	fi
	$(PIP) install -e ../monodromy --quiet
```

- Also ncluded in `Makefile` are the helpful commands: `make format`, `make clean`, `make test`.

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
