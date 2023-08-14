# MIRAGE (Mirror-decomposition Integrated Routing for quantum Algorithm Gate Efficiency)
![Tests](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/tests.yml/badge.svg?branch=main)
![Format Check](https://github.com/Pitt-JonesLab/mirror-gates/actions/workflows/format-check.yml/badge.svg?branch=main)

## 📌 Project Overview

- **Objective**: Optimize quantum transpilation by unifying the layout and routing stages with gate decomposition.
- **Strategy**: Employ the mirror gate of $\texttt{U}$, represented as $\texttt{U} \cdot \texttt{SWAP}$, to achieve more cost-efficient routing without altering decomposition costs. In certain cases, it can even reduce decomposition expenses.

## 🌟 Key Features
- **Mirage Algorithm**: Defined in `src/mirror_gates/mirage.py`

![image](https://github.com/Pitt-JonesLab/mirror-gates/assets/47376937/08408089-561a-4799-9904-a2637d829edd)

## 📊 Results & Comparisons

- **Experiments**: Detailed in `src/notebooks/results`
- **Findings**: Our methodology considerably reduces circuit depth and swap count when compared with conventional techniques across multiple topologies.

![image](https://github.com/Pitt-JonesLab/mirror-gates/assets/47376937/81653cab-24c1-4170-ac5a-438c94d2bab3)

## 🚀 Getting Started
Get started with MIRAGE by exploring the main demo located at `src/mirror_gates/notebooks/bench.ipynb`.

### 💻🐒 Usage

```python
from qiskit.transpiler import CouplingMap
coupling_map = CouplingMap.from_grid(6, 6)
```

#### 1. Use Mirage as a complete pass manager.
Handles all pre-, post-processing stages described in our paper:
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

#### 2. Qiskit-Plugin Approach
Integrate MIRAGE into your existing transpilation pipeline:
>❗ This feature is still currently under development:
```python
from qikist import transpile
mirage_qc = transpile(
                  qc, # input circuit
                  optimization_level = 3, # default: Qiskit's highest level
                  coupling_map=coupling_map,
                  basis_gates= ["u", "xx_plus_yy", "id"],
                  routing_method="mirage",
                  layout_method="sabre_layout_v2",
            )
```

### 📋 Prerequisites
- **Monodromy Dependency**: This needs lrs. To install:
	- `sudo apt install lrslib`

- **Package Dependencies**: By default, two other packages are dependencies:
	- [transpile_benchy](https://github.com/evmckinney9/transpile_benchy): Manages circuit benchmarks, data analytics, and plotting.
	-[monodromy (fork)](https://github.com/evmckinney9/monodromy): modified for Qiskit AnalysisPass integration.

- ⚠️ **Setup**: Running `make init` sets up the required environment and tools. It also clones required repositories.
	- **Optional**: If you want to leverage the additional features from transpile_benchy, especially its submodules for circuit benchmarking, run `make dev-setup` after initialization. This will clone and set up transpile_benchy with its complete functionalities.

### Dive Deeper into the Code 💻🐒
- **Please report any issues**. (Currently the most unstable part is related to the parallel processing. 😺)
- The main logic of the MIRAGE pass is in `src/mirror_gates/mirage.py` which includes a wrapper `ParallelMirage` for parallelism and statisticis, and the class `Mirage`, itself a subclass of `qiskit.transpiler.passes.SabreSwap` to handle serial passes`.
- The complete main pass manager is defined in `src/mirror_gates/pass_managers.py`.
- Circuit benchmarks are defiend as `.txt` in `src/mirror_gates/circuits/`. These are loaded into a `transpile_benchy.Library` object.
- For more details, see code documentation or contact me.

Additional utility commands available in the Makefile:

- **make format**: Formats the codebase.
- **make clean**: Cleans up temporary and unnecessary files.
- **make test**: Runs tests to ensure code functionality.
- For more information about the repository structure, visit my [python-template](https://github.com/evmckinney9/python-template)`.

## 📚 Reference

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
