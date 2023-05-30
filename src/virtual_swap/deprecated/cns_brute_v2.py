"""CNS Brute force search V2.

This is a TransformationPass that acts as a partial decompositon +
routing pass. Importantly, we want to decouple the initial layout stage,
so we should consider that the first layer of the dag is a fixed layout.
Then, the transformation iterates through execution layers and will
evaluate all combinations of the candidate CNS substitutions. This will
act as an improvement over the naive brute force, which considers
combinations over the entire circuit. Further, will act as a starting
place for the development of integration into SABRE.

Second, we need to rewrite the logic of checking for decomposition cost.
Rather than continuously calling the unroller, we should instead force a
2Q block consolidation. Then, we can reason about gates from the
perspective of monodromy and coverage sets. This will allow us to reason
about the cost of gates, without needing to force the transpiler to
remain in a particular gate set.
"""


from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class CNS_Brute_V2(TransformationPass):
    def __init__(self):
        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        return dag
