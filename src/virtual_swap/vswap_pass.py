"""V4 Virtual-SWAP Layout+Routing Pass.

The main thing that we need to take careful consideration of is
the virtual-physical Layout objects. I think I have been mixing
up the indices of which is which. Also, I might prefer to use
a custom DAG class internal to this pass. This would let me edit
attributes of the OpNodes.

Second, we need to focus on preserving the unitary operation.
This means that we should be able to verify that the circuit
before and after transformation is still the same unitary.

Third, consider a version that makes the CNS substitution each iteration,
this makes it easier to visualize what changes are being made. But also,
we can make a version that waits until the end. Swapping the wires of
gate descendants is expensive, and we don't need to do it in order to
calculate topological distance (can just reference a dynamic layout).
(Third, ideally this should be addressed by a more efficient custom DAG operation.
Ref: https://github.com/Qiskit/qiskit-terra/pull/9863)
"""

from qiskit.transpiler.basepasses import TransformationPass


class VSwapPass(TransformationPass):
    def __init__(self):
        super().__init__()

    def run(self, dag):
        pass
