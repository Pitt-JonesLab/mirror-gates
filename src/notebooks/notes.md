# notes

1- sabre is decomposition ignorant
cannot find better Routes
it thinks adding swaps have same costs
but some swaps are cheaper post decomp

2- it better to implement some lookahead where we consider a gate has its outputs reversed before adding to the mapped dag

we have 1 implemented, considering different costs from SWAPs (using virtual-SWAPs), because some SWAPs are a part of other gates that get consoldiated, and overall cheaper

to do 2, we need to consider gates more than just leaves in the intermediate layer

- include some debug print statement, do I ever make a substitution of case 2 that I would not have made before?

simple statements for letters paper:

1- not all SWAPs are equivalent
2- we can choose to flip gates
(we might also check for non-leaf gates, cases where its predecessor was already routed)

to be in intermediate layer, your predecssor is in mapped layer,
to be in front layer, your predecessor is in intermediate or mapped layer.

TODO - build a case by hand that differentiates (1) and (2). Think about 2 gates on different edges that are routable and make a change on the first gate - hint: depends on topology.

### TODO

- [] rewrite SabreCNS
  - [] should take in 2Q gates w.r.t a basis
  - [] should evaluate all gates for subs
  - [] optional to trigger this behavior change
  - [] need to print when this special sub is made
  - [] whether to keep gates in front layer if their predecessors are still in the intermediate layer
- [] parallel multiple attempts
- [] documentation
