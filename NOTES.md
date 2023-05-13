My concern is that the new and faster approach is not going to work.

What this does is makes it so we know if we want to accept a CNOT having its output flipped
so if we are going to take a SWAP path, then we should accept if flipping the outputs puts us closer to the next gate

but this just means at best, we remove 1 swap from everytime we would have needed to go on a path.


okay that is fine, but I think the original intention was better.
 When we make routes, we really want to make as many CNS gates as possible,
well idk is this the same thing? the idea is that we want the swap gates to be cheaper
so place swaps on cnots to turn them into iswaps,....

the issue is that if temperature is 0 it should be at least as good as baseline, but it is not
something more complicated is happening
just bugs relating to counting and decomposing, OR
I think its actually just very inefficiently doing placement and routing still.
Maybe we should try doing our gate as a preinput to built transpiler
it wouldnt be accurate, but imagine we just rewrite CX outputs
then pass the result to qiskit
then pass it back to custom decomposition, idkkkk its hard to let qiskit manipulate stuff
but then also keep track of nodes to know if decomp (a or b)


hmmmm....

idk conceptually its so simple but at the same time so complicated :(
