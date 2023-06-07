twolocal reps=1, 3x4 grid, N=5, LAYOUT/SWAP_TRIALS=6
\_linear: 4Q [6->6], 8Q [14->14]
\_full: 4Q [16->10], 8Q [52->39]
\_circular: 4Q [8,8], 8Q [16->16]
\_scal: 4Q [8->8], 8Q [16->16]

so what is it about that full entanglement that lets it find an advantage?

repeat \full with reps=4:
4Q [50->34], 8Q [238->208]

problem!! if entangling gate is a CX, then we would instead use reverse_linear, which gives the same unitary with less gates.

\reverse_linear:
4Q [6->6], 8Q [14->14]

AaHHHHHHHHHHHHHHHHHHHH

(worth noting, that some results look really good over the MQTBench tests, but they use a full entangling pattern, possibly unknowing they could have instead used a reverse_linear pattern)

#### small worth keeping

toffoli
fredkin
adder
qec_sm
qec_en
variational
hs4
shor
pea
error_correctiond3
simons
qaoa
hhl
dnn
qpe
ising
