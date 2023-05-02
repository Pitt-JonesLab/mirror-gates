make change
evaluate cost
make decision based on temperature to accept it

how make cost function that takes into account where swap gates go

rather than doing virtual swap
just place the swap but have it cost of 0

routing potential- don't need to route

difference between the slack versus the distance..
need to consider slack, if successor is far away but has time to do SWAPs
each qubit has its own successor

look at the qubit's coming in - and based on coming in are they next to each other
if not, have some penalty: Q is how far away is the parent
(go bottom up) from end of circuit to beginning

put on the line how much slack we have on each edge
when get down to evluating the inputs, look at the slack on each input + and the distance
determines how many SWAPs are on critical path


<!--
# interesting idea,
# rather than starting with all as 2sqiswap, could start with all random

# build something modular
# can compare between qiskit, brute force (MC), SA, look ahead, etc.
# the idea I like is to work backwards, continuously checking the sub, then

# idea 1, start with a solved circuit and make changes to it continuously fixing but trying to change cost

# idea 2, harder but build the route from scratch considering both types of gates -->
