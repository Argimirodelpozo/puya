### Changed

- The IR optimizer now describes what an operation or a subroutine can do with one model (`puya.ir.optimize.effects`), in which failing, reading and writing each kind of state, logging, submitting inner transactions and ending the program are separate facts. Calls are described by a summary of the called subroutine, so a call to a subroutine that does not touch some state is no longer a barrier for optimizations of that state.

### Fixed

- An AVM op that can fail is no longer removed because its result is unused, unless its operands or the code around it rule the failure out. Previously, for example, `a - b` as a statement lost its underflow error when optimized. A call whose result is unused is removed only if the called subroutine has no effect, cannot fail and has no loop.
- An assertion that a box exists is no longer removed after something that may have deleted the box: `app_box_del`, an inner transaction, or a call to a subroutine that may do either. An assertion on an exists flag that was read before the box was created or deleted is kept as well.
- A write to a box, to global state or to local state is no longer removed as overwritten when the program could tell the difference: when the two writes can differ in whether they fail (a `box_put` of another size, a state value of another type or a larger size), when the minimum balance of an account is read in between, or when an inner transaction is submitted in between.
