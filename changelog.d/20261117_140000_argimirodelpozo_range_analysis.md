### Added

- The IR optimizer derives the range of a byte array read as a number, the way byte math reads its operands, next to the range of a uint64 and of the length of a byte array (`puya.ir.optimize.value_bounds`). Values that are defined around a loop get bounds as well, and for each point of a subroutine the conditions of the asserts and branches that dominate it are taken into account.
- New optimization `range_rewrites`, which uses these ranges. A comparison whose result is fixed becomes that result, an assert that cannot fail is removed and a branch that can only go one way becomes unconditional (`range_constant_outcomes`). An operation that gives back one of its operands is replaced by that operand: `%` and `b%` of a value below the modulus, `&` and `b&` with a mask that covers the value, `b|` with zeros, and an `extract` or `substring` of a whole byte array (`range_operand_results`). Byte math is replaced by a cheaper form of the same value: `b-` for a subtraction that wraps around a modulus but cannot go below zero, uint64 ops for byte math on converted uint64 values, and one op for the last bytes of a byte array as a uint64 (`range_cheaper_forms`). Each of the three groups can be switched off by its name, like an optimization.

### Changed

- An unused `b-`, `b/` or `b%` is removed where the ranges of its operands show that it cannot fail, and not only where its operands are constants.
