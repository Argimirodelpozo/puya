"""Checks the bounds of values against runs of made-up programs.

Each program is run on random inputs with a reference implementation of the AVM ops. For
every op that runs, the bounds that hold at its point are compared with the values at hand:
- the operands and the result are within their bounds: as a uint64, or as the length and the
  number of a byte array
- a byte array that is found to have no leading zero bytes has none
- an op that is found not to fail does not fail
- a comparison whose outcome is found to be fixed has that outcome
- a value that is found not to be above another is not
And for every register that is given a value, also by a phi, the bounds that hold for it
wherever it is defined: this is what checks the bounds of values defined around loops.
"""

import random
from collections import Counter
from collections.abc import Mapping

import attrs

from puya.ir import models
from puya.ir.optimize.effects import Effect, effects_of
from puya.ir.optimize.value_bounds import BoundsAtPoints, ValueBounds
from tests.utils.avm_reference import Operand
from tests.utils.ir_programs import Program, describe, run, run_op, value_of


@attrs.define
class _Checker:
    subroutine: models.Subroutine
    found: Counter[str] = attrs.field(factory=Counter)
    _everywhere: ValueBounds = attrs.field(init=False)
    _points: BoundsAtPoints = attrs.field(init=False)
    _before_op: dict[models.BasicBlock, list[ValueBounds]] = attrs.field(factory=dict, init=False)

    def __attrs_post_init__(self) -> None:
        self._everywhere = ValueBounds.of(self.subroutine)
        self._points = BoundsAtPoints.of(self.subroutine)
        for block in self.subroutine.body:
            # a copy for each op: the bounds for a block are given more from op to op
            self._before_op[block] = [
                bounds().given() for _, bounds in self._points.before_ops(block)
            ]

    def _check(self, holds: bool, claim: str, *values: object) -> None:  # noqa: FBT001
        assert holds, f"{claim.format(*map(_show, values))}\n{describe(self.subroutine)}"

    def _within(self, bounds: ValueBounds, value: models.Value, actual: Operand) -> None:
        if isinstance(actual, int):
            low, high = bounds.uint64(value)
            self._check(
                low <= actual <= high, "{} = {} is within {}..{}", value, actual, low, high
            )
            return
        low, high = bounds.length(value)
        self._check(
            low <= len(actual) <= high,
            "the length of {} = {} is within {}..{}",
            value,
            len(actual),
            low,
            high,
        )
        number = int.from_bytes(actual, "big")
        low, high = bounds.number(value)
        self._check(low <= number <= high, "{} = {} is within {}..{}", value, number, low, high)
        if bounds.is_minimal(value):
            self.found["no leading zero bytes"] += 1
            self._check(
                not actual.startswith(b"\x00"), "{} = {} has no leading zero bytes", value, actual
            )

    def before_op(
        self, block: models.BasicBlock, index: int, env: Mapping[models.Register, Operand]
    ) -> None:
        bounds = self._before_op[block][index]
        match block.ops[index]:
            case models.Assert(condition=condition):
                self._within(bounds, condition, value_of(env, condition))
            case models.Assignment(source=models.Intrinsic() as source):
                operands = [value_of(env, arg) for arg in source.args]
                for arg, operand in zip(source.args, operands, strict=True):
                    self._within(bounds, arg, operand)
                result = run_op(source, operands)
                if Effect.fails not in effects_of(source, bounds=bounds):
                    self.found["does not fail"] += 1
                    self._check(result is not None, "{} does not fail for {}", source, operands)
                outcome = bounds.outcome(source)
                if outcome is not None and result is not None:
                    self.found["outcome"] += 1
                    self._check(
                        bool(result) == outcome, "{} gives {} for {}", source, outcome, operands
                    )
                match source.args, operands:
                    case [[a, b], [int(x), int(y)]]:
                        self._ordered(bounds, a, b, x, y)
                    case [[a, b], [bytes(x), bytes(y)]]:
                        self._ordered(
                            bounds, a, b, int.from_bytes(x, "big"), int.from_bytes(y, "big")
                        )

    def _ordered(
        self, bounds: ValueBounds, a: models.Value, b: models.Value, x: int, y: int
    ) -> None:
        if bounds.not_above(a, b):
            self.found["not above"] += 1
            self._check(x <= y, "{} = {} is not above {} = {}", a, x, b, y)
        if bounds.not_above(b, a):
            self.found["not above"] += 1
            self._check(y <= x, "{} = {} is not above {} = {}", b, y, a, x)

    def defined(
        self,
        block: models.BasicBlock,
        index: int | None,
        register: models.Register,
        value: Operand,
    ) -> None:
        self._within(self._everywhere, register, value)
        if index is None:
            self._within(self._points.at_start(block), register, value)
        else:
            self._within(self._before_op[block][index], register, value)

    def before_branch(
        self, block: models.BasicBlock, env: Mapping[models.Register, Operand]
    ) -> None:
        match block.terminator:
            case models.ConditionalBranch(condition=condition):
                self._within(self._points.at_end(block), condition, value_of(env, condition))


def _show(value: object) -> str:
    match value:
        case int() if value >= 2**64:
            return hex(value)
        case bytes():
            return f"0x{value.hex()}"
        case list():
            return f"[{', '.join(map(_show, value))}]"
    return str(value)


def check_programs(rng: random.Random, programs: int, runs: int) -> Counter[str]:
    """Makes up programs and runs them, and gives what was found and checked."""
    found = Counter[str]()
    for _ in range(programs):
        program = Program(
            rng, num_variables=rng.choice([2, 3]), num_statements=rng.randrange(6, 18)
        )
        checker = _Checker(program.subroutine)
        for _ in range(runs):
            outcome = run(program.subroutine, program.random_inputs(), checker)
            found[outcome if isinstance(outcome, str) else "completes"] += 1
        found += checker.found
    return found


def test_bounds_hold_on_runs() -> None:
    found = check_programs(random.Random("value ranges"), programs=60, runs=6)
    # programs failed, completed and went around loops, and each kind of claim was checked
    for what in (
        "fails",
        "completes",
        "runs on",
        "does not fail",
        "outcome",
        "not above",
        "no leading zero bytes",
    ):
        assert found[what], what
