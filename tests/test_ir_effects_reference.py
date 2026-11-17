"""Checks the effect model against a reference implementation of the AVM ops that compute values.

For operands picked at random, and bounds picked at random around them:
- an op that is found not to fail does not fail
- the bounds of its result hold the result
- what is learned from the op having completed holds for the operands
"""

import random
from collections.abc import Sequence

import attrs
import pytest

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.effects import Effect, effects_of
from puya.ir.optimize.value_bounds import ValueBounds
from puya.ir.types_ import PrimitiveIRType
from tests.utils.avm_reference import (
    MAX_BYTES,
    MAX_UINT64,
    REFERENCE_OPS,
    Operand,
    ReferenceOp,
    random_bytes,
    random_uint64,
)
from tests.utils.ir import (
    assign,
    block,
    intrinsic,
    parameter,
    register,
    return_,
    subroutine,
    uint64,
)


def _random_bounds(rng: random.Random, value: int, maximum: int) -> tuple[int, int]:
    """Bounds that the value is within: exact, loose on either side, or none."""
    match rng.randrange(6):
        case 0 | 1:
            return value, value
        case 1:
            return 0, value + rng.choice([0, 1, 8, 64, maximum])
        case 2:
            return max(value - rng.choice([0, 1, 8, 64]), 0), maximum
        case 3:
            return max(value - rng.choice([0, 1, 8]), 0), value + rng.choice([0, 1, 8])
        case _:
            return 0, maximum


@attrs.define
class _Case:
    """An op on parameters, with conditions that bound the parameters."""

    rng: random.Random
    spec: ReferenceOp
    values: list[Operand] = attrs.field(factory=list)
    parameters: list[models.Parameter] = attrs.field(factory=list)
    ops: list[models.Op] = attrs.field(factory=list)
    conditions: list[models.Value] = attrs.field(factory=list)
    immediates: Sequence[int] = ()
    op: models.Intrinsic = attrs.field(init=False)
    result: models.Register = attrs.field(init=False)
    bounds: ValueBounds = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        for index, kind in enumerate(self.spec.operands):
            if kind == "u":
                value: Operand = random_uint64(self.rng, self.values)
                param = parameter(f"p{index}")
                self._bound(param, value, MAX_UINT64)
            else:
                value = random_bytes(self.rng, self.values)
                param = parameter(f"p{index}", PrimitiveIRType.bytes)
                length = register(f"length{index}")
                self.ops.append(assign(length, intrinsic(AVMOp.len_, param)))
                self._bound(length, len(value), MAX_BYTES)
            self.values.append(value)
            self.parameters.append(param)
        self.immediates = self.spec.immediates(self.rng)
        self.op = intrinsic(
            self.spec.op, *self.parameters, immediates=self.immediates, types=[self.spec.result]
        )
        self.result = register("result", self.spec.result)
        self.ops.append(assign(self.result, self.op))
        sub = subroutine("case", [block(0, self.ops, return_())], parameters=self.parameters)
        self.bounds = ValueBounds.of(sub)

    def _bound(self, value: models.Register, actual: int, maximum: int) -> None:
        low, high = _random_bounds(self.rng, actual, maximum)
        for name, op, bound in (("low", AVMOp.gte, low), ("high", AVMOp.lte, high)):
            condition = register(f"{value.name}_{name}", PrimitiveIRType.bool)
            self.ops.append(assign(condition, intrinsic(op, value, uint64(min(bound, maximum)))))
            self.conditions.append(condition)

    def run(self) -> Operand | None:
        return self.spec.run(self.immediates, self.values)

    def holds(self, bounds: ValueBounds, value: models.Value, actual: Operand) -> bool:
        if isinstance(actual, int):
            low, high = bounds.uint64(value)
            return low <= actual <= high
        low, high = bounds.length(value)
        return low <= len(actual) <= high

    def describe(self) -> str:
        return f"{self.op} with {self.values!r} given {[str(o) for o in self.ops]}"


@pytest.mark.parametrize("spec", REFERENCE_OPS, ids=str)
def test_model_agrees_with_reference(spec: ReferenceOp) -> None:
    rng = random.Random(f"{spec}")
    outcomes = set[bool]()
    found_not_to_fail = 0
    for _ in range(500):
        case = _Case(rng, spec)
        result = case.run()
        outcomes.add(result is not None)
        for given in (
            case.bounds,
            case.bounds.given(non_zero=case.conditions),
        ):
            # an op that is found not to fail does not fail
            if Effect.fails not in effects_of(case.op, bounds=given):
                found_not_to_fail += 1
                assert result is not None, case.describe()
            if result is not None:
                # the bounds of the result hold the result
                assert case.holds(given, case.result, result), case.describe()
                # as do the bounds of the operands
                for param, value in zip(case.parameters, case.values, strict=True):
                    assert case.holds(given, param, value), case.describe()
        if result is not None:
            # what is learned from the op having completed holds for its operands
            for learned in (
                case.bounds.given(completed=[case.op]),
                case.bounds.given(non_zero=case.conditions, completed=[case.op]),
            ):
                for param, value in zip(case.parameters, case.values, strict=True):
                    assert case.holds(learned, param, value), case.describe()
                assert case.holds(learned, case.result, result), case.describe()
    # the operands and bounds picked cover what the op does and what the model can tell
    assert True in outcomes, f"{spec} never completed"
    assert found_not_to_fail or not spec.decided_by_bounds, f"{spec} was never found not to fail"
