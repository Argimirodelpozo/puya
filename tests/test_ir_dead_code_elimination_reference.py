"""Checks dead code elimination against a reference implementation of the AVM ops that compute
values.

Blocks of operations are made up at random: ops on the parameters, on constants and on earlier
results, with asserts in between. Some results are used, the others are for dead code
elimination to deal with. Each block is then run on random inputs as it is and as it is
optimized: both have to fail for the same inputs, and to give the same values otherwise.
"""

import random
from collections.abc import Mapping, Sequence
from unittest.mock import Mock

import attrs

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.dead_code_elimination import remove_unused_variables
from puya.ir.types_ import PrimitiveIRType
from tests.utils.avm_reference import (
    REFERENCE_OPS,
    Operand,
    ReferenceOp,
    random_uint64,
)
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    intrinsic,
    parameter,
    register,
    return_,
    subroutine,
    uint64,
)

_UINT64_CONSTANTS = [0, 1, 2, 7, 8, 9, 24, 31, 32, 33, 63, 64, 65]
_BYTES_CONSTANTS = [b"", b"\x01", bytes(8), bytes(32), bytes(64)]
_KEY = bytes_(b"key")
# the ops that fail for some operands: an unused one of these is what the tests are about
_FALLIBLE_OPS = [
    spec
    for spec in REFERENCE_OPS
    if spec.op
    in (
        AVMOp.add,
        AVMOp.sub,
        AVMOp.mul,
        AVMOp.div_floor,
        AVMOp.shl,
        AVMOp.btoi,
        AVMOp.concat,
        AVMOp.bzero,
        AVMOp.extract,
        AVMOp.extract3,
        AVMOp.substring3,
        AVMOp.extract_uint16,
        AVMOp.extract_uint64,
        AVMOp.getbyte,
        AVMOp.add_bytes,
    )
]
_COMPARISONS = [spec for spec in REFERENCE_OPS if spec.op in (AVMOp.lt, AVMOp.gte, AVMOp.eq)]


@attrs.define
class _Block:
    """A block of ops on few values, so that what one op says about a value matters to others."""

    rng: random.Random
    parameters: list[models.Parameter] = attrs.field(factory=list)
    uint64s: list[models.Value] = attrs.field(factory=list)
    byte_arrays: list[models.Value] = attrs.field(factory=list)
    results: list[models.Register] = attrs.field(factory=list)
    conditions: list[models.Register] = attrs.field(factory=list)
    ops: list[models.Op] = attrs.field(factory=list)
    specs: dict[models.Intrinsic, ReferenceOp] = attrs.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        number = parameter("number")
        data = parameter("data", PrimitiveIRType.bytes)
        self.parameters += [number, data]
        self.uint64s.append(number)
        self.byte_arrays.append(data)
        (length,) = (spec for spec in REFERENCE_OPS if spec.op == AVMOp.len_)
        self._add_op(length)
        for _ in range(self.rng.randint(3, 12)):
            choice = self.rng.random()
            if choice < 0.15:
                self._add_op(self.rng.choice(_COMPARISONS), is_condition=True)
            elif choice < 0.3 and self.conditions:
                self.ops.append(
                    models.Assert(
                        condition=self.rng.choice(self.conditions),
                        message=None,
                        explicit=True,
                        source_location=None,
                    )
                )
            elif choice < 0.9:
                self._add_op(self.rng.choice(_FALLIBLE_OPS))
            else:
                self._add_op(self.rng.choice(REFERENCE_OPS))
        # some of the results are used
        for result in self.results:
            if self.rng.random() < 0.3:
                self.ops.append(intrinsic(AVMOp.app_global_put, _KEY, result))

    def _operand(self, kind: str) -> models.Value:
        if kind == "u":
            if self.rng.random() < 0.4:
                return uint64(self.rng.choice(_UINT64_CONSTANTS))
            return self.rng.choice(self.uint64s[:4])
        if self.rng.random() < 0.1:
            return bytes_(self.rng.choice(_BYTES_CONSTANTS))
        return self.rng.choice(self.byte_arrays[:3])

    def _add_op(self, spec: ReferenceOp, *, is_condition: bool = False) -> None:
        op = intrinsic(
            spec.op,
            *(self._operand(kind) for kind in spec.operands),
            immediates=spec.immediates(self.rng),
            types=[spec.result],
        )
        result = register(f"result{len(self.results)}", spec.result)
        self.specs[op] = spec
        self.results.append(result)
        self.ops.append(assign(result, op))
        if is_condition:
            self.conditions.append(result)
        elif spec.result == PrimitiveIRType.uint64:
            self.uint64s.append(result)
        else:
            self.byte_arrays.append(result)

    def random_inputs(self) -> dict[models.Value, Operand]:
        number, data = self.parameters
        # mostly around the constants that the ops use, where they start to fail
        length = self.rng.choice(
            [self.rng.randrange(70), self.rng.randrange(70), self.rng.randrange(4000, 4097)]
        )
        inputs: dict[models.Value, Operand] = {
            data: bytes([self.rng.choice([0, 0, 1, 255])]) * length
        }
        inputs[number] = self.rng.choice(
            [
                self.rng.randrange(70),
                self.rng.randrange(70),
                max(length - self.rng.randrange(10), 0),
                2**64 - 1 - self.rng.randrange(70),
                random_uint64(self.rng),
            ]
        )
        return inputs

    def run(
        self, ops: Sequence[models.Op], inputs: Mapping[models.Value, Operand]
    ) -> list[Operand] | None:
        """The values that the block puts in global state, or None if the block fails."""
        values = dict(inputs)

        def value_of(value: models.Value) -> Operand:
            match value:
                case models.UInt64Constant(value=number):
                    return number
                case models.BytesConstant(value=data):
                    return data
                case _:
                    return values[value]

        used = list[Operand]()
        for op in ops:
            match op:
                case models.Assignment(targets=[target], source=models.Intrinsic() as source):
                    result = self.specs[source].run(
                        [int(i) for i in source.immediates], [value_of(a) for a in source.args]
                    )
                    if result is None:
                        return None
                    values[target] = result
                case models.Assert(condition=condition):
                    if not value_of(condition):
                        return None
                case models.Intrinsic(op=AVMOp.app_global_put, args=[_, value]):
                    used.append(value_of(value))
                case _:
                    raise AssertionError(f"unexpected op: {op}")
        return used


def _describe(
    inputs: Mapping[models.Value, Operand],
    original: Sequence[models.Op],
    optimized: Sequence[models.Op],
) -> str:
    return "\n".join(
        (
            "inputs:",
            *(f"  {param} = {value!r}" for param, value in inputs.items()),
            "original:",
            *(f"  {op}" for op in original),
            "optimized:",
            *(f"  {op}" for op in optimized),
        )
    )


def test_dead_code_elimination_agrees_with_reference() -> None:
    rng = random.Random("dead code elimination")
    removed = kept = fails = completes = 0
    for _ in range(1500):
        case = _Block(rng)
        original = list(case.ops)
        sub = subroutine("main", [block(0, case.ops, return_())], parameters=case.parameters)
        context = Mock(effects=None)
        while remove_unused_variables(context, sub):
            pass
        optimized = sub.body[0].ops
        sub.validate_with_ssa()
        removed += len(original) - len(optimized)
        kept += sum(isinstance(op, models.Assignment) for op in optimized)
        for _ in range(40):
            inputs = case.random_inputs()
            expected = case.run(original, inputs)
            if expected is None:
                fails += 1
            else:
                completes += 1
            assert case.run(optimized, inputs) == expected, _describe(inputs, original, optimized)
    # operations were removed and operations were kept, blocks failed and blocks completed
    assert removed
    assert kept
    assert fails
    assert completes
