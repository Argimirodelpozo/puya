"""Bounds on the values of a subroutine, from how they are defined.

Two facts are derived for a value: the range of a uint64, and the range of the length of a
byte array. Both follow SSA definitions only: a constant, the result of an operation with
bounded results, a copy, or a phi over bounded values. A bound that comes from an operation
holds once that operation has completed, which is the case wherever its result can be used.

The declared type of a register is not used as evidence: nothing checks that a value assigned
to a register of a sized type has that size. The result type of an AVM op is used, since the
AVM guarantees it.

`ValueBounds.given` adds what is known at one point of the subroutine only: what the
conditions of the asserts and branches before it say, and what it takes for the operations
around it to complete.
"""

import math
import typing
from collections.abc import Iterable, Mapping

import attrs

from puya import algo_constants
from puya.avm import AVMType
from puya.ir import models
from puya.ir._utils import get_bytes_constant
from puya.ir.avm_ops import AVMOp
from puya.ir.types_ import PrimitiveIRType

Bounds = tuple[int, int]
"""inclusive lower and upper bound"""

MAX_UINT64: typing.Final = algo_constants.MAX_UINT64
MAX_BYTES_LENGTH: typing.Final = algo_constants.MAX_BYTES_LENGTH
MAX_BIGUINT_BYTES: typing.Final = algo_constants.MAX_BIGUINT_BYTES

ANY_UINT64: typing.Final[Bounds] = (0, MAX_UINT64)
ANY_LENGTH: typing.Final[Bounds] = (0, MAX_BYTES_LENGTH)
_BOOL: typing.Final[Bounds] = (0, 1)

# how many definitions deep a bound is followed before giving up
_MAX_DEPTH: typing.Final = 48
# how many definitions deep a condition is taken apart
_MAX_CONDITION_DEPTH: typing.Final = 8

# Ops without operands that read what is fixed while a program runs: with the same immediates,
# each gives the same value every time. `global OpcodeBudget` is the exception.
_FIXED_READS: typing.Final = frozenset(
    (
        AVMOp.txn,
        AVMOp.txna,
        AVMOp.gtxn,
        AVMOp.gtxna,
        AVMOp.global_,
        AVMOp.arg,
        AVMOp.arg_0,
        AVMOp.arg_1,
        AVMOp.arg_2,
        AVMOp.arg_3,
        AVMOp.gaid,
        AVMOp.gload,
    )
)

_NEGATED: typing.Final = {
    AVMOp.eq: AVMOp.neq,
    AVMOp.neq: AVMOp.eq,
    AVMOp.lt: AVMOp.gte,
    AVMOp.gte: AVMOp.lt,
    AVMOp.gt: AVMOp.lte,
    AVMOp.lte: AVMOp.gt,
}

_Definition = tuple[models.ValueProvider | models.Phi, int]


@attrs.define
class ValueBounds:
    """Answers for one subroutine, or for constants alone when built without one."""

    _definitions: Mapping[models.Register, _Definition] = attrs.field(factory=dict)
    _uint64: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _length: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _depth: int = attrs.field(default=0, init=False)
    # what the conditions passed to `given` say, on top of the definitions
    _given_uint64: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _given_length: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    # pairs of values of which the first is not above the second
    _given_ordered: set[tuple[models.Value, models.Value]] = attrs.field(factory=set, init=False)
    # the bounds without those conditions
    _unconditional: "ValueBounds | None" = attrs.field(default=None, init=False)
    # registers that hold the same value as an earlier register, with that register
    _same: Mapping[models.Register, models.Register] = attrs.field(factory=dict, init=False)

    @classmethod
    def of(cls, subroutine: models.Subroutine) -> "ValueBounds":
        definitions = dict[models.Register, _Definition]()
        for block in subroutine.body:
            for phi in block.phis:
                definitions[phi.register] = (phi, 0)
            for op in block.ops:
                if isinstance(op, models.Assignment):
                    for index, target in enumerate(op.targets):
                        definitions[target] = (op.source, index)
        result = cls(definitions)
        result._same = _same_values(definitions)  # noqa: SLF001
        return result

    def given(
        self,
        *,
        non_zero: Iterable[models.Value] = (),
        zero: Iterable[models.Value] = (),
        completed: Iterable[models.Intrinsic] = (),
    ) -> "ValueBounds":
        """Bounds for a point of the subroutine that is only reached when each of the values
        in `non_zero` is not 0, each of the values in `zero` is 0, and each of the operations
        in `completed` has not failed. For example: the conditions of the asserts and branches
        on the way there, and the operations that ran on the way.

        All of this says something about the values that the registers hold at that point.
        A register that is assigned in a loop held other values before, and those are what a
        phi may pick up, so the bounds of a phi do not use it.
        """
        result = ValueBounds(self._definitions)
        result._unconditional = self._unconditional or self  # noqa: SLF001
        result._same = self._same  # noqa: SLF001
        conditions = [(c, True) for c in non_zero] + [(c, False) for c in zero]
        completed = list(completed)
        # twice, so that each can build on what a later one says
        for _ in range(2):
            for condition, holds in conditions:
                result._learn(condition, holds=holds, depth=0)  # noqa: SLF001
            for op in completed:
                result._learn_completed(op)  # noqa: SLF001
        return result

    def _learn_completed(self, op: models.Intrinsic) -> None:
        """Narrows the bounds of the operands of an operation to what it takes for the
        operation not to fail. This mirrors the reasons to fail in puya.ir.optimize.effects."""
        match op.op, op.args:
            case AVMOp.extract, [a]:
                start, length = op.immediates
                if isinstance(start, int) and isinstance(length, int):
                    self._narrow_length(a, (start + length, MAX_BYTES_LENGTH))
            case AVMOp.extract3, [a, b, c]:
                self._narrow_length(a, (self.uint64(b)[0] + self.uint64(c)[0], MAX_BYTES_LENGTH))
            case AVMOp.substring, [a]:
                _, end = op.immediates
                if isinstance(end, int):
                    self._narrow_length(a, (end, MAX_BYTES_LENGTH))
            case AVMOp.substring3, [a, b, c]:
                self._narrow_length(a, (self.uint64(c)[0], MAX_BYTES_LENGTH))
                self._ordered(b, c)
            case AVMOp.extract_uint16, [a, b]:
                self._narrow_length(a, (self.uint64(b)[0] + 2, MAX_BYTES_LENGTH))
            case AVMOp.extract_uint32, [a, b]:
                self._narrow_length(a, (self.uint64(b)[0] + 4, MAX_BYTES_LENGTH))
            case AVMOp.extract_uint64, [a, b]:
                self._narrow_length(a, (self.uint64(b)[0] + 8, MAX_BYTES_LENGTH))
            case ((AVMOp.getbyte | AVMOp.setbyte), [a, b, *_]):
                self._narrow_length(a, (self.uint64(b)[0] + 1, MAX_BYTES_LENGTH))
            case AVMOp.btoi, [a]:
                self._narrow_length(a, (0, 8))
            case AVMOp.concat, [a, b]:
                self._narrow_length(a, (0, MAX_BYTES_LENGTH - self.length(b)[0]))
                self._narrow_length(b, (0, MAX_BYTES_LENGTH - self.length(a)[0]))
            case (
                (
                    AVMOp.add_bytes
                    | AVMOp.sub_bytes
                    | AVMOp.mul_bytes
                    | AVMOp.div_floor_bytes
                    | AVMOp.mod_bytes
                    | AVMOp.bsqrt
                    | AVMOp.eq_bytes
                    | AVMOp.neq_bytes
                    | AVMOp.lt_bytes
                    | AVMOp.lte_bytes
                    | AVMOp.gt_bytes
                    | AVMOp.gte_bytes
                ),
                args,
            ):
                for arg in args:
                    self._narrow_length(arg, (0, MAX_BIGUINT_BYTES))
            case AVMOp.bzero, [a]:
                self._narrow(a, (0, MAX_BYTES_LENGTH))
            case ((AVMOp.shl | AVMOp.shr), [_, b]):
                self._narrow(b, (0, 63))
            case ((AVMOp.div_floor | AVMOp.mod), [_, b]):
                self._narrow(b, (1, MAX_UINT64))
            case AVMOp.add, [a, b]:
                self._narrow(a, (0, MAX_UINT64 - self.uint64(b)[0]))
                self._narrow(b, (0, MAX_UINT64 - self.uint64(a)[0]))
            case AVMOp.mul, [a, b]:
                # a factor of 0 does not limit the other one
                if b_low := self.uint64(b)[0]:
                    self._narrow(a, (0, MAX_UINT64 // b_low))
                if a_low := self.uint64(a)[0]:
                    self._narrow(b, (0, MAX_UINT64 // a_low))
            case AVMOp.sub, [a, b]:
                self._narrow(a, (self.uint64(b)[0], MAX_UINT64))
                self._narrow(b, (0, self.uint64(a)[1]))
                self._ordered(b, a)
            case ((AVMOp.loads | AVMOp.stores), [a, *_]):
                self._narrow(a, (0, 255))

    def _ordered(self, low: models.Value, high: models.Value) -> None:
        self._given_ordered.add((self.same_value(low), self.same_value(high)))

    def _narrow_length(self, value: models.Value, bounds: Bounds) -> None:
        value = self.same_value(value)
        if not isinstance(value, models.Register):
            return
        self._given_length[value] = _intersection(
            self._given_length.get(value, ANY_LENGTH), bounds
        )
        # what was derived so far may be narrower now
        self._uint64.clear()
        self._length.clear()

    def _learn(self, condition: models.Value, *, holds: bool, depth: int) -> None:
        if not isinstance(condition, models.Register):
            return
        self._narrow(condition, (1, MAX_UINT64) if holds else (0, 0))
        definition = self._definitions.get(condition)
        if definition is None or depth >= _MAX_CONDITION_DEPTH:
            return
        source, _ = definition
        match source:
            case models.Register():
                self._learn(source, holds=holds, depth=depth + 1)
            case models.Intrinsic(op=AVMOp.not_, args=[a]):
                self._learn(a, holds=not holds, depth=depth + 1)
            case models.Intrinsic(op=AVMOp.and_, args=[a, b]) if holds:
                self._learn(a, holds=True, depth=depth + 1)
                self._learn(b, holds=True, depth=depth + 1)
            case models.Intrinsic(op=AVMOp.or_, args=[a, b]) if not holds:
                self._learn(a, holds=False, depth=depth + 1)
                self._learn(b, holds=False, depth=depth + 1)
            case models.Intrinsic(op=comparison, args=[a, b]) if (
                comparison in _NEGATED and _is_uint64(a) and _is_uint64(b)
            ):
                self._learn_comparison(comparison if holds else _NEGATED[comparison], a, b)

    def _learn_comparison(self, comparison: AVMOp, a: models.Value, b: models.Value) -> None:
        a_low, a_high = self.uint64(a)
        b_low, b_high = self.uint64(b)
        match comparison:
            case AVMOp.eq:
                both = max(a_low, b_low), min(a_high, b_high)
                self._narrow(a, both)
                self._narrow(b, both)
                self._ordered(a, b)
                self._ordered(b, a)
            case AVMOp.lt:
                self._narrow(a, (0, b_high - 1))
                self._narrow(b, (a_low + 1, MAX_UINT64))
                self._ordered(a, b)
            case AVMOp.lte:
                self._narrow(a, (0, b_high))
                self._narrow(b, (a_low, MAX_UINT64))
                self._ordered(a, b)
            case AVMOp.gt:
                self._narrow(a, (b_low + 1, MAX_UINT64))
                self._narrow(b, (0, a_high - 1))
                self._ordered(b, a)
            case AVMOp.gte:
                self._narrow(a, (b_low, MAX_UINT64))
                self._narrow(b, (0, a_high))
                self._ordered(b, a)

    def _narrow(self, value: models.Value, bounds: Bounds) -> None:
        value = self.same_value(value)
        if not isinstance(value, models.Register):
            return
        self._given_uint64[value] = _intersection(
            self._given_uint64.get(value, ANY_UINT64), bounds
        )
        # a bound on the length of a byte array is a bound on the byte array
        match self._intrinsic(value):
            case models.Intrinsic(op=AVMOp.len_, args=[array]):
                self._narrow_length(array, (bounds[0], min(bounds[1], MAX_BYTES_LENGTH)))
        # what was derived so far may be narrower now
        self._uint64.clear()
        self._length.clear()

    def operands_of_definition(self, value: models.Value) -> Iterable[models.Register]:
        """The registers that a value is computed from, if it is the result of an AVM op."""
        op = self._intrinsic(value)
        if op is None:
            return ()
        return [arg for arg in op.args if isinstance(arg, models.Register)]

    def same_value(self, value: models.Value) -> models.Value:
        """The first register that is known to hold the same value as the given one, which
        stands for all of them."""
        if isinstance(value, models.Register):
            return self._same.get(value, value)
        return value

    def is_fixed_read(self, register: models.Register) -> bool:
        """True if the register holds something that is fixed while the program runs, read by
        an op that gives the same outcome every time."""
        return register in self._same

    def not_above(self, a: models.Value, b: models.Value) -> bool:
        """True if the uint64 value `a` is known to be at most the uint64 value `b`."""
        a = self.same_value(a)
        b = self.same_value(b)
        return a == b or (a, b) in self._given_ordered or self.uint64(a)[1] <= self.uint64(b)[0]

    def sum_within(self, a: models.Value, b: models.Value, array: models.Value) -> bool:
        """True if the sum of the uint64 values `a` and `b` is known to be at most the length
        of a byte array."""
        if self.uint64(a)[1] + self.uint64(b)[1] <= self.length(array)[0]:
            return True
        array = self.same_value(array)
        # a register that holds the sum, and is not above one that holds the length
        for total, length in self._given_ordered:
            match self._intrinsic(total), self._intrinsic(length):
                case (
                    models.Intrinsic(op=AVMOp.add, args=[x, y]),
                    models.Intrinsic(op=AVMOp.len_, args=[measured]),
                ) if (x, y) in ((a, b), (b, a)) and self.same_value(measured) == array:
                    return True
        return False

    def _intrinsic(self, value: models.Value) -> models.Intrinsic | None:
        """The AVM op that a value is the result of."""
        if isinstance(value, models.Register):
            match self._definitions.get(value):
                case (models.Intrinsic() as op, _):
                    return op
        return None

    def uint64(self, value: models.Value) -> Bounds:
        """The range of a uint64 value."""
        if isinstance(value, models.UInt64Constant):
            return value.value, value.value
        if not isinstance(value, models.Register):
            return ANY_UINT64
        given = self._given_uint64.get(self._same.get(value, value), ANY_UINT64)
        if value not in self._definitions:
            return given
        try:
            return self._uint64[value]
        except KeyError:
            pass
        # until this definition is resolved, a definition that depends on itself sees no bound
        self._uint64[value] = given
        result = self._bounded(value, ANY_UINT64, self._uint64_of_definition)
        result = self._uint64[value] = _intersection(result, given)
        return result

    def length(self, value: models.Value) -> Bounds:
        """The range of the length of a byte array value."""
        if isinstance(value, models.Constant):
            constant = get_bytes_constant(value)
            if constant is None:
                return ANY_LENGTH
            return len(constant), len(constant)
        if not isinstance(value, models.Register):
            return ANY_LENGTH
        given = self._given_length.get(self._same.get(value, value), ANY_LENGTH)
        if value not in self._definitions:
            return given
        try:
            return self._length[value]
        except KeyError:
            pass
        self._length[value] = given
        result = self._bounded(value, ANY_LENGTH, self._length_of_definition)
        result = self._length[value] = _intersection(result, given)
        return result

    def _bounded(
        self,
        register: models.Register,
        unknown: Bounds,
        of_definition: typing.Callable[[models.ValueProvider | models.Phi, int], Bounds],
    ) -> Bounds:
        definition = self._definitions.get(register)
        if definition is None or self._depth >= _MAX_DEPTH:
            return unknown
        self._depth += 1
        try:
            low, high = of_definition(*definition)
        finally:
            self._depth -= 1
        low = max(low, unknown[0])
        high = min(high, unknown[1])
        # a definition that can never complete may produce an empty range
        if low > high:
            return unknown
        return low, high

    def _uint64_of_definition(
        self, source: models.ValueProvider | models.Phi, index: int
    ) -> Bounds:
        match source:
            case models.Value():
                return self.uint64(source)
            case models.ValueTuple(values=values):
                return self.uint64(values[index])
            case models.Phi(args=args):
                unconditional = self._unconditional or self
                return _union(unconditional.uint64(arg.value) for arg in args)
            case models.Intrinsic() if len(source.types) == 1:
                return self._uint64_of_intrinsic(source)
            case _:
                return ANY_UINT64

    def _length_of_definition(
        self, source: models.ValueProvider | models.Phi, index: int
    ) -> Bounds:
        match source:
            case models.Value():
                return self.length(source)
            case models.ValueTuple(values=values):
                return self.length(values[index])
            case models.Phi(args=args):
                unconditional = self._unconditional or self
                return _union(unconditional.length(arg.value) for arg in args)
            case models.Intrinsic() if len(source.types) == 1:
                return self._length_of_intrinsic(source)
            case _:
                return ANY_LENGTH

    def _uint64_of_intrinsic(self, op: models.Intrinsic) -> Bounds:
        match op.op, op.args:
            case AVMOp.len_, [a]:
                return self.length(a)
            case AVMOp.btoi, [a]:
                _, max_length = self.length(a)
                return 0, 256 ** min(max_length, 8) - 1
            case AVMOp.getbyte, _:
                return 0, 255
            case AVMOp.extract_uint16, _:
                return 0, 2**16 - 1
            case AVMOp.extract_uint32, _:
                return 0, 2**32 - 1
            case AVMOp.add, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                return a_low + b_low, a_high + b_high
            case AVMOp.sub, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                return max(a_low - b_high, 0), a_high - b_low
            case AVMOp.mul, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                return a_low * b_low, a_high * b_high
            case AVMOp.div_floor, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                if b_high == 0:
                    return ANY_UINT64
                return a_low // b_high, a_high // max(b_low, 1)
            case AVMOp.mod, [a, b]:
                _, a_high = self.uint64(a)
                _, b_high = self.uint64(b)
                if b_high == 0:
                    return ANY_UINT64
                return 0, min(a_high, b_high - 1)
            case AVMOp.bitwise_and, [a, b]:
                return 0, min(self.uint64(a)[1], self.uint64(b)[1])
            case AVMOp.bitwise_or, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                return max(a_low, b_low), _all_bits(max(a_high, b_high))
            case AVMOp.bitwise_xor, [a, b]:
                return 0, _all_bits(max(self.uint64(a)[1], self.uint64(b)[1]))
            case AVMOp.shr, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                return a_low >> min(b_high, 63), a_high >> min(b_low, 63)
            case AVMOp.shl, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                # the result is taken modulo 2**64, so a shift that can overflow says nothing
                if a_high << min(b_high, 63) > MAX_UINT64:
                    return ANY_UINT64
                return a_low << min(b_low, 63), a_high << min(b_high, 63)
            case AVMOp.sqrt, [a]:
                a_low, a_high = self.uint64(a)
                return math.isqrt(a_low), math.isqrt(a_high)
            case AVMOp.bitlen, [a]:
                if _is_uint64(a):
                    a_low, a_high = self.uint64(a)
                    return a_low.bit_length(), a_high.bit_length()
                return 0, 8 * self.length(a)[1]
            case AVMOp.select, [a, b, _] if _is_uint64(a) and _is_uint64(b):
                return _union((self.uint64(a), self.uint64(b)))
            case _:
                (result_type,) = op.op_signature.returns
                if result_type == PrimitiveIRType.bool:
                    return _BOOL
                return ANY_UINT64

    def _length_of_intrinsic(self, op: models.Intrinsic) -> Bounds:
        match op.op, op.args:
            case AVMOp.concat, [a, b]:
                a_min, a_max = self.length(a)
                b_min, b_max = self.length(b)
                return a_min + b_min, a_max + b_max
            case AVMOp.bzero, [a]:
                return self.uint64(a)
            case AVMOp.extract, [a]:
                start, length = op.immediates
                assert isinstance(start, int)
                assert isinstance(length, int)
                if length:
                    return length, length
                a_min, a_max = self.length(a)
                return max(a_min - start, 0), a_max - start
            case AVMOp.extract3, [a, _, c]:
                c_low, c_high = self.uint64(c)
                return c_low, min(c_high, self.length(a)[1])
            case AVMOp.substring, [_]:
                start, end = op.immediates
                assert isinstance(start, int)
                assert isinstance(end, int)
                return end - start, end - start
            case AVMOp.substring3, [a, b, c]:
                b_low, b_high = self.uint64(b)
                c_low, c_high = self.uint64(c)
                return max(c_low - b_high, 0), min(c_high - b_low, self.length(a)[1])
            case ((AVMOp.replace2 | AVMOp.replace3 | AVMOp.setbyte), [a, *_]):
                return self.length(a)
            case AVMOp.setbit, [a, *_] if _is_bytes(a):
                return self.length(a)
            case AVMOp.bitwise_not_bytes, [a]:
                return self.length(a)
            case (
                (AVMOp.bitwise_or_bytes | AVMOp.bitwise_and_bytes | AVMOp.bitwise_xor_bytes),
                [a, b],
            ):
                a_min, a_max = self.length(a)
                b_min, b_max = self.length(b)
                return max(a_min, b_min), max(a_max, b_max)
            # the results of byte math have no leading zero bytes, and its inputs are at most
            # MAX_BIGUINT_BYTES long whenever there is a result
            case AVMOp.add_bytes, [a, b]:
                return 0, max(self._math_length(a), self._math_length(b)) + 1
            case AVMOp.mul_bytes, [a, b]:
                return 0, self._math_length(a) + self._math_length(b)
            case ((AVMOp.sub_bytes | AVMOp.div_floor_bytes), [a, _]):
                return 0, self._math_length(a)
            case AVMOp.mod_bytes, [_, b]:
                return 0, self._math_length(b)
            case AVMOp.bsqrt, [a]:
                return 0, (self._math_length(a) + 1) // 2
            case AVMOp.select, [a, b, _] if _is_bytes(a) and _is_bytes(b):
                return _union((self.length(a), self.length(b)))
            case _:
                (result_type,) = op.op_signature.returns
                if result_type.maybe_avm_type is AVMType.bytes:
                    num_bytes = result_type.num_bytes
                    if num_bytes is not None:
                        return num_bytes, num_bytes
                return ANY_LENGTH

    def _math_length(self, value: models.Value) -> int:
        return min(self.length(value)[1], MAX_BIGUINT_BYTES)


def _union(bounds: typing.Iterable[Bounds]) -> Bounds:
    lows, highs = list[int](), list[int]()
    for low, high in bounds:
        lows.append(low)
        highs.append(high)
    if not lows:
        # nothing to choose from, so nothing is claimed: wider than any value
        return 0, 2**512
    return min(lows), max(highs)


def _same_values(
    definitions: Mapping[models.Register, _Definition],
) -> Mapping[models.Register, models.Register]:
    same = dict[models.Register, models.Register]()
    first = dict[object, models.Register]()
    for register, (source, _) in definitions.items():
        match source:
            case models.Intrinsic(op=op, immediates=immediates, args=[]) if (
                op in _FIXED_READS and immediates != ["OpcodeBudget"] and len(source.types) == 1
            ):
                same[register] = first.setdefault((op, tuple(immediates)), register)
    return same


def _intersection(a: Bounds, b: Bounds) -> Bounds:
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    if low > high:
        # no value is in both: the program does not get to where both hold, so either will do
        return a
    return low, high


def _is_uint64(value: models.Value) -> bool:
    return value.ir_type.maybe_avm_type is AVMType.uint64


def _is_bytes(value: models.Value) -> bool:
    return value.ir_type.maybe_avm_type is AVMType.bytes


def _all_bits(value: int) -> int:
    return (1 << value.bit_length()) - 1
