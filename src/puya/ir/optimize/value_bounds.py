"""Bounds on the values of a subroutine, from how they are defined.

Three facts are derived for a value: the range of a uint64, the range of the length of a byte
array, and the range of a byte array read as an unsigned big-endian number, which is how byte
math and its comparisons read their operands. All follow SSA definitions only: a constant, the
result of an operation with bounded results, a copy, or a phi over bounded values. A bound
that comes from an operation holds once that operation has completed, which is the case
wherever its result can be used.

The declared type of a register is not used as evidence: nothing checks that a value assigned
to a register of a sized type has that size. The result type of an AVM op is used, since the
AVM guarantees it.

The number and the length of a byte array are separate facts. 32 zero bytes are the number 0
and are still too long for `btoi`, and a small number says nothing about the length of the
byte array that holds it. The exception are the results of byte math (`b+`, `b-`, `b*`, `b/`,
`b%`, `bsqrt`): the AVM encodes them without leading zero bytes, so their length follows from
their number. `ValueBounds.is_minimal` tells which values are known to be encoded that way.
Two such values that are the same number are the same bytes.

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
ANY_NUMBER: typing.Final[Bounds] = (0, 256**MAX_BYTES_LENGTH - 1)
_BOOL: typing.Final[Bounds] = (0, 1)
# the largest number that byte math takes as an operand
_MAX_MATH_NUMBER: typing.Final = 256**MAX_BIGUINT_BYTES - 1
_MAX_BITS: typing.Final = 8 * MAX_BYTES_LENGTH

# how many definitions deep a bound is followed before giving up
_MAX_DEPTH: typing.Final = 48
# how many definitions deep a condition is taken apart
_MAX_CONDITION_DEPTH: typing.Final = 8
# how many ordered pairs, and how many parts taken of byte arrays, are kept for one point
_MAX_RELATIONS: typing.Final = 64

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
# the comparisons of byte math, with the uint64 comparison that does the same to numbers
_NUMBER_COMPARISONS: typing.Final = {
    AVMOp.eq_bytes: AVMOp.eq,
    AVMOp.neq_bytes: AVMOp.neq,
    AVMOp.lt_bytes: AVMOp.lt,
    AVMOp.gte_bytes: AVMOp.gte,
    AVMOp.gt_bytes: AVMOp.gt,
    AVMOp.lte_bytes: AVMOp.lte,
}
# byte math: the operands are read as numbers, and the result has no leading zero bytes
_BYTE_MATH: typing.Final = frozenset(
    (
        AVMOp.add_bytes,
        AVMOp.sub_bytes,
        AVMOp.mul_bytes,
        AVMOp.div_floor_bytes,
        AVMOp.mod_bytes,
        AVMOp.bsqrt,
    )
)

_Definition = tuple[models.ValueProvider | models.Phi, int]


@attrs.define
class ValueBounds:
    """Answers for one subroutine, or for constants alone when built without one."""

    _definitions: Mapping[models.Register, _Definition] = attrs.field(factory=dict)
    _uint64: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _length: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _number: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    # how a value is encoded does not depend on conditions, so this is shared with `given`
    _minimal: dict[models.Register, bool] = attrs.field(factory=dict, init=False)
    _depth: int = attrs.field(default=0, init=False)
    # what the conditions passed to `given` say, on top of the definitions
    _given_uint64: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _given_length: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    _given_number: dict[models.Register, Bounds] = attrs.field(factory=dict, init=False)
    # pairs of values of which the first is not above the second: two uint64 values, or two
    # byte arrays read as numbers
    _given_ordered: set[tuple[models.Value, models.Value]] = attrs.field(factory=set, init=False)
    # a byte array, a start and a size, where the start plus the size is within the byte array
    _given_within: set[tuple[models.Value, models.Value, models.Value | int]] = attrs.field(
        factory=set, init=False
    )
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
        on the way there, and the operations that ran on the way. Called on bounds that are
        themselves given something, this adds to it.

        All of this says something about the values that the registers hold at that point.
        A register that is assigned in a loop held other values before, and those are what a
        phi may pick up, so the bounds of a phi do not use it.
        """
        result = ValueBounds(self._definitions)
        result._unconditional = self._unconditional or self  # noqa: SLF001
        result._same = self._same  # noqa: SLF001
        result._minimal = self._minimal  # noqa: SLF001
        result._given_uint64 = dict(self._given_uint64)  # noqa: SLF001
        result._given_length = dict(self._given_length)  # noqa: SLF001
        result._given_number = dict(self._given_number)  # noqa: SLF001
        result._given_ordered = set(self._given_ordered)  # noqa: SLF001
        result._given_within = set(self._given_within)  # noqa: SLF001
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
                self._within(a, b, c)
            case AVMOp.substring, [a]:
                _, end = op.immediates
                if isinstance(end, int):
                    self._narrow_length(a, (end, MAX_BYTES_LENGTH))
            case AVMOp.substring3, [a, b, c]:
                self._narrow_length(a, (self.uint64(c)[0], MAX_BYTES_LENGTH))
                self._ordered(b, c)
                self._within(a, c, 0)
            case AVMOp.extract_uint16, [a, b]:
                self._narrow_length(a, (self.uint64(b)[0] + 2, MAX_BYTES_LENGTH))
                self._within(a, b, 2)
            case AVMOp.extract_uint32, [a, b]:
                self._narrow_length(a, (self.uint64(b)[0] + 4, MAX_BYTES_LENGTH))
                self._within(a, b, 4)
            case AVMOp.extract_uint64, [a, b]:
                self._narrow_length(a, (self.uint64(b)[0] + 8, MAX_BYTES_LENGTH))
                self._within(a, b, 8)
            case ((AVMOp.getbyte | AVMOp.setbyte), [a, b, *_]):
                self._narrow_length(a, (self.uint64(b)[0] + 1, MAX_BYTES_LENGTH))
                self._within(a, b, 1)
            case AVMOp.btoi, [a]:
                self._narrow_length(a, (0, 8))
            case AVMOp.concat, [a, b]:
                self._narrow_length(a, (0, MAX_BYTES_LENGTH - self.length(b)[0]))
                self._narrow_length(b, (0, MAX_BYTES_LENGTH - self.length(a)[0]))
            case op_code, args if op_code in _BYTE_MATH or op_code in _NUMBER_COMPARISONS:
                for arg in args:
                    self._narrow_length(arg, (0, MAX_BIGUINT_BYTES))
                match op_code, args:
                    case AVMOp.sub_bytes, [a, b]:
                        self._narrow_number(a, (self.number(b)[0], ANY_NUMBER[1]))
                        self._narrow_number(b, (0, self.number(a)[1]))
                        self._ordered(b, a)
                    case ((AVMOp.div_floor_bytes | AVMOp.mod_bytes), [_, b]):
                        self._narrow_number(b, (1, ANY_NUMBER[1]))
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
        pair = (self.same_value(low), self.same_value(high))
        if len(self._given_ordered) < _MAX_RELATIONS and pair not in self._given_ordered:
            self._given_ordered.add(pair)
            # the outcome of a comparison of the two may be fixed now
            self._forget_derived()

    def _within(self, array: models.Value, start: models.Value, size: models.Value | int) -> None:
        if len(self._given_within) < _MAX_RELATIONS:
            self._given_within.add(
                (self.same_value(array), self.same_value(start), self._plain(size))
            )

    def _plain(self, value: models.Value | int) -> models.Value | int:
        if isinstance(value, models.UInt64Constant):
            return value.value
        if isinstance(value, models.Value):
            return self.same_value(value)
        return value

    def _covers(self, known: models.Value | int, size: models.Value | int) -> bool:
        """True if the first of two sizes is at least the second."""
        if known == size:
            return True
        known_low = known if isinstance(known, int) else self.uint64(known)[0]
        size_high = size if isinstance(size, int) else self.uint64(size)[1]
        return known_low >= size_high

    def _forget_derived(self) -> None:
        # what was derived so far may be narrower now
        self._uint64.clear()
        self._length.clear()
        self._number.clear()

    def _narrow_length(self, value: models.Value, bounds: Bounds) -> None:
        value = self.same_value(value)
        if not isinstance(value, models.Register):
            return
        known = self._given_length.get(value, ANY_LENGTH)
        narrowed = _intersection(known, bounds)
        if narrowed != known:
            self._given_length[value] = narrowed
            self._forget_derived()

    def _narrow_number(self, value: models.Value, bounds: Bounds) -> None:
        value = self.same_value(value)
        if not isinstance(value, models.Register):
            return
        known = self._given_number.get(value, ANY_NUMBER)
        narrowed = _intersection(known, bounds)
        if narrowed == known:
            return
        self._given_number[value] = narrowed
        # a bound on the bytes of a uint64 is a bound on the uint64
        match self._intrinsic(value):
            case models.Intrinsic(op=AVMOp.itob, args=[number]):
                self._narrow(number, (bounds[0], min(bounds[1], MAX_UINT64)))
        self._forget_derived()

    def _narrow_range(self, value: models.Value, bounds: Bounds) -> None:
        if _is_bytes(value):
            self._narrow_number(value, bounds)
        else:
            self._narrow(value, (bounds[0], min(bounds[1], MAX_UINT64)))

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
            case models.Intrinsic(op=comparison, args=[a, b]) if (
                comparison in _NEGATED and _is_bytes(a) and _is_bytes(b)
            ):
                # the same bytes, or not: only the former says something
                if (comparison is AVMOp.eq) == holds:
                    both = _intersection(self.length(a), self.length(b))
                    self._narrow_length(a, both)
                    self._narrow_length(b, both)
                    self._learn_comparison(AVMOp.eq, a, b)
            case models.Intrinsic(op=comparison, args=[a, b]) if (
                comparison in _NUMBER_COMPARISONS and _is_bytes(a) and _is_bytes(b)
            ):
                # the comparison gave a result, whichever it was
                self._learn_completed(source)
                comparison = _NUMBER_COMPARISONS[comparison]
                self._learn_comparison(comparison if holds else _NEGATED[comparison], a, b)

    def _learn_comparison(self, comparison: AVMOp, a: models.Value, b: models.Value) -> None:
        """Narrows two uint64 values, or two byte arrays read as numbers, to what it takes for
        a comparison of them to hold."""
        a_low, a_high = self._range(a)
        b_low, b_high = self._range(b)
        top = ANY_NUMBER[1] if _is_bytes(a) else MAX_UINT64
        match comparison:
            case AVMOp.eq:
                both = max(a_low, b_low), min(a_high, b_high)
                self._narrow_range(a, both)
                self._narrow_range(b, both)
                self._ordered(a, b)
                self._ordered(b, a)
            case AVMOp.lt:
                self._narrow_range(a, (0, b_high - 1))
                self._narrow_range(b, (a_low + 1, top))
                self._ordered(a, b)
            case AVMOp.lte:
                self._narrow_range(a, (0, b_high))
                self._narrow_range(b, (a_low, top))
                self._ordered(a, b)
            case AVMOp.gt:
                self._narrow_range(a, (b_low + 1, top))
                self._narrow_range(b, (0, a_high - 1))
                self._ordered(b, a)
            case AVMOp.gte:
                self._narrow_range(a, (b_low, top))
                self._narrow_range(b, (0, a_high))
                self._ordered(b, a)

    def _narrow(self, value: models.Value, bounds: Bounds) -> None:
        value = self.same_value(value)
        if not isinstance(value, models.Register):
            return
        known = self._given_uint64.get(value, ANY_UINT64)
        narrowed = _intersection(known, bounds)
        if narrowed == known:
            return
        self._given_uint64[value] = narrowed
        match self._intrinsic(value):
            # a bound on the length of a byte array is a bound on the byte array
            case models.Intrinsic(op=AVMOp.len_, args=[array]):
                self._narrow_length(array, (bounds[0], min(bounds[1], MAX_BYTES_LENGTH)))
            # a bound on the uint64 that a byte array converts to is a bound on its number
            case models.Intrinsic(op=AVMOp.btoi, args=[array]):
                self._narrow_number(array, bounds)
            # and so is a bound on the number of bits it takes
            case models.Intrinsic(op=AVMOp.bitlen, args=[operand]):
                low = 1 << (bounds[0] - 1) if 0 < bounds[0] <= _MAX_BITS else 0
                high = (1 << bounds[1]) - 1 if bounds[1] < _MAX_BITS else ANY_NUMBER[1]
                self._narrow_range(operand, (low, high))
        self._forget_derived()

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
        """True if `a` is known to be at most `b`: two uint64 values, or two byte arrays read
        as numbers."""
        a = self.same_value(a)
        b = self.same_value(b)
        return a == b or (a, b) in self._given_ordered or self._range(a)[1] <= self._range(b)[0]

    def within(
        self, array: models.Value, start: models.Value, size: models.Value | int = 0
    ) -> bool:
        """True if the uint64 value `start`, plus `size`, is known to be at most the length of
        a byte array."""
        size = self._plain(size)
        size_high = size if isinstance(size, int) else self.uint64(size)[1]
        if self.uint64(start)[1] + size_high <= self.length(array)[0]:
            return True
        array = self.same_value(array)
        start = self.same_value(start)
        # an operation that has completed took at least as much of the array from that start
        for known_array, known_start, known_size in self._given_within:
            if known_array == array and known_start == start and self._covers(known_size, size):
                return True
        # a register that holds a sum is not above a register that holds the length
        for total, length in self._given_ordered:
            match self._intrinsic(total), self._intrinsic(length):
                case (
                    models.Intrinsic(op=AVMOp.add, args=[x, y]),
                    models.Intrinsic(op=AVMOp.len_, args=[measured]),
                ) if self.same_value(measured) == array:
                    for first, second in ((x, y), (y, x)):
                        if self.same_value(first) == start and self._covers(
                            self._plain(second), size
                        ):
                            return True
        return False

    def outcome(self, op: models.Intrinsic) -> bool | None:
        """What a comparison or a logical op gives, if the bounds of its operands leave only
        one result. This does not say whether the op can fail."""
        match op.op, op.args:
            case comparison, [a, b] if comparison in _NEGATED and _is_uint64(a) and _is_uint64(b):
                return self._compare(comparison, a, b)
            case comparison, [a, b] if (
                comparison in _NUMBER_COMPARISONS and _is_bytes(a) and _is_bytes(b)
            ):
                return self._compare(_NUMBER_COMPARISONS[comparison], a, b)
            case ((AVMOp.eq | AVMOp.neq) as comparison, [a, b]) if _is_bytes(a) and _is_bytes(b):
                # byte arrays of different lengths, or that are different numbers, differ
                if _apart(self.length(a), self.length(b)) or _apart(
                    self.number(a), self.number(b)
                ):
                    return comparison is AVMOp.neq
            case AVMOp.not_, [a]:
                low, high = self.uint64(a)
                if high == 0:
                    return True
                if low > 0:
                    return False
            case AVMOp.and_, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                if a_low > 0 and b_low > 0:
                    return True
                if a_high == 0 or b_high == 0:
                    return False
            case AVMOp.or_, [a, b]:
                a_low, a_high = self.uint64(a)
                b_low, b_high = self.uint64(b)
                if a_low > 0 or b_low > 0:
                    return True
                if a_high == 0 and b_high == 0:
                    return False
        return None

    def _compare(self, comparison: AVMOp, a: models.Value, b: models.Value) -> bool | None:
        a_bounds = self._range(a)
        b_bounds = self._range(b)
        match comparison:
            case AVMOp.lt:
                return _not(self._at_most(b, a, b_bounds, a_bounds))
            case AVMOp.lte:
                return self._at_most(a, b, a_bounds, b_bounds)
            case AVMOp.gt:
                return _not(self._at_most(a, b, a_bounds, b_bounds))
            case AVMOp.gte:
                return self._at_most(b, a, b_bounds, a_bounds)
            case AVMOp.eq | AVMOp.neq:
                if _apart(a_bounds, b_bounds):
                    return comparison is AVMOp.neq
                if a_bounds[0] == a_bounds[1] == b_bounds[0] == b_bounds[1]:
                    return comparison is AVMOp.eq
        return None

    def _at_most(
        self, a: models.Value, b: models.Value, a_bounds: Bounds, b_bounds: Bounds
    ) -> bool | None:
        """Whether `a <= b`, if that is known either way."""
        if a_bounds[1] <= b_bounds[0] or self.not_above(a, b):
            return True
        if a_bounds[0] > b_bounds[1]:
            return False
        return None

    def _range(self, value: models.Value) -> Bounds:
        """The range of a uint64 value, or of the number that a byte array is."""
        if _is_bytes(value):
            return self.number(value)
        return self.uint64(value)

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
        try:
            result = self._uint64[value]
        except KeyError:
            pass
        else:
            return result
        given = self._given_uint64.get(self._same.get(value, value), ANY_UINT64)
        if value not in self._definitions:
            return given
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
        try:
            result = self._length[value]
        except KeyError:
            pass
        else:
            return result
        given = self._given_length.get(self._same.get(value, value), ANY_LENGTH)
        if value not in self._definitions:
            return given
        self._length[value] = given
        result = self._bounded(value, ANY_LENGTH, self._length_of_definition)
        result = _intersection(result, given)
        if self.is_minimal(value):
            # without leading zero bytes, the number says how long the byte array is
            low, high = self.number(value)
            result = _intersection(result, (_num_bytes(low), _num_bytes(high)))
        self._length[value] = result
        return result

    def number(self, value: models.Value) -> Bounds:
        """The range of a byte array value read as an unsigned big-endian number, the way byte
        math reads its operands. Leading zero bytes do not count, so this says nothing about
        the length of the byte array, unless `is_minimal` holds for it."""
        if isinstance(value, models.Constant):
            constant = get_bytes_constant(value)
            if constant is None:
                return ANY_NUMBER
            number = int.from_bytes(constant, "big")
            return number, number
        if not isinstance(value, models.Register) or not _is_bytes(value):
            return ANY_NUMBER
        try:
            result = self._number[value]
        except KeyError:
            pass
        else:
            return result
        same = self._same.get(value, value)
        if self.is_minimal(value):
            # the length of such a value follows from its number, not the other way around
            longest = self._given_length.get(same, ANY_LENGTH)[1]
        else:
            longest = self.length(value)[1]
        given = _intersection(
            self._given_number.get(same, ANY_NUMBER), (0, _largest_number(longest))
        )
        if value not in self._definitions:
            return given
        self._number[value] = given
        result = self._bounded(value, ANY_NUMBER, self._number_of_definition)
        result = self._number[value] = _intersection(result, given)
        return result

    def is_minimal(self, value: models.Value) -> bool:
        """True if a byte array value is known to have no leading zero bytes, which is how the
        results of byte math are encoded. The number 0 is the empty byte array then."""
        if isinstance(value, models.Constant):
            constant = get_bytes_constant(value)
            return constant is not None and not constant.startswith(b"\x00")
        if not isinstance(value, models.Register):
            return False
        try:
            return self._minimal[value]
        except KeyError:
            pass
        if self._depth >= _MAX_DEPTH:
            return False
        # until this definition is resolved, a definition that depends on itself is not
        self._minimal[value] = False
        self._depth += 1
        try:
            match self._definitions.get(value):
                case (models.Value() as source, _):
                    result = self.is_minimal(source)
                case (models.ValueTuple(values=values), index):
                    result = self.is_minimal(values[index])
                case (models.Intrinsic(op=op), _) if op in _BYTE_MATH:
                    result = True
                case (models.Intrinsic(op=AVMOp.select, args=[a, b, _]), _):
                    result = self.is_minimal(a) and self.is_minimal(b)
                case (models.Phi(args=args), _):
                    result = all(self.is_minimal(arg.value) for arg in args)
                case _:
                    result = False
        finally:
            self._depth -= 1
        self._minimal[value] = result
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

    def _number_of_definition(
        self, source: models.ValueProvider | models.Phi, index: int
    ) -> Bounds:
        match source:
            case models.Value():
                return self.number(source)
            case models.ValueTuple(values=values):
                return self.number(values[index])
            case models.Phi(args=args):
                unconditional = self._unconditional or self
                return _union(unconditional.number(arg.value) for arg in args)
            case models.Intrinsic() if len(source.types) == 1:
                return self._number_of_intrinsic(source)
            case _:
                return ANY_NUMBER

    def _uint64_of_intrinsic(self, op: models.Intrinsic) -> Bounds:
        match op.op, op.args:
            case AVMOp.len_, [a]:
                return self.length(a)
            case AVMOp.btoi, [a]:
                # the number of a byte array of at most 8 bytes, which is what there is a
                # result for
                low, high = self.number(a)
                return low, min(high, MAX_UINT64)
            case AVMOp.getbyte, [a, b]:
                return self._uint64_of_part(a, b, 1)
            case AVMOp.extract_uint16, [a, b]:
                return self._uint64_of_part(a, b, 2)
            case AVMOp.extract_uint32, [a, b]:
                return self._uint64_of_part(a, b, 4)
            case AVMOp.extract_uint64, [a, b]:
                return self._uint64_of_part(a, b, 8)
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
                # a byte array is read as a number
                a_low, a_high = self._range(a)
                return a_low.bit_length(), a_high.bit_length()
            case AVMOp.select, [a, b, _] if _is_uint64(a) and _is_uint64(b):
                return _union((self.uint64(a), self.uint64(b)))
            case _:
                (result_type,) = op.op_signature.returns
                if result_type == PrimitiveIRType.bool:
                    match self.outcome(op):
                        case True:
                            return 1, 1
                        case False:
                            return 0, 0
                    return _BOOL
                return ANY_UINT64

    def _uint64_of_part(self, array: models.Value, start: models.Value, size: int) -> Bounds:
        low, high = self._number_of_part(array, self.uint64(start), (size, size))
        return low, min(high, 256**size - 1)

    def _number_of_part(self, array: models.Value, start: Bounds, size: Bounds | None) -> Bounds:
        """The range of the number that some bytes of a byte array are: `size` bytes from
        `start`, or without a size all the bytes from `start`. This is known where the part
        is known and the bytes before it are all zero."""
        length, longest = self.length(array)
        first, last = start
        if length != longest or first != last:
            return ANY_NUMBER
        if size is None:
            count = length - first
        elif size[0] != size[1]:
            return ANY_NUMBER
        else:
            count = size[0]
        after = length - first - count
        # with a part that is not within the byte array there is no result
        if count < 0 or after < 0:
            return ANY_NUMBER
        low, high = self.number(array)
        low >>= 8 * after
        high >>= 8 * after
        if high > _largest_number(count):
            # the bytes before the part may be anything
            return 0, _largest_number(count)
        return low, high

    def _math_number(self, value: models.Value) -> Bounds:
        # the operands of byte math are at most MAX_BIGUINT_BYTES long whenever there is a result
        low, high = self.number(value)
        return low, min(high, _MAX_MATH_NUMBER)

    def _number_of_intrinsic(self, op: models.Intrinsic) -> Bounds:
        match op.op, op.args:
            case AVMOp.itob, [a]:
                return self.uint64(a)
            case AVMOp.bzero, _:
                return 0, 0
            case AVMOp.add_bytes, [a, b]:
                a_low, a_high = self._math_number(a)
                b_low, b_high = self._math_number(b)
                return a_low + b_low, a_high + b_high
            case AVMOp.sub_bytes, [a, b]:
                a_low, a_high = self._math_number(a)
                b_low, b_high = self._math_number(b)
                return max(a_low - b_high, 0), a_high - b_low
            case AVMOp.mul_bytes, [a, b]:
                a_low, a_high = self._math_number(a)
                b_low, b_high = self._math_number(b)
                return a_low * b_low, a_high * b_high
            case AVMOp.div_floor_bytes, [a, b]:
                a_low, a_high = self._math_number(a)
                b_low, b_high = self._math_number(b)
                if b_high == 0:
                    return ANY_NUMBER
                return a_low // b_high, a_high // max(b_low, 1)
            case AVMOp.mod_bytes, [a, b]:
                a_low, a_high = self._math_number(a)
                b_low, b_high = self._math_number(b)
                if b_high == 0:
                    return ANY_NUMBER
                if a_high < b_low:
                    # what is below the modulus stays as it is
                    return a_low, a_high
                return 0, min(a_high, b_high - 1)
            case AVMOp.bsqrt, [a]:
                a_low, a_high = self._math_number(a)
                return math.isqrt(a_low), math.isqrt(a_high)
            case AVMOp.bitwise_and_bytes, [a, b]:
                return 0, min(self.number(a)[1], self.number(b)[1])
            case AVMOp.bitwise_or_bytes, [a, b]:
                a_low, a_high = self.number(a)
                b_low, b_high = self.number(b)
                return max(a_low, b_low), min(_all_bits(max(a_high, b_high)), a_high + b_high)
            case AVMOp.bitwise_xor_bytes, [a, b]:
                a_high = self.number(a)[1]
                b_high = self.number(b)[1]
                return 0, min(_all_bits(max(a_high, b_high)), a_high + b_high)
            case AVMOp.bitwise_not_bytes, [a]:
                length, longest = self.length(a)
                if length != longest:
                    return ANY_NUMBER
                a_low, a_high = self.number(a)
                return _largest_number(length) - a_high, _largest_number(length) - a_low
            case AVMOp.concat, [a, b]:
                # the first operand moves up by the length of the second
                a_low, a_high = self.number(a)
                b_low, b_high = self.number(b)
                shortest, longest = self.length(b)
                return (a_low << (8 * shortest)) + b_low, (a_high << (8 * longest)) + b_high
            case AVMOp.extract, [a]:
                start, size = op.immediates
                assert isinstance(start, int)
                assert isinstance(size, int)
                # a size of 0 takes the rest
                return self._number_of_part(a, (start, start), (size, size) if size else None)
            case AVMOp.extract3, [a, b, c]:
                return self._number_of_part(a, self.uint64(b), self.uint64(c))
            case AVMOp.substring, [a]:
                start, end = op.immediates
                assert isinstance(start, int)
                assert isinstance(end, int)
                return self._number_of_part(a, (start, start), (end - start, end - start))
            case AVMOp.substring3, [a, b, c]:
                b_low, b_high = self.uint64(b)
                c_low, c_high = self.uint64(c)
                return self._number_of_part(a, (b_low, b_high), (c_low - b_high, c_high - b_low))
            case AVMOp.select, [a, b, _] if _is_bytes(a) and _is_bytes(b):
                return _union((self.number(a), self.number(b)))
            case _:
                return ANY_NUMBER

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
        return 0, ANY_NUMBER[1]
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


def _apart(a: Bounds, b: Bounds) -> bool:
    """True if no value is within both."""
    return a[1] < b[0] or b[1] < a[0]


def _not(value: bool | None) -> bool | None:
    return None if value is None else not value


def _is_uint64(value: models.Value) -> bool:
    return value.ir_type.maybe_avm_type is AVMType.uint64


def _is_bytes(value: models.Value) -> bool:
    return value.ir_type.maybe_avm_type is AVMType.bytes


def _all_bits(value: int) -> int:
    return (1 << value.bit_length()) - 1


def _num_bytes(number: int) -> int:
    """How many bytes a number takes without leading zero bytes."""
    return (number.bit_length() + 7) // 8


def _largest_number(num_bytes: int) -> int:
    return (1 << (8 * num_bytes)) - 1
