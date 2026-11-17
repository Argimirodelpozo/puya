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
around it to complete. `BoundsAtPoints` does that for every point of a subroutine.

Which run of a definition a fact is about
-----------------------------------------
A register is defined once in the text of a subroutine, and each time control passes its
definition while the subroutine runs. A fact that is learned at a point A, such as the
condition of an assert, is about the values that the registers hold when A runs: for each
register the value of the last run of its definition. Such a fact is used at a point P only if
A dominates P, and it mentions only registers whose definitions dominate A. Between a run of A
and the arrival at P none of these definitions can run again: a way from the definition to P
that does not pass A again would, after a way from the start of the subroutine to the
definition, be a way to P around A. So at P the registers still hold the values that the fact
is about, and so does every register that is computed from them and can be used at P.

A phi is the exception. It takes the value that its argument holds when control passes from a
predecessor to the block of the phi, and the argument may be defined again before the phi is
read. So what is known at a point about the argument of a phi says nothing about the phi.

Values defined around a loop
----------------------------
A phi with an argument that is computed from the phi itself gets its bounds by induction over
the runs of the phi. Suppose that a range R holds every value the phi had so far. When control
passes from a predecessor to the phi, the argument for that predecessor is within the bounds
that follow from R and from facts that hold at the end of the predecessor: the condition of
the branch that leads from it to the phi, and the asserts, branches and completed operations
of the blocks of the loop that dominate it, from the block of the phi on. If for every
predecessor those bounds are within R, then R holds the next value of the phi as well, and by
induction all of them.

R starts as the bounds of the arguments that do not depend on the phi. It is extended by what
passes around the loop a fixed number of times; if it has not settled by then, each side that
moved is widened to the limit of its type. Whatever the candidate, it is only used if the
check above holds for it, and is otherwise given up, so that it does not matter for the
result being right how the candidate was found. After widening, the range that the check
itself arrives at is tried as a narrower candidate.

While a candidate is being checked, everything that is derived from it is kept apart from what
is known for good and discarded afterwards. This is refused, in favour of the bounds that hold
whatever came around the loop:
- a phi whose arguments all depend on it
- a phi that is reached while the candidate of another phi is being checked, for that check
- more than a fixed number of checks for one subroutine, so that the time this takes is
  bounded however many loops there are
And these facts are not used for a phi, though they hold on the way into it:
- what was established before the loop. At each point of the loop it is known all the same,
  see `BoundsAtPoints`
- the facts of more than a fixed number of blocks of the loop, of which the block of the phi
  and the blocks closest to the predecessor are the ones that are used
- the condition of a branch into a block that can be entered another way as well, other
  than from a block that it dominates itself
"""

import math
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import attrs
import networkx as nx  # type: ignore[import-untyped]

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
# how many blocks of a loop the facts that hold on the way into a phi are taken from
_MAX_EDGE_BLOCKS: typing.Final = 8
# how many times candidate bounds for the phis of one subroutine are checked
_MAX_LOOP_CHECKS: typing.Final = 256
# how many times the bounds of a value defined around a loop are extended before widening
_LOOP_ROUNDS: typing.Final = 2
# how many candidates for the bounds of such values are checked inside each other
_MAX_NESTED_LOOPS: typing.Final = 1
# how many ordered pairs, and how many parts taken of byte arrays, are kept for one point
_MAX_RELATIONS: typing.Final = 64
# how many derived bounds are discarded altogether when one of them may have changed, rather
# than only the ones that were derived from it
_FEW_DERIVED: typing.Final = 24

_Fact = typing.Literal["uint64", "length", "number"]
_UNKNOWN: typing.Final[Mapping[_Fact, Bounds]] = {
    "uint64": ANY_UINT64,
    "length": ANY_LENGTH,
    "number": ANY_NUMBER,
}

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
# the ops that `ValueBounds.given` learns from when they have completed
_TEACHES_WHEN_COMPLETED: typing.Final = frozenset(
    (
        AVMOp.extract,
        AVMOp.extract3,
        AVMOp.substring,
        AVMOp.substring3,
        AVMOp.extract_uint16,
        AVMOp.extract_uint32,
        AVMOp.extract_uint64,
        AVMOp.getbyte,
        AVMOp.setbyte,
        AVMOp.btoi,
        AVMOp.concat,
        *_BYTE_MATH,
        *_NUMBER_COMPARISONS,
        AVMOp.bzero,
        AVMOp.shl,
        AVMOp.shr,
        AVMOp.div_floor,
        AVMOp.mod,
        AVMOp.add,
        AVMOp.mul,
        AVMOp.sub,
        AVMOp.loads,
        AVMOp.stores,
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
    # the blocks of the subroutine, and for each phi the block it is in
    _blocks: Sequence[models.BasicBlock] = attrs.field(default=(), init=False)
    _phi_blocks: Mapping[models.Register, models.BasicBlock] = attrs.field(
        factory=dict, init=False
    )
    _dominators: Mapping[models.BasicBlock, models.BasicBlock] | None = attrs.field(
        default=None, init=False
    )
    _edge_facts: dict[
        tuple[models.BasicBlock, models.BasicBlock],
        tuple[list[models.Value], list[models.Value], list[models.Intrinsic]],
    ] = attrs.field(factory=dict, init=False)
    # While the bounds of a phi are being derived (see the module docstring):
    # the phis whose arguments are being looked at, with whether the phi was read meanwhile
    _watching: dict[tuple[_Fact, models.Register], bool] = attrs.field(factory=dict, init=False)
    # what was derived meanwhile and has to go when the candidate bounds of the phi do
    _provisional: list[tuple[dict[models.Register, Bounds], models.Register]] = attrs.field(
        factory=list, init=False
    )
    _recording: int = attrs.field(default=0, init=False)
    # how many candidates for phis are being checked inside each other
    _nested_loops: int = attrs.field(default=0, init=False)
    # how many more candidates may be checked
    _loop_checks: int = attrs.field(default=0, init=False)
    # the registers whose bounds are being derived, innermost last, with where the result goes
    _deriving: list[tuple[dict[models.Register, Bounds], models.Register]] = attrs.field(
        factory=list, init=False
    )
    # how many comparisons the conditions that are being learned have come down to
    _comparisons_learned: int = attrs.field(default=0, init=False)
    # for each register the registers that are defined in terms of it
    _users: Mapping[models.Register, Sequence[models.Register]] | None = attrs.field(
        default=None, init=False
    )

    @classmethod
    def of(cls, subroutine: models.Subroutine, *, loops: bool = True) -> "ValueBounds":
        """The bounds of the values of a subroutine. Without `loops`, a value that is defined
        around a loop only gets the bounds that hold whatever came around the loop, which
        takes less time to work out."""
        definitions = dict[models.Register, _Definition]()
        phi_blocks = dict[models.Register, models.BasicBlock]()
        for block in subroutine.body:
            for phi in block.phis:
                definitions[phi.register] = (phi, 0)
                phi_blocks[phi.register] = block
            for op in block.ops:
                if isinstance(op, models.Assignment):
                    for index, target in enumerate(op.targets):
                        definitions[target] = (op.source, index)
        result = cls(definitions)
        result._same = _same_values(definitions)  # noqa: SLF001
        result._blocks = subroutine.body  # noqa: SLF001
        result._phi_blocks = phi_blocks  # noqa: SLF001
        if loops:
            result._loop_checks = _MAX_LOOP_CHECKS  # noqa: SLF001
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
        result = self._extended()
        result._learn_all(non_zero, zero, completed)  # noqa: SLF001
        return result

    def _extended(self) -> "ValueBounds":
        """Bounds that are given what these are, to be given more."""
        result = ValueBounds(self._definitions)
        result._unconditional = self._unconditional or self  # noqa: SLF001
        result._same = self._same  # noqa: SLF001
        result._minimal = self._minimal  # noqa: SLF001
        result._given_uint64 = dict(self._given_uint64)  # noqa: SLF001
        result._given_length = dict(self._given_length)  # noqa: SLF001
        result._given_number = dict(self._given_number)  # noqa: SLF001
        result._given_ordered = set(self._given_ordered)  # noqa: SLF001
        result._given_within = set(self._given_within)  # noqa: SLF001
        return result

    def _learn_all(
        self,
        non_zero: Iterable[models.Value],
        zero: Iterable[models.Value],
        completed: Iterable[models.Intrinsic],
    ) -> None:
        conditions = [(c, True) for c in non_zero] + [(c, False) for c in zero]
        completed = list(completed)
        self._comparisons_learned = 0
        for condition, holds in conditions:
            self._learn(condition, holds=holds, depth=0)
        for op in completed:
            self._learn_completed(op)
        if self._comparisons_learned + len(completed) > 1:
            # once more, so that each can build on what a later one says
            for condition, holds in conditions:
                self._learn(condition, holds=holds, depth=0)
            for op in completed:
                self._learn_completed(op)

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
            for value in pair:
                if isinstance(value, models.Register):
                    self._forget_derived(value)

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

    def _forget_derived(self, register: models.Register) -> None:
        """Discards the bounds that were derived from what was known about a register, which
        may be narrower now."""
        memos = (self._uint64, self._length, self._number)
        if len(self._uint64) + len(self._length) + len(self._number) <= _FEW_DERIVED:
            for memo in memos:
                memo.clear()
            return
        unconditional = self._unconditional or self
        users = unconditional._users  # noqa: SLF001
        if users is None:
            users = unconditional._users = _users_of_registers(  # noqa: SLF001
                self._definitions, self._same
            )
        # Whatever is derived from a register is derived after the register itself, and kept
        # with it. So a register that nothing is kept for has nothing kept that depends on it,
        # unless it is the register that was narrowed, which may have no definition
        pending = [register]
        seen = {register}
        while pending:
            current = pending.pop()
            for memo in memos:
                memo.pop(current, None)
            for user in users.get(current, ()):
                if user not in seen and any(user in memo for memo in memos):
                    seen.add(user)
                    pending.append(user)

    def _narrow_length(self, value: models.Value, bounds: Bounds) -> None:
        value = self.same_value(value)
        if not isinstance(value, models.Register):
            return
        known = self._given_length.get(value, ANY_LENGTH)
        narrowed = _intersection(known, bounds)
        if narrowed != known:
            self._given_length[value] = narrowed
            self._forget_derived(value)

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
        self._forget_derived(value)

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
        self._comparisons_learned += 1
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
        self._forget_derived(value)

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
            if self._watching:
                self._read("uint64", value)
            return result
        given = self._given_uint64.get(self._same.get(value, value), ANY_UINT64)
        if value not in self._definitions:
            return given
        # until this definition is resolved, a definition that depends on itself sees no bound
        self._derived(self._uint64, value, given)
        result = self._bounded(value, ANY_UINT64, self._uint64_of_definition)
        result = self._uint64[value] = _intersection(result, given)
        self._deriving.pop()
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
            if self._watching:
                self._read("length", value)
            return result
        given = self._given_length.get(self._same.get(value, value), ANY_LENGTH)
        if value not in self._definitions:
            return given
        self._derived(self._length, value, given)
        result = self._bounded(value, ANY_LENGTH, self._length_of_definition)
        result = _intersection(result, given)
        if self.is_minimal(value):
            # without leading zero bytes, the number says how long the byte array is
            low, high = self.number(value)
            result = _intersection(result, (_num_bytes(low), _num_bytes(high)))
        self._length[value] = result
        self._deriving.pop()
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
            if self._watching:
                self._read("number", value)
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
        self._derived(self._number, value, given)
        result = self._bounded(value, ANY_NUMBER, self._number_of_definition)
        result = self._number[value] = _intersection(result, given)
        self._deriving.pop()
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

    def _fact(self, fact: _Fact, value: models.Value) -> Bounds:
        match fact:
            case "uint64":
                return self.uint64(value)
            case "length":
                return self.length(value)
            case "number":
                return self.number(value)

    def _memo(self, fact: _Fact) -> dict[models.Register, Bounds]:
        match fact:
            case "uint64":
                return self._uint64
            case "length":
                return self._length
            case "number":
                return self._number

    def _read(self, fact: _Fact, register: models.Register) -> None:
        key = (fact, register)
        if key in self._watching:
            self._watching[key] = True

    def _derived(
        self, memo: dict[models.Register, Bounds], register: models.Register, bounds: Bounds
    ) -> None:
        memo[register] = bounds
        self._deriving.append((memo, register))
        if self._recording:
            self._provisional.append((memo, register))

    def _discard_provisional(self, since: int) -> None:
        while len(self._provisional) > since:
            memo, register = self._provisional.pop()
            memo.pop(register, None)

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

    def _of_phi(self, phi: models.Phi, fact: _Fact) -> Bounds:
        unconditional = self._unconditional
        if unconditional is None:
            return self._around_loop(phi, fact)
        # what holds for a phi whenever it is defined does not depend on conditions. The
        # definitions that were followed to get here count towards the limit there as well
        depth = unconditional._depth  # noqa: SLF001
        unconditional._depth = max(depth, self._depth)  # noqa: SLF001
        try:
            return unconditional._fact(fact, phi.register)  # noqa: SLF001
        finally:
            unconditional._depth = depth  # noqa: SLF001

    def _around_loop(self, phi: models.Phi, fact: _Fact) -> Bounds:
        """The bounds of a phi. If some of its arguments are computed from the phi, they are
        found by induction as the module docstring describes."""
        if not self._loop_checks:
            # the phi itself has no bound at this point
            return _union(self._fact(fact, arg.value) for arg in phi.args)
        register = phi.register
        if fact == "length":
            # the length of a byte array without leading zero bytes follows from its number,
            # so that comes first
            self.number(register)
        key = (fact, register)
        # Which arguments are computed from the phi: the ones that read it when definitions
        # are followed and nothing else is, as without the bounds around loops
        dependent = list[models.PhiArgument]()
        independent = list[models.PhiArgument]()
        loose = list[Bounds]()
        checks = self._loop_checks
        since = len(self._provisional)
        self._loop_checks = 0
        self._recording += 1
        try:
            for arg in phi.args:
                self._watching[key] = False
                loose.append(self._fact(fact, arg.value))
                (dependent if self._watching[key] else independent).append(arg)
        finally:
            # not there for a phi without arguments
            self._watching.pop(key, None)
            self._recording -= 1
            self._loop_checks = checks
            # derived without the bounds around loops, which are what is wanted
            self._discard_provisional(since)
        block = self._phi_blocks.get(register)
        if (
            not dependent
            or not independent
            or block is None
            or self._nested_loops >= _MAX_NESTED_LOOPS
        ):
            # the phi itself has no bound at this point
            return _union(self._fact(fact, arg.value) for arg in phi.args)
        # what holds without knowing anything about the values that came around the loop
        result = _union(loose)
        unknown = _UNKNOWN[fact]
        start = _union(self._fact(fact, arg.value) for arg in independent)
        candidate = start
        for _ in range(_LOOP_ROUNDS):
            around = self._through_loop(phi, dependent, block, fact, candidate)
            if _contains(candidate, around):
                return _intersection(result, candidate)
            candidate = _union((candidate, around))
        # not settled: each side that moved goes as far as it can
        candidate = (
            unknown[0] if candidate[0] < start[0] else start[0],
            unknown[1] if candidate[1] > start[1] else start[1],
        )
        around = self._through_loop(phi, dependent, block, fact, candidate)
        if not _contains(candidate, around):
            return result
        # what came around is a narrower candidate, to be confirmed like any other
        narrower = _union((start, around))
        if narrower != candidate and _contains(
            narrower, self._through_loop(phi, dependent, block, fact, narrower)
        ):
            candidate = narrower
        return _intersection(result, candidate)

    def _through_loop(
        self,
        phi: models.Phi,
        arguments: Sequence[models.PhiArgument],
        block: models.BasicBlock,
        fact: _Fact,
        candidate: Bounds,
    ) -> Bounds:
        """The bounds of the values that reach a phi through the given arguments, if the
        candidate bounds hold for every value the phi had before."""
        if not self._loop_checks:
            # as much as any value: the candidate is not confirmed
            return 0, ANY_NUMBER[1]
        self._loop_checks -= 1
        memo = self._memo(fact)
        memo[phi.register] = candidate
        since = len(self._provisional)
        self._recording += 1
        self._nested_loops += 1
        try:
            return _union(
                self._on_edge(arg.through, block)._fact(fact, arg.value)  # noqa: SLF001
                for arg in arguments
            )
        finally:
            self._nested_loops -= 1
            self._recording -= 1
            # all of it was derived from the candidate
            self._discard_provisional(since)

    def _on_edge(self, predecessor: models.BasicBlock, block: models.BasicBlock) -> "ValueBounds":
        """The bounds that hold when control passes from a block of a loop to the block that
        the loop starts with, given what the blocks of the loop establish."""
        try:
            non_zero, zero, completed = self._edge_facts[predecessor, block]
        except KeyError:
            non_zero, zero, completed = [], [], []
            # the blocks of the loop that dominate the predecessor, from the last one up
            dominating = [predecessor]
            while dominating[-1] is not block:
                dominator = self.immediate_dominator(dominating[-1])
                if dominator is None:
                    # the block of the phi does not dominate the predecessor: no facts
                    dominating.clear()
                    break
                dominating.append(dominator)
            # the first block of the loop, and the ones closest to the predecessor
            kept = set(dominating[: _MAX_EDGE_BLOCKS - 1])
            kept.add(block)
            # in the order of the program
            for index in reversed(range(len(dominating))):
                current = dominating[index]
                if current not in kept:
                    continue
                asserted, ran = facts_of_ops(current.ops)
                non_zero.extend(asserted)
                completed.extend(ran)
                # the branch that leads on: to the next block if that block is only entered
                # from this one, and from the predecessor to the block of the phi
                if index:
                    successor = dominating[index - 1]
                    if not self.only_entered_from_dominator(successor):
                        continue
                else:
                    successor = block
                match branch_condition(current, successor):
                    case condition, True:
                        non_zero.append(condition)
                    case condition, False:
                        zero.append(condition)
            self._edge_facts[predecessor, block] = non_zero, zero, completed
        if not (non_zero or zero or completed):
            return self
        result = self._extended()
        # What has been derived holds on the way into the phi as well, so it is taken along,
        # with the candidates for phis. Not what is still being derived, which depends on
        # the phi and is to be derived from the candidate
        result._uint64 = dict(self._uint64)  # noqa: SLF001
        result._length = dict(self._length)  # noqa: SLF001
        result._number = dict(self._number)  # noqa: SLF001
        for memo, register in self._deriving:
            if register not in self._phi_blocks:
                if memo is self._uint64:
                    result._uint64.pop(register, None)  # noqa: SLF001
                elif memo is self._length:
                    result._length.pop(register, None)  # noqa: SLF001
                else:
                    result._number.pop(register, None)  # noqa: SLF001
        result._depth = self._depth  # noqa: SLF001
        result._learn_all(non_zero, zero, completed)  # noqa: SLF001
        return result

    def immediate_dominator(self, block: models.BasicBlock) -> models.BasicBlock | None:
        """The closest block that every way from the start of the subroutine to the given
        block leads through. None for the first block, for a block that cannot be reached, and
        for bounds that were built without a subroutine."""
        unconditional = self._unconditional or self
        dominators = unconditional._dominators  # noqa: SLF001
        if dominators is None:
            dominators = unconditional._dominators = _immediate_dominators(  # noqa: SLF001
                unconditional._blocks  # noqa: SLF001
            )
        return dominators.get(block)

    def only_entered_from_dominator(self, block: models.BasicBlock) -> bool:
        """True if control only gets into a block from its immediate dominator: any other
        predecessor is a block that it dominates, which is reached through the block itself.
        What holds when control passes from the dominator to the block then holds in the
        block, whichever way it was reached."""
        dominator = self.immediate_dominator(block)
        if dominator is None:
            return False
        for predecessor in block.predecessors:
            if predecessor is dominator:
                continue
            current: models.BasicBlock | None = predecessor
            while current is not None and current is not block:
                current = self.immediate_dominator(current)
            if current is None:
                return False
        return True

    def _uint64_of_definition(
        self, source: models.ValueProvider | models.Phi, index: int
    ) -> Bounds:
        match source:
            case models.Value():
                return self.uint64(source)
            case models.ValueTuple(values=values):
                return self.uint64(values[index])
            case models.Phi():
                return self._of_phi(source, "uint64")
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
            case models.Phi():
                return self._of_phi(source, "length")
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
            case models.Phi():
                return self._of_phi(source, "number")
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


@attrs.define
class BoundsAtPoints:
    """The bounds that hold at each point of a subroutine: at the start of a block what the
    blocks that dominate it establish, with the condition of the branch into it if it is only
    entered that way, and before each op of a block also what the ops before it establish.

    The bounds are derived from the subroutine as it was when this was built. An op that is
    replaced by another that gives the same values, or removed because it cannot fail, leaves
    them valid; `ValueBounds.is_minimal` and the lengths of a register whose bytes change do
    not stay valid.
    """

    _bounds: ValueBounds
    _at_end: dict[models.BasicBlock, ValueBounds] = attrs.field(factory=dict, init=False)

    @classmethod
    def of(cls, subroutine: models.Subroutine) -> "BoundsAtPoints":
        return cls(ValueBounds.of(subroutine))

    def at_start(self, block: models.BasicBlock) -> ValueBounds:
        """The bounds that hold before the first op of a block."""
        dominator = self._bounds.immediate_dominator(block)
        if dominator is None:
            return self._bounds
        result = self.at_end(dominator)
        if self._bounds.only_entered_from_dominator(block):
            match branch_condition(dominator, block):
                case condition, True:
                    result = result.given(non_zero=[condition])
                case condition, False:
                    result = result.given(zero=[condition])
        return result

    def at_end(self, block: models.BasicBlock) -> ValueBounds:
        """The bounds that hold after the last op of a block."""
        # dominators first, without recursion: one block may be dominated by a long chain
        missing = list[models.BasicBlock]()
        current: models.BasicBlock | None = block
        while current is not None and current not in self._at_end:
            missing.append(current)
            current = self._bounds.immediate_dominator(current)
        for current in reversed(missing):
            result = self.at_start(current)
            asserted, ran = facts_of_ops(current.ops)
            if asserted or ran:
                result = result.given(non_zero=asserted, completed=ran)
            self._at_end[current] = result
        return self._at_end[block]

    def before_ops(
        self, block: models.BasicBlock
    ) -> Iterator[tuple[models.Op, Callable[[], ValueBounds]]]:
        """Each op of a block, with the bounds that hold before it. The bounds are worked out
        when asked for. They are one object for all the ops of the block, which is given more
        as the ops are passed: what it says holds for an op until the next op is asked for,
        and `given()` of it is a copy that stays as it is. An op may be replaced or removed
        before the next one is asked for."""
        current: ValueBounds | None = None
        asserted = list[models.Value]()
        ran = list[models.Intrinsic]()

        def bounds() -> ValueBounds:
            nonlocal current
            if current is None:
                # what holds at the start of the block may be shared with other blocks
                current = self.at_start(block)._extended()  # noqa: SLF001
            if asserted or ran:
                current._learn_all(asserted, (), ran)  # noqa: SLF001
                asserted.clear()
                ran.clear()
            return current

        for op in block.ops.copy():
            yield op, bounds
            # as the op is now
            new_asserted, new_ran = facts_of_ops([op])
            asserted.extend(new_asserted)
            ran.extend(new_ran)
        if current is not None and block not in self._at_end:
            self._at_end[block] = bounds()


def facts_of_ops(ops: Iterable[models.Op]) -> tuple[list[models.Value], list[models.Intrinsic]]:
    """What a sequence of ops establishes for what follows it: the conditions of its asserts,
    which are not 0, and its AVM ops that say something about their operands by having
    completed."""
    asserted = list[models.Value]()
    ran = list[models.Intrinsic]()
    for op in ops:
        if isinstance(op, models.Assert):
            asserted.append(op.condition)
            continue
        source = op.source if isinstance(op, models.Assignment) else op
        if isinstance(source, models.Intrinsic) and source.op in _TEACHES_WHEN_COMPLETED:
            ran.append(source)
    return asserted, ran


def branch_condition(
    block: models.BasicBlock, successor: models.BasicBlock
) -> tuple[models.Value, bool] | None:
    """The condition that a block branches on, and whether it is not 0 when the branch leads
    to the given successor. None if the block does not end in such a branch, or if it leads
    to the successor either way."""
    match block.terminator:
        case models.ConditionalBranch(condition=condition, non_zero=taken, zero=other) if (
            taken is successor and other is not successor
        ):
            return condition, True
        case models.ConditionalBranch(condition=condition, non_zero=other, zero=taken) if (
            taken is successor and other is not successor
        ):
            return condition, False
    return None


def _immediate_dominators(
    blocks: Sequence[models.BasicBlock],
) -> Mapping[models.BasicBlock, models.BasicBlock]:
    if not blocks:
        return {}
    graph = nx.DiGraph()
    for block in blocks:
        graph.add_node(block)
        for successor in block.successors:
            graph.add_edge(block, successor)
    entry = blocks[0]
    dominators: dict[models.BasicBlock, models.BasicBlock] = nx.immediate_dominators(graph, entry)
    # the first block has no dominator, however it is reported
    dominators.pop(entry, None)
    return dominators


def _users_of_registers(
    definitions: Mapping[models.Register, _Definition],
    same: Mapping[models.Register, models.Register],
) -> Mapping[models.Register, Sequence[models.Register]]:
    """For each register, the registers whose bounds can depend on its bounds: the ones that
    are defined in terms of it, and the ones that hold the same value."""
    users = dict[models.Register, list[models.Register]]()
    for register, (source, _) in definitions.items():
        match source:
            case models.Register():
                operands: Iterable[models.Value] = (source,)
            case models.ValueTuple(values=operands) | models.Intrinsic(args=operands):
                pass
            case models.Phi(args=args):
                operands = [arg.value for arg in args]
            case _:
                operands = ()
        for operand in operands:
            if isinstance(operand, models.Register):
                users.setdefault(operand, []).append(register)
    for register, first in same.items():
        if register != first:
            users.setdefault(first, []).append(register)
    return users


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


def _contains(outer: Bounds, inner: Bounds) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


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
