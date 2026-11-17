"""Rewrites that the ranges of the values of a subroutine allow.

The ranges are those of `puya.ir.optimize.value_bounds`, for the point of each operation: what
the definitions say, the conditions of the asserts and branches that dominate the point, and
what it took for the operations before it to complete. `range_rewrites` is the optimization,
which goes through a subroutine once. It has two groups of rewrites, each of which can be
switched off by its name like an optimization:

`range_constant_outcomes`
    A comparison whose result the ranges fix becomes that result, an assert whose condition
    they show to hold is removed, and a branch whose condition they decide gets a constant
    condition. The proof is the range, so what established the range stays where it is: the
    assert or branch whose condition says so, or the operation that fails otherwise.

`range_operand_results`
    An operation that gives back one of its operands is replaced by that operand: `%` and `b%`
    of a value below the modulus, `&` and `b&` with a mask that covers the value, `b|` and
    `b^` with zeros, and taking all of a byte array.

Two things are required of every rewrite, besides the result being the same value:

- No failure is lost and none is added. An operation is only replaced if the ranges show that
  it cannot fail, or if what replaces it fails for exactly the same operands.
- A byte array stays the same bytes, and not only the same number. Byte math reads its
  operands as numbers and gives results without leading zero bytes, so `b% x m` is `x` for a
  small `x` only if `x` has no leading zero bytes either. Where that is not known, the
  operation is still replaced if everything that uses its result reads a number: byte math,
  its comparisons and `bitlen`. What is known about the bytes of such a result no longer
  holds, so nothing else is decided by it for the rest of the pass.
"""

import enum
import typing
from collections.abc import Container, Mapping

import attrs

from puya import log
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize._utils import SSAReadTracker
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import Effect, effects_of
from puya.ir.optimize.value_bounds import MAX_BIGUINT_BYTES, BoundsAtPoints, ValueBounds

logger = log.get_logger(__name__)


class _Group(enum.Enum):
    constant_outcomes = "range_constant_outcomes"
    operand_results = "range_operand_results"


def range_rewrites(context: IROptimizationContext, subroutine: models.Subroutine) -> bool:
    """Replaces what the ranges of the values of a subroutine show to have a simpler form."""
    switches = context.options.optimizations_override
    groups = {group for group in _Group if switches.get(group.value, True)}
    return bool(groups) and _Rewriter(subroutine, groups).rewrite()


def range_constant_outcomes(
    _context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    """Replaces comparisons, asserts and branch conditions that the ranges of their operands
    decide."""
    return _Rewriter(subroutine, {_Group.constant_outcomes}).rewrite()


def range_operand_results(_context: IROptimizationContext, subroutine: models.Subroutine) -> bool:
    """Replaces operations that the ranges of their operands show to give back an operand."""
    return _Rewriter(subroutine, {_Group.operand_results}).rewrite()


_NUMBER_COMPARISONS: typing.Final[Mapping[AVMOp, AVMOp]] = {
    AVMOp.eq_bytes: AVMOp.eq,
    AVMOp.neq_bytes: AVMOp.neq,
    AVMOp.lt_bytes: AVMOp.lt,
    AVMOp.lte_bytes: AVMOp.lte,
    AVMOp.gt_bytes: AVMOp.gt,
    AVMOp.gte_bytes: AVMOp.gte,
}
_DECIDABLE: typing.Final = frozenset(
    (
        AVMOp.eq,
        AVMOp.neq,
        AVMOp.lt,
        AVMOp.lte,
        AVMOp.gt,
        AVMOp.gte,
        *_NUMBER_COMPARISONS,
        AVMOp.not_,
        AVMOp.and_,
        AVMOp.or_,
    )
)
# ops that read a byte array as a number: its leading zero bytes make no difference to the
# result, and neither does its length as long as byte math takes it
_READS_NUMBER: typing.Final = frozenset(
    (
        AVMOp.add_bytes,
        AVMOp.sub_bytes,
        AVMOp.mul_bytes,
        AVMOp.div_floor_bytes,
        AVMOp.mod_bytes,
        AVMOp.bsqrt,
        *_NUMBER_COMPARISONS,
        AVMOp.bitlen,
    )
)

_Replacement = tuple[str, models.ValueProvider]


@attrs.define
class _Rewriter:
    subroutine: models.Subroutine
    groups: Container[_Group]
    modified: int = attrs.field(default=0, init=False)
    _points: BoundsAtPoints = attrs.field(init=False)
    _reads: SSAReadTracker | None = attrs.field(default=None, init=False)
    # registers that are another byte array than when the bounds were derived
    _other_bytes: set[models.Value] = attrs.field(factory=set, init=False)

    def rewrite(self) -> bool:
        self._points = BoundsAtPoints.of(self.subroutine)
        outcomes = _Group.constant_outcomes in self.groups
        for block in self.subroutine.body:
            for op, bounds in self._points.before_ops(block):
                match op:
                    case models.Assert(condition=models.Register() as condition) if outcomes:
                        if bounds().uint64(condition)[0] > 0:
                            logger.debug(f"range rewrite (assert): removing {op}")
                            block.ops.remove(op)
                            self.modified += 1
                    case models.Assignment(targets=[_], source=models.Intrinsic() as source):
                        if not self._other_bytes or self._other_bytes.isdisjoint(source.args):
                            self._intrinsic(op, source, bounds)
            if outcomes:
                self._branch(block)
        return self.modified > 0

    def _branch(self, block: models.BasicBlock) -> None:
        match block.terminator:
            case models.ConditionalBranch(condition=models.Register() as condition) as branch:
                low, high = self._points.at_end(block).uint64(condition)
                if low > 0 or high == 0:
                    decided = models.UInt64Constant(value=int(low > 0), source_location=None)
                    logger.debug(f"range rewrite (branch): {condition} is {decided} in {branch}")
                    block.terminator = attrs.evolve(branch, condition=decided)
                    self.modified += 1

    def _intrinsic(
        self,
        assignment: models.Assignment,
        source: models.Intrinsic,
        bounds: typing.Callable[[], ValueBounds],
    ) -> None:
        replacement: _Replacement | None = None
        if _Group.constant_outcomes in self.groups and source.op in _DECIDABLE:
            replacement = _outcome(source, bounds())
        if replacement is None and _Group.operand_results in self.groups:
            replacement = self._operand_result(assignment, source, bounds)
        if replacement is not None:
            rule, new_source = replacement
            logger.debug(f"range rewrite ({rule}): {source} -> {new_source}")
            self._replace(assignment, new_source)

    def _replace(self, assignment: models.Assignment, source: models.ValueProvider) -> None:
        if self._reads is None:
            assignment.source = source
        else:
            with self._reads.update(assignment):
                assignment.source = source
        self.modified += 1

    def _tracked_reads(self) -> SSAReadTracker:
        if self._reads is None:
            self._reads = SSAReadTracker()
            for block in self.subroutine.body:
                for op in block.all_ops:
                    self._reads.add(op)
        return self._reads

    def _only_read_as_number(self, register: models.Register) -> bool:
        """True if everything that reads a register reads it as a number."""
        for read in self._tracked_reads().get(register):
            source = read.source if isinstance(read, models.Assignment) else read
            if not isinstance(source, models.Intrinsic) or source.op not in _READS_NUMBER:
                return False
        return True

    # operations that give back an operand

    def _operand_result(
        self,
        assignment: models.Assignment,
        source: models.Intrinsic,
        at: typing.Callable[[], ValueBounds],
    ) -> _Replacement | None:
        match source.op, source.args:
            case AVMOp.mod, [a, b]:
                bounds = at()
                if bounds.uint64(a)[1] < bounds.uint64(b)[0]:
                    return "% below modulus", a
            case AVMOp.bitwise_and, [a, b]:
                bounds = at()
                for value, mask in ((a, b), (b, a)):
                    if _covers(bounds.uint64(mask), bounds.uint64(value)):
                        return "& with all bits", value
            case AVMOp.mod_bytes, [a, b]:
                bounds = at()
                if bounds.number(a)[1] < bounds.number(b)[0] and not _may_fail(source, bounds):
                    # two byte arrays without leading zero bytes that are the same number
                    return self._same_number(
                        "b% below modulus", assignment, a, same_bytes=bounds.is_minimal(a)
                    )
            case AVMOp.bitwise_and_bytes, [a, b]:
                bounds = at()
                for value, mask in ((a, b), (b, a)):
                    if _covers(bounds.number(mask), bounds.number(value)) and (
                        replacement := self._same_number(
                            "b& with all bits",
                            assignment,
                            value,
                            same_bytes=_not_longer(bounds, mask, value),
                            other=(bounds, mask),
                        )
                    ):
                        return replacement
            case ((AVMOp.bitwise_or_bytes | AVMOp.bitwise_xor_bytes) as op, [a, b]):
                bounds = at()
                for value, zeros in ((a, b), (b, a)):
                    if bounds.number(zeros)[1] == 0 and (
                        replacement := self._same_number(
                            f"{op.code} with zeros",
                            assignment,
                            value,
                            same_bytes=_not_longer(bounds, zeros, value),
                            other=(bounds, zeros),
                        )
                    ):
                        return replacement
            case (
                (AVMOp.extract | AVMOp.extract3 | AVMOp.substring | AVMOp.substring3),
                [a, *_],
            ):
                if _takes_all(source, at):
                    return "whole byte array", a
        return None

    def _same_number(
        self,
        rule: str,
        assignment: models.Assignment,
        value: models.Value,
        *,
        same_bytes: bool,
        other: tuple[ValueBounds, models.Value] | None = None,
    ) -> _Replacement | None:
        """The replacement of an operation by an operand that is the same number as its
        result, if that is allowed: where it is the same bytes, or where nothing can tell."""
        if same_bytes:
            return rule, value
        (target,) = assignment.targets
        if not self._only_read_as_number(target):
            return None
        if other is not None:
            # the result is as long as the longer operand, and byte math fails on a long one
            bounds, operand = other
            if bounds.length(operand)[1] > MAX_BIGUINT_BYTES:
                return None
        self._other_bytes.add(target)
        return f"{rule}, as a number", value


def _outcome(source: models.Intrinsic, bounds: ValueBounds) -> _Replacement | None:
    outcome = bounds.outcome(source)
    # a comparison of byte math fails on an operand that is too long, whatever it would give
    if outcome is None or _may_fail(source, bounds):
        return None
    return "comparison", models.UInt64Constant(
        value=int(outcome), source_location=source.source_location
    )


def _may_fail(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    return Effect.fails in effects_of(op, bounds=bounds)


def _covers(mask: tuple[int, int], value: tuple[int, int]) -> bool:
    """True if a known mask has every bit that a value can have."""
    low, high = mask
    bits = (1 << value[1].bit_length()) - 1
    return low == high and low & bits == bits


def _not_longer(bounds: ValueBounds, a: models.Value, b: models.Value) -> bool:
    return bounds.length(a)[1] <= bounds.length(b)[0]


def _takes_all(source: models.Intrinsic, at: typing.Callable[[], ValueBounds]) -> bool:
    """True if an op that takes a part of a byte array takes all of it."""
    match source.op, source.args, source.immediates:
        case AVMOp.extract, [a], [0, int(length)]:
            # a length of 0 takes the rest
            return length == 0 or at().length(a) == (length, length)
        case AVMOp.substring, [a], [0, int(end)]:
            return at().length(a) == (end, end)
        case ((AVMOp.extract3 | AVMOp.substring3), [a, start, end], []):
            bounds = at()
            length, longest = bounds.length(a)
            return (
                bounds.uint64(start) == (0, 0)
                and length == longest
                and bounds.uint64(end) == (length, length)
            )
    return False
