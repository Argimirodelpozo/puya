"""Rewrites that the ranges of the values of a subroutine allow.

The ranges are those of `puya.ir.optimize.value_bounds`, for the point of each operation: what
the definitions say, the conditions of the asserts and branches that dominate the point, and
what it took for the operations before it to complete. `range_rewrites` is the optimization,
which goes through a subroutine once. It has one group of rewrites so far, which can be
switched off by its name like an optimization:

`range_constant_outcomes`
    A comparison whose result the ranges fix becomes that result, an assert whose condition
    they show to hold is removed, and a branch whose condition they decide gets a constant
    condition. The proof is the range, so what established the range stays where it is: the
    assert or branch whose condition says so, or the operation that fails otherwise.

Besides the result being the same value, no failure may be lost: a comparison is only replaced
if the ranges show that it cannot fail.
"""

import enum
import typing
from collections.abc import Container, Mapping

import attrs

from puya import log
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import Effect, effects_of
from puya.ir.optimize.value_bounds import BoundsAtPoints, ValueBounds

logger = log.get_logger(__name__)


class _Group(enum.Enum):
    constant_outcomes = "range_constant_outcomes"


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

_Replacement = tuple[str, models.ValueProvider]


@attrs.define
class _Rewriter:
    subroutine: models.Subroutine
    groups: Container[_Group]
    modified: int = attrs.field(default=0, init=False)
    _points: BoundsAtPoints = attrs.field(init=False)

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
                    case models.Assignment(
                        targets=[_], source=models.Intrinsic(op=avm_op) as source
                    ) if outcomes and avm_op in _DECIDABLE:
                        replacement = _outcome(source, bounds())
                        if replacement is not None:
                            rule, decided = replacement
                            logger.debug(f"range rewrite ({rule}): {source} -> {decided}")
                            op.source = decided
                            self.modified += 1
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
