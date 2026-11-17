import functools
import typing
from collections.abc import Iterable

import attrs

from puya import log
from puya.context import CompileContext
from puya.ir import models, visitor
from puya.ir._puya_lib import PuyaLibIR
from puya.ir._utils import bfs_block_order
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import (
    OBSERVABLE,
    Effect,
    EffectSummaries,
    effects_of,
    lost_if_removed,
)
from puya.ir.optimize.value_bounds import ValueBounds
from puya.utils import StableSet

logger = log.get_logger(__name__)


@attrs.define
class SubroutineCollector(visitor.IRTraverser):
    subroutines: StableSet[models.Subroutine] = attrs.field(factory=StableSet)
    referenced_libs: StableSet[PuyaLibIR] = attrs.field(factory=StableSet)

    @classmethod
    def collect(cls, program: models.Program) -> StableSet[models.Subroutine]:
        collector = cls()
        collector.visit_subroutine(program.main)
        # also include any referenced library functions
        referenced_subs = [s for s in program.subroutines if s.id in collector.referenced_libs]
        for referenced_sub in referenced_subs:
            collector.visit_subroutine(referenced_sub)
        return collector.subroutines

    def visit_subroutine(self, subroutine: models.Subroutine) -> None:
        if subroutine not in self.subroutines:
            self.subroutines.add(subroutine)
            self.visit_all_blocks(subroutine.body)

    def visit_extract_value(self, _: models.ExtractValue) -> None:
        self.referenced_libs |= (
            PuyaLibIR.static_array_read_dynamic_element,
            PuyaLibIR.static_array_read_byte_length_element,
            PuyaLibIR.dynamic_array_read_dynamic_element,
            PuyaLibIR.dynamic_array_read_byte_length_element,
            PuyaLibIR.dynamic_assert_index,
        )

    def visit_replace_value(self, _: models.ReplaceValue) -> None:
        self.referenced_libs |= (
            PuyaLibIR.dynamic_array_replace_byte_length_head,
            PuyaLibIR.dynamic_array_replace_dynamic_element,
            PuyaLibIR.static_array_replace_byte_length_head,
            PuyaLibIR.static_array_replace_dynamic_element,
            PuyaLibIR.static_array_read_dynamic_element,
            PuyaLibIR.static_array_read_byte_length_element,
            PuyaLibIR.dynamic_array_read_dynamic_element,
            PuyaLibIR.dynamic_array_read_byte_length_element,
            PuyaLibIR.dynamic_assert_index,
        )

    def visit_array_pop(self, _: models.ArrayPop) -> None:
        self.referenced_libs |= (
            PuyaLibIR.dynamic_array_pop_bit,
            PuyaLibIR.r_trim,
            PuyaLibIR.dynamic_array_pop_fixed_size,
            PuyaLibIR.dynamic_array_pop_byte_length_head,
            PuyaLibIR.dynamic_array_pop_dynamic_element,
            PuyaLibIR.box_dynamic_array_pop_fixed_size,
            PuyaLibIR.box_dynamic_array_pop_bit,
            PuyaLibIR.box_update_offset_dec,
        )

    def visit_array_concat(self, _: models.ArrayConcat) -> None:
        self.referenced_libs |= (
            PuyaLibIR.dynamic_array_concat_fixed,
            PuyaLibIR.dynamic_array_concat_dynamic_element,
            PuyaLibIR.dynamic_array_concat_bits,
            PuyaLibIR.dynamic_array_concat_byte_length_head,
            PuyaLibIR.box_dynamic_array_concat_fixed,
            PuyaLibIR.box_dynamic_array_concat_bits,
            PuyaLibIR.box_update_offset_inc,
        )

    def visit_invoke_subroutine(self, callsub: models.InvokeSubroutine) -> None:
        self.visit_subroutine(callsub.target)


def remove_unused_subroutines(program: models.Program) -> bool:
    subroutines = SubroutineCollector.collect(program)
    if modified := (len(subroutines) != (1 + len(program.subroutines))):
        to_keep = [p for p in program.subroutines if p in subroutines]
        for p in program.subroutines:
            if p not in subroutines:
                logger.debug(f"removing unused subroutine {p.id}")
        program.subroutines = to_keep
    return modified


# Typed operations that are removed when unused although they can fail.
# - A box read fails if the box does not exist. Reads of parts of a box are combined into
#   operations on the box itself, which leaves the read of the whole box unused, and a box
#   can be larger than a value may be.
# - The others fail on an index out of bounds, on a value that does not fit its encoding, or
#   on a value that is not encoded as its type says. Merging chained reads and removing
#   encode/decode round trips leave unused operations of these kinds behind.
_REMOVED_ALTHOUGH_FALLIBLE: typing.Final = (
    models.BoxRead,
    models.ExtractValue,
    models.ReplaceValue,
    models.DecodeBytes,
    models.BytesEncode,
    models.ArrayLength,
    models.ArrayPop,
    models.ArrayConcat,
)


def remove_unused_variables(context: IROptimizationContext, subroutine: models.Subroutine) -> bool:
    """Removes assignments that nothing reads, unless the program would lose something other
    than the values: an effect, or a failure that the operands do not rule out."""
    modified = 0
    assignments = dict[tuple[models.BasicBlock, models.Assignment], set[models.Register]]()
    for block, op, register in UnusedRegisterCollector.collect(subroutine):
        if isinstance(op, models.Assignment):
            assignments.setdefault((block, op), set()).add(register)
        else:
            assert register == op.register
            block.phis.remove(op)
            logger.debug(f"Removing unused variable {register.local_id}")
            modified += 1

    @functools.cache
    def bounds() -> ValueBounds:
        return ValueBounds.of(subroutine)

    for (block, ass), registers in assignments.items():
        if registers.symmetric_difference(ass.targets):
            continue  # some registers still used
        lost = lost_if_removed(ass.source, summaries=context.effects, bounds=bounds)
        if lost == Effect.fails and isinstance(ass.source, models.Intrinsic):
            # what is known to hold whenever the operation runs may rule the failure out
            non_zero, zero = _conditions_established(subroutine, block, ass)
            completed = _related_operations(context.effects, bounds(), block, ass, ass.source)
            if non_zero or zero or completed:
                given = bounds().given(non_zero=non_zero, zero=zero, completed=completed)
                lost = effects_of(ass.source, bounds=given) & OBSERVABLE
        elif lost == Effect.fails and isinstance(ass.source, _REMOVED_ALTHOUGH_FALLIBLE):
            lost = Effect.none
        if not lost:
            for reg in sorted(registers, key=lambda r: r.local_id):
                logger.debug(f"Removing unused variable {reg.local_id}")
            block.ops.remove(ass)
            modified += 1
        else:
            logger.debug(
                f"Not removing unused assignment since its source has effects"
                f" ({lost.name}): {ass}"
            )
    return modified > 0


# how many blocks back the conditions that hold for an operation are collected from
_MAX_CONDITION_BLOCKS: typing.Final = 16


def _conditions_established(
    subroutine: models.Subroutine, block: models.BasicBlock, op: models.Op
) -> tuple[list[models.Value], list[models.Value]]:
    """The values that are not 0, and the values that are 0, whenever an op of a block runs:
    the conditions of the asserts that passed before it, and of the branches taken to it."""
    non_zero = list[models.Value]()
    zero = list[models.Value]()
    ops_before = block.ops[: block.ops.index(op)]
    for _ in range(_MAX_CONDITION_BLOCKS):
        non_zero.extend(o.condition for o in ops_before if isinstance(o, models.Assert))
        # a block that is entered from one block only runs after all the ops of that block.
        # The first block of a subroutine is also entered by the call
        if block is subroutine.entry:
            break
        try:
            (predecessor,) = block.predecessors
        except ValueError:
            break
        match predecessor.terminator:
            case models.ConditionalBranch(condition=condition, non_zero=taken, zero=other) if (
                taken is block and other is not block
            ):
                non_zero.append(condition)
            case models.ConditionalBranch(condition=condition, non_zero=other, zero=taken) if (
                taken is block and other is not block
            ):
                zero.append(condition)
        block = predecessor
        ops_before = block.ops
    return non_zero, zero


def _related_operations(
    summaries: EffectSummaries | None,
    bounds: ValueBounds,
    block: models.BasicBlock,
    op: models.Op,
    intrinsic: models.Intrinsic,
) -> list[models.Intrinsic]:
    """The operations of the block that have a say about the operands of an unused operation,
    and that either have completed when it runs or are certain to run after it.

    If the unused operation can only fail when one of these fails as well, the program fails
    with or without it: an operation that follows it in its block runs unless the program has
    failed by then, as long as nothing in between can end the program in another way.
    """
    operands = {arg for arg in intrinsic.args if isinstance(arg, models.Register)}
    # and what the operands are computed from, as in (a + 1) + 1 next to a + 2
    for operand in list(operands):
        operands.update(bounds.operands_of_definition(operand))
    if not operands:
        return []
    related = list[models.Intrinsic]()
    after = False
    for other in block.ops:
        if other is op:
            after = True
            continue
        source = other.source if isinstance(other, models.Assignment) else other
        if isinstance(source, models.Intrinsic):
            if not operands.isdisjoint(source.args):
                related.append(source)
        elif after and Effect.exits in effects_of(source, summaries=summaries):
            break
    return related


@attrs.define(kw_only=True)
class UnusedRegisterCollector(visitor.IRTraverser):
    used: set[models.Register] = attrs.field(factory=set)
    assigned: dict[models.Register, tuple[models.BasicBlock, models.Assignment | models.Phi]] = (
        attrs.field(factory=dict)
    )
    active_block: models.BasicBlock

    @classmethod
    def collect(
        cls, sub: models.Subroutine
    ) -> Iterable[tuple[models.BasicBlock, models.Assignment | models.Phi, models.Register]]:
        collector = cls(active_block=sub.entry)
        collector.visit_all_blocks(sub.body)
        for reg, (block, ass) in collector.assigned.items():
            if reg not in collector.used:
                yield block, ass, reg

    @typing.override
    def visit_block(self, block: models.BasicBlock) -> None:
        self.active_block = block
        super().visit_block(block)

    @typing.override
    def visit_assignment(self, ass: models.Assignment) -> None:
        for target in ass.targets:
            self.assigned[target] = (self.active_block, ass)
        ass.source.accept(self)

    @typing.override
    def visit_phi(self, phi: models.Phi) -> None:
        # don't visit phi.register, as this would mean the phi can never be considered unused
        for arg in phi.args:
            arg.accept(self)
        self.assigned[phi.register] = (self.active_block, phi)

    @typing.override
    def visit_register(self, reg: models.Register) -> None:
        self.used.add(reg)


def remove_unreachable_blocks(_context: CompileContext, subroutine: models.Subroutine) -> bool:
    reachable_set = frozenset(bfs_block_order(subroutine.entry))
    if len(reachable_set) == len(subroutine.body):
        return False

    reachable_blocks = [subroutine.body[0]]
    for block in subroutine.body[1:]:
        if block in reachable_set:
            reachable_blocks.append(block)
        else:
            logger.debug(f"Removing unreachable block: {block}")
            for succ in block.successors:
                if succ in reachable_set:
                    did_remove = succ.remove_predecessor(block)
                    assert did_remove

    subroutine.body[:] = reachable_blocks
    return True
