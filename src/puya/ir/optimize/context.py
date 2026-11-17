from collections.abc import Mapping

import attrs

from puya.context import ArtifactCompileContext
from puya.ir import models
from puya.ir._puya_lib import PuyaLibIR
from puya.ir.optimize.effects import EffectSummaries
from puya.ir.optimize.slot_storage import SlotWrite


@attrs.frozen(kw_only=True)
class StateEffects:
    """What a subroutine does to state that outlives it, itself or through a callee."""

    reads: frozenset[str] = frozenset()
    """the kinds of state its outcome depends on"""
    writes: frozenset[str] = frozenset()
    """the kinds of state it can change, other than through declared slot storage"""
    slot_writes: frozenset[SlotWrite] = frozenset()
    """its writes to declared slot storage"""
    repeatable: bool = True
    """has no effect of its own, and its outcome is fixed by its arguments and by the state in
    `reads`: a second identical call with that state unchanged ends the same way as the first"""


@attrs.define(kw_only=True)
class IROptimizationContext(ArtifactCompileContext):
    expand_all_bytes: bool

    inlineable_calls: set[tuple[str, str]] = attrs.field(factory=set)
    """src -> dst pairs that can/should be inlined"""
    constant_with_constant_args: dict[str, bool] = attrs.field(factory=dict)
    embedded_funcs: Mapping[PuyaLibIR, models.Subroutine]
    effects: EffectSummaries | None = None
    """what a call to each subroutine may do, as of the start of the current optimization pass.
    None until analysed, in which case a call may do everything"""
    state_effects: Mapping[str, StateEffects] | None = None
    """by subroutine id; None until analysed"""
    storage_roles: Mapping[str, str] = attrs.field(factory=dict)
    """subroutine id -> its role in declared slot storage"""
    pages_are_boxes: bool = False
    """whether the pages of declared slot storage are boxes of the program itself"""
