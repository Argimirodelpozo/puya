from collections.abc import Mapping

import attrs

from puya.context import ArtifactCompileContext
from puya.ir import models
from puya.ir._puya_lib import PuyaLibIR


@attrs.frozen(kw_only=True)
class SlotWrite:
    """A write to declared slot storage (see `puya.options.SlotStorage`), as far as it can be
    described without the values of the subroutine it is in. With neither `constant` nor
    `ending`, it can be to any slot."""

    constant: int | None = None
    """the slot, if it is a constant number"""
    count: int = 1
    """how many slots from a constant one on it can reach"""
    ending: bytes | None = None
    """for a slot that is the result of a hash op plus a number below 2**64: the constant bytes
    the hashed bytes end in"""


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
    state_effects: Mapping[str, StateEffects] | None = None
    """by subroutine id; None until analysed"""
    storage_roles: Mapping[str, str] = attrs.field(factory=dict)
    """subroutine id -> its role in declared slot storage"""
    pages_are_boxes: bool = False
    """whether the pages of declared slot storage are boxes of the program itself"""
