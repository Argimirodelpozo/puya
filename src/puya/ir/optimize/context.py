from collections.abc import Mapping

import attrs

from puya.context import ArtifactCompileContext
from puya.ir import models
from puya.ir._puya_lib import PuyaLibIR


@attrs.frozen
class ScratchEffects:
    writes: bool
    """can store to a scratch slot, itself or through a callee"""
    repeatable: bool
    """runs only operations whose outcome is fixed by their arguments and by scratch space,
    so a second identical call with scratch space unchanged ends the same way as the first"""


@attrs.define(kw_only=True)
class IROptimizationContext(ArtifactCompileContext):
    expand_all_bytes: bool

    inlineable_calls: set[tuple[str, str]] = attrs.field(factory=set)
    """src -> dst pairs that can/should be inlined"""
    constant_with_constant_args: dict[str, bool] = attrs.field(factory=dict)
    embedded_funcs: Mapping[PuyaLibIR, models.Subroutine]
    scratch_effects: Mapping[str, ScratchEffects] | None = None
    """by subroutine id; None until analysed"""
    program_subroutines: Mapping[str, models.Subroutine] | None = None
    """by subroutine id; None until analysed"""
