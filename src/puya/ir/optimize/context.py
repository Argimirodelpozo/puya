from collections.abc import Mapping

import attrs

from puya.context import ArtifactCompileContext
from puya.ir import models
from puya.ir._puya_lib import PuyaLibIR
from puya.ir.optimize.effects import EffectSummaries


@attrs.define(kw_only=True)
class IROptimizationContext(ArtifactCompileContext):
    expand_all_bytes: bool

    inlineable_calls: set[tuple[str, str]] = attrs.field(factory=set)
    """src -> dst pairs that can/should be inlined"""
    constant_with_constant_args: dict[str, bool] = attrs.field(factory=dict)
    embedded_funcs: Mapping[PuyaLibIR, models.Subroutine]
    modifications: dict[str, int] = attrs.field(factory=dict)
    """how many times an optimization has modified each subroutine, by the id of the
    subroutine"""
    effects: EffectSummaries | None = None
    """what a call to each subroutine may do, as of the start of the current optimization pass.
    None until analysed, in which case a call may do everything"""
