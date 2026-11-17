from collections import Counter

from puya import log
from puya.context import ArtifactCompileContext
from puya.errors import InternalError
from puya.mir import models as mir
from puya.program_refs import ProgramKind
from puya.teal import models as teal_models
from puya.teal.builder import TealBuilder
from puya.teal.optimize.main import optimize_teal_program
from puya.teal.output import maybe_output_intermediate_teal

logger = log.get_logger(__name__)


def mir_to_teal(
    context: ArtifactCompileContext, program_mir: mir.Program
) -> teal_models.TealProgram:
    main = TealBuilder.build_subroutine(
        program_mir.main, slot_allocation=program_mir.slot_allocation
    )
    subroutines = [TealBuilder.build_subroutine(mir_sub) for mir_sub in program_mir.subroutines]
    if program_mir.autosalt is not None:
        autosalt = program_mir.autosalt
    else:
        # salt lsigs by default on any version
        # leave contracts unsalted by default
        autosalt = program_mir.kind is ProgramKind.logic_signature
    teal = teal_models.TealProgram(
        kind=program_mir.kind,
        avm_version=program_mir.avm_version,
        main=main,
        subroutines=subroutines,
        autosalt=autosalt,
    )
    maybe_output_intermediate_teal(context, teal, qualifier="lowered")
    initial_check_set = _collect_explicit_checks(teal)
    unreachable_check_set = optimize_teal_program(context, teal)
    post_allocation_check_set = _collect_explicit_checks(teal)
    _ensure_explicit_checks_kept(
        initial=initial_check_set,
        final=post_allocation_check_set,
        unreachable=unreachable_check_set,
    )
    return teal


_ExplicitChecks = Counter[tuple[str, teal_models.Assert | teal_models.Err]]


def _ensure_explicit_checks_kept(
    *, initial: _ExplicitChecks, final: _ExplicitChecks, unreachable: _ExplicitChecks
) -> None:
    """An explicit check must still be there after optimization, unless the optimizer
    removed it together with code that nothing can reach."""
    for check_data, initial_count in initial.items():
        # less than rather than != since we can duplicate ops for inlining
        if final.get(check_data, 0) + unreachable.get(check_data, 0) < initial_count:
            raise InternalError("explicit condition check(s) removed during TEAL optimization")


def _collect_explicit_checks(program: teal_models.TealProgram) -> _ExplicitChecks:
    return Counter(
        (str(sub.signature), op)
        for sub in program.all_subroutines
        for block in sub.blocks
        for op in block.ops
        if isinstance(op, teal_models.Assert | teal_models.Err) and op.explicit
    )
