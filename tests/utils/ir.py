"""Shorthand for building IR by hand, for tests of optimizations that work on IR."""

from collections.abc import Callable, Sequence
from pathlib import Path
from unittest.mock import Mock

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import analyse_subroutine_effects
from puya.ir.types_ import AVMBytesEncoding, IRType, PrimitiveIRType
from puya.parse import SourceLocation
from puya.program_refs import ProgramKind

_location = SourceLocation(file=Path("/test_ir.py"), line=1)


def uint64(value: int) -> models.UInt64Constant:
    return models.UInt64Constant(value=value, source_location=None)


def bytes_(value: bytes) -> models.BytesConstant:
    return models.BytesConstant(
        value=value, encoding=AVMBytesEncoding.base16, source_location=None
    )


def biguint(value: int) -> models.BigUIntConstant:
    return models.BigUIntConstant(value=value, source_location=None)


def register(name: str, ir_type: IRType = PrimitiveIRType.uint64) -> models.Register:
    return models.Register(name=name, version=0, ir_type=ir_type, source_location=None)


def parameter(name: str, ir_type: IRType = PrimitiveIRType.uint64) -> models.Parameter:
    return models.Parameter(
        name=name, version=0, ir_type=ir_type, source_location=None, implicit_return=False
    )


def intrinsic(
    avm_op: AVMOp,
    *args: models.Value,
    immediates: Sequence[str | int] = (),
    types: Sequence[IRType] | None = None,
) -> models.Intrinsic:
    if types is None:
        return models.Intrinsic(
            op=avm_op, args=list(args), immediates=list(immediates), source_location=None
        )
    return models.Intrinsic(
        op=avm_op, args=list(args), immediates=list(immediates), types=types, source_location=None
    )


def assign(target: models.Register, source: models.ValueProvider) -> models.Assignment:
    return models.Assignment(targets=[target], source=source, source_location=None)


def block(
    block_id: int, ops: Sequence[models.Op], terminator: models.ControlOp
) -> models.BasicBlock:
    return models.BasicBlock(
        id=block_id, source_location=_location, ops=list(ops), terminator=terminator
    )


def return_(*values: models.Value) -> models.SubroutineReturn:
    return models.SubroutineReturn(result=list(values), source_location=None)


def goto(block: models.BasicBlock, target: models.BasicBlock) -> None:
    block.terminator = models.Goto(target=target, source_location=None)
    target.add_predecessor(block)


def subroutine(
    name: str,
    body: Sequence[models.BasicBlock],
    *,
    parameters: Sequence[models.Parameter] = (),
    returns: Sequence[IRType] = (),
) -> models.Subroutine:
    return models.Subroutine(
        id=name,
        short_name=name,
        source_location=None,
        parameters=parameters,
        returns=returns,
        body=list(body),
        inline=None,
    )


def call(target: models.Subroutine, *args: models.Value) -> models.InvokeSubroutine:
    return models.InvokeSubroutine(target=target, args=list(args), source_location=None)


def program(main: models.Subroutine, *subroutines: models.Subroutine) -> models.Program:
    return models.Program(
        kind=ProgramKind.approval,
        main=main,
        subroutines=subroutines,
        avm_version=11,
        slot_allocation=models.SlotAllocation(
            reserved=set(), strategy=models.SlotAllocationStrategy.none
        ),
    )


def optimize_block(
    optimization: Callable[[IROptimizationContext, models.Subroutine], bool],
    ops: Sequence[models.Op],
    *subroutines: models.Subroutine,
    parameters: Sequence[models.Parameter] = (),
) -> list[str]:
    """Runs an optimization over a block of ops that makes up the main subroutine of a
    program, and returns the ops that are left, as text."""
    main = subroutine("main", [block(0, ops, return_())], parameters=parameters)
    context = Mock(effects=analyse_subroutine_effects(program(main, *subroutines)))
    optimization(context, main)
    main.validate_with_ssa()
    return [str(op) for op in main.body[0].ops]
