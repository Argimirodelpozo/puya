from puya.teal import models
from puya.teal.optimize.constant_stack_shuffling import (
    constant_dup2_insertion,
    constant_dupn_insertion,
)
from puya.teal.optimize.peephole import peephole


def _global(field: str) -> models.TealOp:
    return models.Intrinsic(
        op_code="global",
        immediates=(field,),
        error_message=None,
        consumes=0,
        produces=1,
        source_location=None,
    )


def _block(*ops: models.TealOp, height: int) -> models.TealBlock:
    return models.TealBlock(
        label="block", ops=list(ops), x_stack_in=(), entry_stack_height=0, exit_stack_height=height
    )


def _teal(block: models.TealBlock) -> list[str]:
    return [op.teal() for op in block.ops]


def test_two_reads_of_the_opcode_budget_stay_two_reads() -> None:
    block = _block(_global("OpcodeBudget"), _global("OpcodeBudget"), height=2)
    assert not constant_dupn_insertion(block)
    assert _teal(block) == ["global OpcodeBudget", "global OpcodeBudget"]


def test_two_reads_of_a_fixed_global_are_one_read_and_a_dup() -> None:
    block = _block(_global("CurrentApplicationID"), _global("CurrentApplicationID"), height=2)
    assert constant_dupn_insertion(block)
    assert _teal(block) == ["global CurrentApplicationID", "dup"]


def test_a_repeated_pair_with_an_opcode_budget_read_is_not_a_dup2() -> None:
    one = models.Int(1, source_location=None)
    block = _block(_global("OpcodeBudget"), one, _global("OpcodeBudget"), one, height=4)
    assert not constant_dup2_insertion(block)
    assert _teal(block) == ["global OpcodeBudget", "int 1", "global OpcodeBudget", "int 1"]


def test_an_opcode_budget_read_is_not_moved_past_another_load() -> None:
    budget_then_one = _block(
        _global("OpcodeBudget"),
        models.Int(1, source_location=None),
        models.Swap(source_location=None),
        height=2,
    )
    peephole(budget_then_one, 2)
    assert _teal(budget_then_one) == ["global OpcodeBudget", "int 1", "swap"]
    fixed_then_one = _block(
        _global("CurrentApplicationID"),
        models.Int(1, source_location=None),
        models.Swap(source_location=None),
        height=2,
    )
    peephole(fixed_then_one, 2)
    assert _teal(fixed_then_one) == ["int 1", "global CurrentApplicationID"]
