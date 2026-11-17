from puya.teal import models
from puya.teal.optimize.repeated_rotations_search import repeated_rotation_ops_search


def _block(*ops: models.TealOp) -> models.TealBlock:
    return models.TealBlock(
        label="block", ops=list(ops), x_stack_in=(), entry_stack_height=3, exit_stack_height=3
    )


def _record(number: int) -> models.StackManipulation:
    """A stack manipulation that can be told from the others."""
    return models.StackExtend([f"value_{number}"])


def _rotations_that_cancel() -> list[models.TealOp]:
    """Four rotations that leave the top three values of the stack where they were, each with
    a stack manipulation of its own."""
    return [
        models.Cover(2, source_location=None, stack_manipulations=[_record(0)]),
        models.Uncover(1, source_location=None, stack_manipulations=[_record(1)]),
        models.Cover(2, source_location=None, stack_manipulations=[_record(2)]),
        models.Cover(1, source_location=None, stack_manipulations=[_record(3)]),
    ]


_RUN_MANIPULATIONS = tuple(_record(number) for number in range(4))


def _teal(block: models.TealBlock) -> list[str]:
    return [op.teal() for op in block.ops]


def test_run_that_cancels_out_gives_its_stack_manipulations_to_the_op_before() -> None:
    before = models.Int(1, source_location=None, stack_manipulations=[models.StackExtend(["one"])])
    block = _block(before, *_rotations_that_cancel(), models.Pop(source_location=None))
    repeated_rotation_ops_search(block)
    assert _teal(block) == ["int 1", "pop"]
    assert block.ops[0].stack_manipulations == (
        models.StackExtend(["one"]),
        *_RUN_MANIPULATIONS,
    )
    assert block.ops[1].stack_manipulations == ()


def test_run_that_cancels_out_at_the_start_gives_them_to_the_op_after() -> None:
    after = models.Pop(source_location=None, stack_manipulations=[models.StackConsume(1)])
    block = _block(*_rotations_that_cancel(), after)
    repeated_rotation_ops_search(block)
    assert _teal(block) == ["pop"]
    assert block.ops[0].stack_manipulations == (*_RUN_MANIPULATIONS, models.StackConsume(1))


def test_block_of_nothing_but_a_run_that_cancels_out_keeps_ops_for_its_manipulations() -> None:
    block = _block(*_rotations_that_cancel())
    repeated_rotation_ops_search(block)
    # the last three rotations are one, which takes their stack manipulations
    assert _teal(block) == ["cover 2", "uncover 2"]
    assert block.ops[0].stack_manipulations == (_record(0),)
    assert block.ops[1].stack_manipulations == _RUN_MANIPULATIONS[1:]


def test_run_with_a_shorter_equivalent_is_replaced_between_other_ops() -> None:
    block = _block(
        models.Int(1, source_location=None),
        models.Uncover(2, source_location=None, stack_manipulations=[_record(0)]),
        models.Uncover(2, source_location=None, stack_manipulations=[_record(1)]),
        models.Pop(source_location=None),
        # a single rotation is left alone
        models.Cover(1, source_location=None),
        models.Pop(source_location=None),
    )
    repeated_rotation_ops_search(block)
    assert _teal(block) == ["int 1", "cover 2", "pop", "cover 1", "pop"]
    assert block.ops[1].stack_manipulations == (_record(0), _record(1))
