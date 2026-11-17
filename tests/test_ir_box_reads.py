from collections.abc import Sequence

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.effects import BoxChanges, analyse_subroutine_effects
from puya.ir.optimize.state_reads import repeated_state_reads_elimination
from puya.ir.types_ import PrimitiveIRType
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    call,
    intrinsic,
    optimize_block,
    optimize_subroutine,
    parameter,
    program,
    register,
    return_,
    subroutine,
    uint64,
)

_key = bytes_(b"key")
_other_key = bytes_(b"other")
_name = parameter("name", PrimitiveIRType.bytes)
_size = parameter("size")
_data = parameter("data", PrimitiveIRType.bytes)
_flag = parameter("flag")
_parameters = (_name, _size, _data, _flag)

_SAME_LENGTH = ["let again#0: uint64 = first#0", "let again_exists#0: bool = first_exists#0"]
_SAME_PART = "let again#0: bytes = first#0"


def _bytes(name: str) -> models.Register:
    return register(name, PrimitiveIRType.bytes)


def _sub(name: str, *ops: models.Op) -> models.Subroutine:
    return subroutine(name, [block(0, ops, return_())])


def _length(name: str, key: models.Value = _key) -> models.Assignment:
    return models.Assignment(
        targets=[register(name), register(f"{name}_exists", PrimitiveIRType.bool)],
        source=intrinsic(AVMOp.box_len, key),
        source_location=None,
    )


def _part(name: str, start: int, size: int, key: models.Value = _key) -> models.Assignment:
    return assign(_bytes(name), intrinsic(AVMOp.box_extract, key, uint64(start), uint64(size)))


def _deleted(key: models.Value) -> models.Assignment:
    return assign(register("deleted", PrimitiveIRType.bool), intrinsic(AVMOp.box_del, key))


def _optimized(ops: Sequence[models.Op], *subroutines: models.Subroutine) -> list[str]:
    return optimize_block(
        repeated_state_reads_elimination, ops, *subroutines, parameters=_parameters
    )


def _length_again(*between: models.Op) -> tuple[list[str], list[models.Op]]:
    """The ops that are left of two reads of the length of a box with the given ops in
    between, from the second read on, and that read as it was."""
    ops = [_length("first"), *between, _length("again")]
    return _optimized(ops)[1 + len(between) :], ops[-1:]


def test_repeated_box_read_takes_the_earlier_result() -> None:
    left, _ = _length_again()
    assert left == _SAME_LENGTH
    ops = [_part("first", 0, 8), _part("again", 0, 8)]
    assert _optimized(ops) == [str(ops[0]), _SAME_PART]
    whole = models.Assignment(
        targets=[_bytes("again"), register("again_exists", PrimitiveIRType.bool)],
        source=intrinsic(AVMOp.box_get, _key),
        source_location=None,
    )
    put = intrinsic(AVMOp.box_put, _key, _data)
    # what was put in a box is what a read of the box gives
    assert _optimized([put, whole]) == [
        str(put),
        "let again#0: bytes = data#0",
        "let again_exists#0: bool = 1u",
    ]


def test_box_length_is_kept_at_an_op_that_cannot_change_it() -> None:
    # bytes replaced in the box itself
    left, _ = _length_again(intrinsic(AVMOp.box_replace, _key, _size, _data))
    assert left == _SAME_LENGTH
    # a box with another constant name
    left, _ = _length_again(intrinsic(AVMOp.box_put, _other_key, _data), _deleted(_other_key))
    assert left == _SAME_LENGTH
    left, _ = _length_again(intrinsic(AVMOp.log, _data))
    assert left == _SAME_LENGTH


def test_box_length_is_forgotten_at_an_op_that_may_change_it() -> None:
    for between in (
        _deleted(_key),
        intrinsic(AVMOp.box_put, _key, _data),
        # a name that is not a constant may be the name of this box
        _deleted(_name),
        # an application that is called can share access to the boxes of this one
        intrinsic(AVMOp.itxn_submit),
    ):
        left, read = _length_again(between)
        assert left == [str(op) for op in read], str(between)


def test_box_length_is_known_once_the_box_is_created_or_resized() -> None:
    again = _length("again")
    for op in (
        assign(
            register("created", PrimitiveIRType.bool), intrinsic(AVMOp.box_create, _key, _size)
        ),
        intrinsic(AVMOp.box_resize, _key, _size),
    ):
        assert _optimized([op, again]) == [
            str(op),
            "let again#0: uint64 = size#0",
            "let again_exists#0: bool = 1u",
        ]


def test_part_of_a_box_is_kept_unless_bytes_that_may_be_among_it_are_replaced() -> None:
    def last(between: models.Op) -> str:
        return _optimized([_part("first", 8, 8), between, _part("again", 8, 8)])[-1]

    eight_bytes = bytes_(b"12345678")
    # another part of the box
    assert last(intrinsic(AVMOp.box_replace, _key, uint64(16), eight_bytes)) == _SAME_PART
    assert last(intrinsic(AVMOp.box_replace, _key, uint64(0), eight_bytes)) == _SAME_PART
    # another box
    assert last(intrinsic(AVMOp.box_replace, _other_key, uint64(8), eight_bytes)) == _SAME_PART
    read_again = str(_part("again", 8, 8))
    # bytes among the ones read
    assert last(intrinsic(AVMOp.box_replace, _key, uint64(12), eight_bytes)) == read_again
    # bytes of unknown number, or at an unknown place, or of a box that may be this one
    assert last(intrinsic(AVMOp.box_replace, _key, uint64(0), _data)) == read_again
    assert last(intrinsic(AVMOp.box_replace, _key, _size, eight_bytes)) == read_again
    assert last(intrinsic(AVMOp.box_replace, _name, uint64(16), eight_bytes)) == read_again
    # the bytes that were put there are the ones that a read of that part gives
    assert last(intrinsic(AVMOp.box_replace, _key, uint64(8), eight_bytes)) == str(
        assign(_bytes("again"), eight_bytes)
    )


def test_call_forgets_what_the_called_subroutine_may_change() -> None:
    writes_other = _sub("writes_other", intrinsic(AVMOp.box_put, _other_key, bytes_(b"new")))
    writes_same = _sub("writes_same", _deleted(_key))
    name = parameter("name", PrimitiveIRType.bytes)
    writes_any = subroutine(
        "writes_any",
        [block(0, [intrinsic(AVMOp.box_put, name, bytes_(b"new"))], return_())],
        parameters=[name],
    )
    indirect = _sub("indirect", call(writes_other))
    subroutines = (writes_other, writes_same, writes_any, indirect)
    summaries = analyse_subroutine_effects(program(_sub("main"), *subroutines))
    assert summaries.box_changes(indirect) == BoxChanges(boxes=frozenset({b"other"}))
    assert summaries.box_changes(writes_any) == BoxChanges(other=True)

    def left(target: models.Subroutine, *args: models.Value) -> tuple[list[str], models.Op]:
        ops = [_length("first"), call(target, *args), _length("again")]
        return _optimized(ops, *subroutines)[2:], ops[-1]

    remaining, _ = left(indirect)
    assert remaining == _SAME_LENGTH
    remaining, read = left(writes_same)
    assert remaining == [str(read)]
    remaining, read = left(writes_any, _data)
    assert remaining == [str(read)]


def test_box_read_is_known_after_a_branch_if_it_is_known_on_every_way() -> None:
    def joined(then_op: models.Op) -> list[str]:
        entry = block(0, [_length("first")], return_())
        then_block = block(1, [then_op], return_())
        after = block(2, [_length("again")], return_())
        entry.terminator = models.ConditionalBranch(
            condition=_flag, non_zero=then_block, zero=after, source_location=None
        )
        then_block.terminator = models.Goto(target=after, source_location=None)
        then_block.add_predecessor(entry)
        after.add_predecessor(entry)
        after.add_predecessor(then_block)
        main = subroutine("main", [entry, then_block, after], parameters=_parameters)
        optimize_subroutine(repeated_state_reads_elimination, main)
        return [str(op) for op in after.ops]

    assert joined(intrinsic(AVMOp.box_replace, _key, _size, _data)) == _SAME_LENGTH
    assert joined(_deleted(_key)) == [str(_length("again"))]
