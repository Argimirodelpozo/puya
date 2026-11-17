from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.eliminate_box_asserts import minimize_box_exist_asserts
from puya.ir.types_ import PrimitiveIRType
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    call,
    intrinsic,
    optimize_block,
    parameter,
    register,
    return_,
    subroutine,
    uint64,
)

_key = bytes_(b"key")
_other_key = bytes_(b"other")
_a = parameter("a")
_b = parameter("b")
_data = parameter("data", PrimitiveIRType.bytes)
_parameters = (_a, _b, _data)


def _sub(name: str, *ops: models.Op) -> models.Subroutine:
    return subroutine(name, [block(0, ops, return_())])


def _exists_flag(name: str, key: models.Value = _key) -> tuple[models.Op, models.Register]:
    length = register(f"{name}_length")
    exists = register(f"{name}_exists", PrimitiveIRType.bool)
    read = models.Assignment(
        targets=[length, exists], source=intrinsic(AVMOp.box_len, key), source_location=None
    )
    return read, exists


def _assert(condition: models.Value) -> models.Assert:
    return models.Assert(
        condition=condition, message="box exists", explicit=False, source_location=None
    )


def test_assert_of_box_known_to_exist_is_removed() -> None:
    read, exists = _exists_flag("box")
    remaining = optimize_block(
        minimize_box_exist_asserts,
        [intrinsic(AVMOp.box_put, _key, _data), read, _assert(exists)],
        parameters=_parameters,
    )
    assert remaining == [
        "(box_put 0x6b6579 data#0)",
        "let (box_length#0: uint64, box_exists#0: bool) = (box_len 0x6b6579)",
    ]


def test_second_assert_of_box_exists_is_removed() -> None:
    first_read, first_exists = _exists_flag("first")
    second_read, second_exists = _exists_flag("second")
    ops = [first_read, _assert(first_exists), second_read, _assert(second_exists)]
    remaining = optimize_block(minimize_box_exist_asserts, ops, parameters=_parameters)
    assert remaining == [str(op) for op in ops[:3]]


def test_box_exists_assert_is_kept_after_box_is_deleted_through_application_id() -> None:
    read, exists = _exists_flag("box")
    ops = [
        assign(register("created", PrimitiveIRType.bool), intrinsic(AVMOp.box_create, _key, _a)),
        # the application can be the one that is running
        assign(register("deleted", PrimitiveIRType.bool), intrinsic(AVMOp.app_box_del, _b, _key)),
        read,
        _assert(exists),
    ]
    remaining = optimize_block(minimize_box_exist_asserts, ops, parameters=_parameters)
    assert remaining == [str(op) for op in ops]


def test_box_exists_assert_is_kept_after_inner_transaction() -> None:
    read, exists = _exists_flag("box")
    ops = [
        assign(register("created", PrimitiveIRType.bool), intrinsic(AVMOp.box_create, _key, _a)),
        intrinsic(AVMOp.itxn_begin),
        # an application called here may share access to the boxes of this one
        intrinsic(AVMOp.itxn_submit),
        read,
        _assert(exists),
    ]
    remaining = optimize_block(minimize_box_exist_asserts, ops, parameters=_parameters)
    assert remaining == [str(op) for op in ops]


def test_box_exists_assert_is_kept_if_flag_was_read_before_box_was_deleted() -> None:
    first_read, first_exists = _exists_flag("first")
    second_read, second_exists = _exists_flag("second")
    ops = [
        first_read,
        assign(register("deleted", PrimitiveIRType.bool), intrinsic(AVMOp.box_del, _key)),
        # says the box did exist, not that it does
        _assert(first_exists),
        second_read,
        _assert(second_exists),
    ]
    remaining = optimize_block(minimize_box_exist_asserts, ops, parameters=_parameters)
    assert remaining == [str(op) for op in ops]


def test_box_exists_assert_is_kept_if_flag_was_read_before_box_was_created() -> None:
    read, exists = _exists_flag("box")
    ops = [
        read,
        assign(register("created", PrimitiveIRType.bool), intrinsic(AVMOp.box_create, _key, _a)),
        _assert(exists),
    ]
    remaining = optimize_block(minimize_box_exist_asserts, ops, parameters=_parameters)
    assert remaining == [str(op) for op in ops]


def test_box_exists_assert_and_calls() -> None:
    deleter = _sub(
        "deleter",
        assign(register("deleted", PrimitiveIRType.bool), intrinsic(AVMOp.box_del, _other_key)),
    )
    indirect_deleter = _sub("indirect_deleter", call(deleter))
    writer = _sub("writer", intrinsic(AVMOp.box_replace, _other_key, uint64(0), bytes_(b"new")))

    def after_call(target: models.Subroutine) -> list[str]:
        read, exists = _exists_flag("box")
        return optimize_block(
            minimize_box_exist_asserts,
            [intrinsic(AVMOp.box_put, _key, _data), call(target), read, _assert(exists)],
            deleter,
            indirect_deleter,
            writer,
            parameters=_parameters,
        )

    # a subroutine that may delete a box, itself or through another
    assert after_call(deleter)[-1] == "(assert box_exists#0) // box exists"
    assert after_call(indirect_deleter)[-1] == "(assert box_exists#0) // box exists"
    # a subroutine that changes boxes but cannot delete one
    assert after_call(writer)[-1].startswith("let (box_length#0: uint64, box_exists#0: bool)")
