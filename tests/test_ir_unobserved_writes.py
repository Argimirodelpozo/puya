from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.repeated_loads_elimination import (
    constant_reads_and_unobserved_writes_elimination,
)
from puya.ir.types_ import PrimitiveIRType, SlotType
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


def _harmless() -> models.Subroutine:
    param = parameter("param")
    result = register("result")
    return subroutine(
        "harmless",
        [
            block(
                0,
                [assign(result, intrinsic(AVMOp.mod, param, uint64(7)))],
                return_(result),
            )
        ],
        parameters=[param],
        returns=[PrimitiveIRType.uint64],
    )


def _sub(name: str, *ops: models.Op) -> models.Subroutine:
    return subroutine(name, [block(0, ops, return_())])


def test_box_write_is_kept_if_next_write_has_another_size() -> None:
    # with no box, the second put fails, and with a box of 2 bytes, the first one does
    ops = [
        intrinsic(AVMOp.box_put, _key, bytes_(b"1")),
        intrinsic(AVMOp.box_put, _key, bytes_(b"22")),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(op) for op in ops]


def test_box_write_is_kept_if_size_of_next_write_is_unknown() -> None:
    ops = [
        intrinsic(AVMOp.box_put, _key, bytes_(b"1")),
        intrinsic(AVMOp.box_put, _key, _data),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(op) for op in ops]


def test_box_write_is_removed_if_next_write_has_same_size() -> None:
    first = register("first", PrimitiveIRType.bytes)
    second = register("second", PrimitiveIRType.bytes)
    ops = [
        assign(first, intrinsic(AVMOp.itob, _a)),
        assign(second, intrinsic(AVMOp.itob, _b)),
        intrinsic(AVMOp.box_put, _key, first),
        intrinsic(AVMOp.box_put, _key, second),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(op) for op in [*ops[:2], ops[3]]]


def test_box_write_is_kept_if_minimum_balance_is_read_before_next_write() -> None:
    address = register("address", PrimitiveIRType.account)
    ops = [
        assign(address, intrinsic(AVMOp.global_, immediates=["CurrentApplicationAddress"])),
        intrinsic(AVMOp.box_put, _key, bytes_(b"1")),
        # sees whether the box has been created
        assign(register("minimum"), intrinsic(AVMOp.min_balance, address)),
        intrinsic(AVMOp.box_put, _key, bytes_(b"2")),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(op) for op in ops]


def test_state_write_is_kept_if_inner_transaction_runs_before_next_write() -> None:
    ops = [
        intrinsic(AVMOp.app_global_put, _key, uint64(1)),
        intrinsic(AVMOp.itxn_begin),
        # an application called here can read the state of this one
        intrinsic(AVMOp.itxn_submit),
        intrinsic(AVMOp.app_global_put, _key, uint64(0)),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(op) for op in ops]


def test_state_write_is_removed_if_overwritten_by_value_of_same_type() -> None:
    ops = [
        intrinsic(AVMOp.app_global_put, _key, _a),
        intrinsic(AVMOp.app_global_put, _key, _b),
        intrinsic(AVMOp.app_local_put, uint64(0), _key, bytes_(b"12")),
        intrinsic(AVMOp.app_local_put, uint64(0), _key, bytes_(b"345")),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(ops[1]), str(ops[3])]


def test_state_write_is_kept_if_next_write_may_fit_where_it_does_not() -> None:
    ops = [
        # counts against the byte slices of the schema, the next one against the uints
        intrinsic(AVMOp.app_global_put, _key, _data),
        intrinsic(AVMOp.app_global_put, _key, _a),
        # may be longer than the next one, and too long
        intrinsic(AVMOp.app_local_put, uint64(0), _key, _data),
        intrinsic(AVMOp.app_local_put, uint64(0), _key, bytes_(b"345")),
    ]
    remaining = optimize_block(
        constant_reads_and_unobserved_writes_elimination, ops, parameters=_parameters
    )
    assert remaining == [str(op) for op in ops]


def test_writes_and_reads_across_calls() -> None:
    slot = parameter("slot", SlotType(PrimitiveIRType.uint64))
    harmless = _harmless()
    scratch_reader = _sub(
        "scratch_reader",
        assign(
            register("loaded"),
            intrinsic(AVMOp.loads, uint64(3), types=[PrimitiveIRType.uint64]),
        ),
    )
    state_writer = _sub("state_writer", intrinsic(AVMOp.app_global_put, _other_key, uint64(1)))
    exiting = subroutine(
        "exiting", [block(0, [], models.ProgramExit(result=uint64(1), source_location=None))]
    )

    def around_call(target: models.Subroutine, *args: models.Value) -> list[str]:
        return optimize_block(
            constant_reads_and_unobserved_writes_elimination,
            [
                models.WriteSlot(slot=slot, value=uint64(1), source_location=None),
                assign(
                    register("before"),
                    intrinsic(AVMOp.app_global_get, _key, types=[PrimitiveIRType.uint64]),
                ),
                call(target, *args),
                models.WriteSlot(slot=slot, value=uint64(2), source_location=None),
                assign(
                    register("after"),
                    intrinsic(AVMOp.app_global_get, _key, types=[PrimitiveIRType.uint64]),
                ),
            ],
            harmless,
            scratch_reader,
            state_writer,
            exiting,
            parameters=[slot],
        )

    first_write = "store(slot#0, 1u)"
    repeated_read = "let after#0: uint64 = (app_global_get 0x6b6579)"
    forwarded_read = "let after#0: uint64 = before#0"

    # a call that neither reads nor writes any state changes nothing
    remaining = around_call(harmless, uint64(1))
    assert first_write not in remaining
    assert forwarded_read in remaining
    # a call that reads scratch space may see the first write
    remaining = around_call(scratch_reader)
    assert first_write in remaining
    assert forwarded_read in remaining
    # a call that writes global state may change what is read
    remaining = around_call(state_writer)
    assert first_write not in remaining
    assert repeated_read in remaining
    # a call that may end the program leaves the first write in place
    remaining = around_call(exiting)
    assert first_write in remaining
    assert forwarded_read in remaining
