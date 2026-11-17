import attrs

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.state_reads import repeated_state_reads_elimination
from puya.ir.types_ import PrimitiveIRType
from puya.options import SlotStorage
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
from tests.utils.slot_storage import (
    PAGE,
    STORAGE,
    read_page_word,
    read_word,
    slot_add,
    word,
    write_page_word,
    write_words,
)

_read_page = read_page_word()
_write_page = write_page_word()
_read = read_word()
_write = write_words()
_add = slot_add()
_declared = (_read_page, _write_page, _read, _write, _add)

_slot = parameter("slot", PrimitiveIRType.bytes)
_other_slot = parameter("other_slot", PrimitiveIRType.bytes)
_value = parameter("value", PrimitiveIRType.bytes)
_parameters = (_slot, _other_slot, _value)

_FORWARDED = "let again#0: bytes = first#0"


def _bytes(name: str) -> models.Register:
    return register(name, PrimitiveIRType.bytes)


def _sub(name: str, *ops: models.Op) -> models.Subroutine:
    return subroutine(name, [block(0, ops, return_())])


def _optimized(
    ops: list[models.Op], *subroutines: models.Subroutine, storage: SlotStorage | None = STORAGE
) -> list[str]:
    return optimize_block(
        repeated_state_reads_elimination,
        ops,
        *_declared,
        *subroutines,
        parameters=_parameters,
        storage=storage,
    )


def _read_again(*between: models.Op, slot: models.Value = _slot) -> list[models.Op]:
    """A word read, the given ops, and a read of the same word."""
    return [
        assign(_bytes("first"), call(_read, slot)),
        *between,
        assign(_bytes("again"), call(_read, slot)),
    ]


def test_repeated_word_read_takes_the_word_read_before() -> None:
    ops = _read_again()
    assert _optimized(ops) == [str(ops[0]), _FORWARDED]


def test_repeated_page_word_read_takes_the_word_read_before() -> None:
    ops = [
        assign(_bytes("first"), call(_read_page, PAGE, uint64(32))),
        assign(_bytes("other"), call(_read_page, PAGE, uint64(64))),
        assign(_bytes("again"), call(_read_page, PAGE, uint64(32))),
    ]
    assert _optimized(ops) == [str(ops[0]), str(ops[1]), _FORWARDED]


def test_word_read_takes_the_word_written_before() -> None:
    first = _bytes("first")
    changed = _bytes("changed")
    ops = [
        assign(first, call(_read, _slot)),
        assign(changed, intrinsic(AVMOp.replace2, first, bytes_(b"\x01"), immediates=[31])),
        call(_write, _slot, changed),
        assign(_bytes("again"), call(_read, _slot)),
    ]
    assert _optimized(ops) == [*map(str, ops[:3]), "let again#0: bytes = changed#0"]


def test_word_read_takes_a_shorter_value_written_before_with_zeros_in_front() -> None:
    ops = [
        call(_write_page, PAGE, uint64(32), _value),
        assign(_bytes("again"), call(_read_page, PAGE, uint64(32))),
    ]
    zeros = _bytes("again%zeros")
    assert _optimized(ops) == [
        str(ops[0]),
        str(assign(zeros, intrinsic(AVMOp.bzero, uint64(32)))),
        str(assign(_bytes("again"), intrinsic(AVMOp.bitwise_or_bytes, zeros, _value))),
    ]


def test_word_read_takes_the_first_word_of_several_written_before() -> None:
    ops = [
        call(_write, _slot, _value),
        assign(_bytes("again"), call(_read, _slot)),
    ]
    assert _optimized(ops) == [
        str(ops[0]),
        str(assign(_bytes("again"), intrinsic(AVMOp.extract, _value, immediates=[0, 32]))),
    ]


def test_word_is_kept_at_a_write_to_another_slot() -> None:
    near = _bytes("near")
    # the same base at another distance
    ops = _read_again(assign(near, call(_add, _slot, uint64(1))), call(_write, near, word(1)))
    assert _optimized(ops)[-1] == _FORWARDED
    # another constant, out of reach of the words written
    ops = _read_again(call(_write, word(6), _value), slot=word(5))
    assert _optimized(ops)[-1] == _FORWARDED


def test_word_is_forgotten_at_a_write_to_a_slot_that_may_be_the_same() -> None:
    ops = _read_again(call(_write, _other_slot, word(1)))
    assert _optimized(ops) == [str(op) for op in ops]
    # several words from the slot before it on
    ops = _read_again(call(_write, word(4), _value), slot=word(5))
    assert _optimized(ops) == [str(op) for op in ops]


def test_words_of_slots_from_different_hashes_are_apart_if_declared() -> None:
    def hashed(name: str) -> tuple[list[models.Op], models.Register]:
        preimage = _bytes(f"{name}_preimage")
        slot = _bytes(f"{name}_slot")
        return [
            assign(preimage, intrinsic(AVMOp.concat, _value, bytes_(name.encode()))),
            assign(slot, intrinsic(AVMOp.keccak256, preimage)),
        ], slot

    def optimized(storage: SlotStorage) -> tuple[list[str], list[models.Op]]:
        balances_ops, balances = hashed("balances")
        allowances_ops, allowances = hashed("allowances")
        ops = [
            *balances_ops,
            *allowances_ops,
            *_read_again(
                call(_write, allowances, word(1)),
                # a constant below 2**64 is not a hash either
                call(_write, word(5), word(1)),
                slot=balances,
            ),
        ]
        return _optimized(ops, storage=storage), ops

    remaining, ops = optimized(STORAGE)
    assert remaining == [str(op) for op in ops]
    remaining, ops = optimized(attrs.evolve(STORAGE, hashed_slots_apart=True))
    assert remaining == [*map(str, ops[:-1]), _FORWARDED]


def test_word_is_forgotten_at_an_operation_that_may_change_a_box() -> None:
    def last(between: models.Op) -> tuple[str, str]:
        ops = _read_again(between)
        return _optimized(ops)[-1], str(ops[-1])

    # what does not change a box
    remaining, _ = last(intrinsic(AVMOp.log, _value))
    assert remaining == _FORWARDED
    remaining, _ = last(intrinsic(AVMOp.app_global_put, _value, _value))
    assert remaining == _FORWARDED
    # what does
    remaining, read = last(intrinsic(AVMOp.box_put, _value, _value))
    assert remaining == read
    remaining, read = last(
        assign(register("deleted", PrimitiveIRType.bool), intrinsic(AVMOp.box_del, _value))
    )
    assert remaining == read
    # an application that is called can share access to the boxes of this one
    remaining, read = last(intrinsic(AVMOp.itxn_submit))
    assert remaining == read


def test_word_is_kept_across_a_call_that_cannot_write_it() -> None:
    writes_five = _sub("writes_five", call(_write, word(5), word(1)))
    writes_box = _sub("writes_box", intrinsic(AVMOp.box_put, bytes_(b"key"), bytes_(b"value")))
    computes = _sub(
        "computes", assign(register("sum"), intrinsic(AVMOp.add, uint64(1), uint64(2)))
    )
    subroutines = (writes_five, writes_box, computes)

    def last(target: models.Subroutine, slot: models.Value) -> tuple[str, str]:
        ops = _read_again(call(target), slot=slot)
        return _optimized(ops, *subroutines)[-1], str(ops[-1])

    # the summary of the subroutine says which word it may write
    remaining, _ = last(writes_five, word(7))
    assert remaining == _FORWARDED
    remaining, read = last(writes_five, word(5))
    assert remaining == read
    # a slot that is not a constant may be that one
    remaining, read = last(writes_five, _slot)
    assert remaining == read
    # a subroutine that changes a box in another way may have changed any word
    remaining, read = last(writes_box, word(7))
    assert remaining == read
    remaining, _ = last(computes, word(5))
    assert remaining == _FORWARDED


def _global_reader() -> models.Subroutine:
    value = register("value")
    return subroutine(
        "global_reader",
        [
            block(
                0,
                [
                    assign(
                        value,
                        intrinsic(
                            AVMOp.app_global_get, bytes_(b"key"), types=[PrimitiveIRType.uint64]
                        ),
                    )
                ],
                return_(value),
            )
        ],
        returns=[PrimitiveIRType.uint64],
    )


def test_repeated_call_takes_the_earlier_result_until_what_it_reads_changes() -> None:
    reader = _global_reader()

    def last(*between: models.Op) -> tuple[str, str]:
        ops = [
            assign(register("first"), call(reader)),
            *between,
            assign(register("again"), call(reader)),
        ]
        return _optimized(ops, reader)[-1], str(ops[-1])

    forwarded = "let again#0: uint64 = first#0"
    remaining, _ = last()
    assert remaining == forwarded
    # a box is not what the subroutine reads
    remaining, _ = last(intrinsic(AVMOp.box_put, _value, _value))
    assert remaining == forwarded
    remaining, repeated = last(intrinsic(AVMOp.app_global_put, bytes_(b"other"), uint64(1)))
    assert remaining == repeated
    # a write through a declared subroutine changes boxes, and so what a box reader gives
    box_reader = read_word("box_reader")
    ops = [
        assign(_bytes("first"), call(box_reader, _slot)),
        call(_write, _other_slot, word(1)),
        assign(_bytes("again"), call(box_reader, _slot)),
    ]
    assert _optimized(ops, box_reader) == [str(op) for op in ops]


def test_program_without_boxes_is_left_as_it_is_unless_storage_is_declared() -> None:
    reader = _global_reader()
    ops = [assign(register("first"), call(reader)), assign(register("again"), call(reader))]
    remaining = optimize_block(
        repeated_state_reads_elimination, ops, reader, parameters=_parameters
    )
    assert remaining == [str(op) for op in ops]
    # a program that reads a box, with nothing declared
    box_reader = read_word("box_reader")
    ops = [
        assign(_bytes("first"), call(box_reader, _slot)),
        assign(_bytes("again"), call(box_reader, _slot)),
    ]
    remaining = optimize_block(
        repeated_state_reads_elimination, ops, box_reader, parameters=_parameters
    )
    assert remaining == [str(ops[0]), _FORWARDED]
