import attrs

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.state_reads import (
    repeated_state_reads_elimination,
    replaced_word_writes_elimination,
)
from puya.ir.types_ import PrimitiveIRType
from puya.options import SlotStorage
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    call,
    goto,
    intrinsic,
    optimize_block,
    optimize_subroutine,
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

_PAGES = attrs.evolve(STORAGE, pages_are_boxes=True)
"""what a word write of a page does to its box is declared"""
_PROMISED = attrs.evolve(STORAGE, overwritten_writes_removable=True)
"""the program does not depend on a write that is replaced before anything can tell"""

_slot = parameter("slot", PrimitiveIRType.bytes)
_other_slot = parameter("other_slot", PrimitiveIRType.bytes)
_value = parameter("value", PrimitiveIRType.bytes)
_other_value = parameter("other_value", PrimitiveIRType.bytes)
_flag = parameter("flag")
_parameters = (_slot, _other_slot, _value, _other_value, _flag)

_AT = uint64(32)
"""where the word of slot 1 is in its page"""


def _bytes(name: str) -> models.Register:
    return register(name, PrimitiveIRType.bytes)


def _sub(name: str, *ops: models.Op) -> models.Subroutine:
    return subroutine(name, [block(0, ops, return_())])


def _fail() -> models.Fail:
    return models.Fail(error_message=None, explicit=True, source_location=None)


def _optimized(
    ops: list[models.Op], *subroutines: models.Subroutine, storage: SlotStorage
) -> list[str]:
    return optimize_block(
        replaced_word_writes_elimination,
        ops,
        *_declared,
        *subroutines,
        parameters=_parameters,
        storage=storage,
    )


def _page_writes(*between: models.Op) -> list[models.Op]:
    """A write of zero to a page word, the given ops, and a write of that word again: the
    first write changes nothing where the box does not hold the word, so it can go unless
    one of the ops in between can tell."""
    return [
        call(_write_page, PAGE, _AT, word(0)),
        *between,
        call(_write_page, PAGE, _AT, _value),
    ]


def test_page_word_write_is_removed_if_box_is_known_to_hold_the_word() -> None:
    ops = [
        # a word that is not zero is stored, so the box reaches past the word of slot 1
        call(_write_page, PAGE, uint64(64), word(7)),
        call(_write_page, PAGE, _AT, _value),
        call(_write_page, PAGE, _AT, _other_value),
    ]
    assert _optimized(ops, storage=_PAGES) == [str(ops[0]), str(ops[2])]


def test_page_word_write_of_zero_is_removed() -> None:
    ops = _page_writes()
    assert _optimized(ops, storage=_PAGES) == [str(ops[1])]


def test_page_word_write_is_removed_if_later_write_is_not_zero() -> None:
    ops = [
        call(_write_page, PAGE, _AT, _value),
        call(_write_page, PAGE, _AT, word(7)),
    ]
    assert _optimized(ops, storage=_PAGES) == [str(ops[1])]


def test_page_word_write_is_kept_if_its_allocation_could_be_lost() -> None:
    # The first write makes the box long enough to hold the word unless its value is zero,
    # and the second one does not if its value is zero: with the first one gone, the box
    # could stay shorter.
    ops = [
        call(_write_page, PAGE, _AT, _value),
        call(_write_page, PAGE, _AT, _other_value),
    ]
    assert _optimized(ops, storage=_PAGES) == [str(op) for op in ops]
    # the declaration can promise that the program does not depend on that
    promised = attrs.evolve(_PAGES, overwritten_writes_removable=True)
    assert _optimized(ops, storage=promised) == [str(ops[1])]
    # The promise is not enough where the bytes of a word are replaced in the box further on.
    # That can be there in place of a word write, because the box was known to hold the
    # word, and the first write can be what made the box that long.
    ops = [*ops, intrinsic(AVMOp.box_replace, PAGE, uint64(0), word(1))]
    assert _optimized(ops, storage=promised) == [str(op) for op in ops]


def test_page_word_write_is_kept_if_later_write_is_through_another_subroutine() -> None:
    # the same slot, but nothing says that the two fail alike or leave the same box
    ops = [
        call(_write_page, PAGE, _AT, word(0)),
        call(_write, word(1), word(7)),
    ]
    assert _optimized(ops, storage=_PAGES) == [str(op) for op in ops]


def test_page_word_write_is_kept_if_a_box_may_get_shorter_before_the_later_write() -> None:
    # A write to another kind of slot may delete its box. The first write could be what
    # takes the transaction over the bytes it may write, and with the box gone the later
    # write could stay within them.
    ops = _page_writes(call(_write, _slot, word(0)))
    assert _optimized(ops, storage=_PAGES) == [str(op) for op in ops]
    # a write to another word of a page makes no box shorter
    ops = _page_writes(call(_write_page, PAGE, uint64(64), _other_value))
    assert _optimized(ops, storage=_PAGES) == [str(ops[1]), str(ops[2])]


def test_page_word_write_is_kept_without_what_a_write_does_to_its_box() -> None:
    ops = _page_writes()
    assert _optimized(ops, storage=STORAGE) == [str(op) for op in ops]


def test_write_is_kept_if_the_word_is_read_before_the_later_write() -> None:
    def remaining(read: models.ValueProvider) -> tuple[list[str], list[models.Op]]:
        ops = _page_writes(assign(_bytes("read"), read))
        return _optimized(ops, storage=_PAGES), ops

    # the word itself
    left, ops = remaining(call(_read_page, PAGE, _AT))
    assert left == [str(op) for op in ops]
    # a word that may be the same one
    left, ops = remaining(call(_read, _slot))
    assert left == [str(op) for op in ops]
    # a word that is certainly another one
    left, ops = remaining(call(_read_page, PAGE, uint64(64)))
    assert left == [str(ops[1]), str(ops[2])]


def test_write_is_kept_if_an_operation_before_the_later_write_can_tell() -> None:
    address = register("address", PrimitiveIRType.account)
    read_address = assign(
        address, intrinsic(AVMOp.global_, immediates=["CurrentApplicationAddress"])
    )

    def is_kept(*between: models.Op) -> bool:
        ops = [read_address, *_page_writes(*between)]
        left = _optimized(ops, storage=_PAGES)
        if left == [str(op) for op in ops]:
            return True
        assert left == [str(op) for op in [ops[0], *ops[2:]]]
        return False

    # the size of a box, or its contents
    assert is_kept(
        models.Assignment(
            targets=[register("length"), register("exists", PrimitiveIRType.bool)],
            source=intrinsic(AVMOp.box_len, PAGE),
            source_location=None,
        )
    )
    assert is_kept(intrinsic(AVMOp.box_replace, bytes_(b"other"), uint64(0), _other_value))
    # the minimum balance follows the sizes of the boxes
    assert is_kept(assign(register("minimum"), intrinsic(AVMOp.min_balance, address)))
    # an application that is called can read the boxes of this one
    assert is_kept(intrinsic(AVMOp.itxn_begin), intrinsic(AVMOp.itxn_submit))
    # none of these depends on a box
    assert not is_kept(
        intrinsic(AVMOp.log, _other_value),
        intrinsic(AVMOp.app_global_put, bytes_(b"key"), _other_value),
        models.Assert(condition=_flag, message=None, explicit=True, source_location=None),
        assign(register("sum"), intrinsic(AVMOp.add, _flag, uint64(1))),
    )


def test_write_is_kept_if_a_call_before_the_later_write_can_tell() -> None:
    box_reader = read_word("box_reader")
    indirect_reader = _sub("indirect_reader", assign(_bytes("read"), call(box_reader, word(9))))
    exiting = subroutine(
        "exiting", [block(0, [], models.ProgramExit(result=uint64(1), source_location=None))]
    )
    logs = _sub("logs", intrinsic(AVMOp.log, bytes_(b"logged")))
    subroutines = (box_reader, indirect_reader, exiting, logs)

    def remaining(target: models.Subroutine) -> tuple[list[str], list[models.Op]]:
        ops = _page_writes(call(target))
        return _optimized(ops, *subroutines, storage=_PAGES), ops

    # a subroutine that reads a box, itself or through another
    left, ops = remaining(indirect_reader)
    assert left == [str(op) for op in ops]
    # a subroutine that may end the program, with the first write in place
    left, ops = remaining(exiting)
    assert left == [str(op) for op in ops]
    left, ops = remaining(logs)
    assert left == [str(ops[1]), str(ops[2])]


def _branches(
    then_ops: list[models.Op], else_ops: list[models.Op], else_terminator: models.ControlOp
) -> tuple[models.Subroutine, models.Op]:
    """A write of zero to a page word, and then one of two ways on."""
    first = call(_write_page, PAGE, _AT, word(0))
    entry = block(0, [first], return_())
    then_block = block(1, then_ops, return_())
    else_block = block(2, else_ops, else_terminator)
    entry.terminator = models.ConditionalBranch(
        condition=_flag, non_zero=then_block, zero=else_block, source_location=None
    )
    then_block.add_predecessor(entry)
    else_block.add_predecessor(entry)
    main = subroutine("main", [entry, then_block, else_block], parameters=_parameters)
    return main, first


def test_write_is_kept_unless_it_is_replaced_on_every_way_on() -> None:
    def is_kept(main: models.Subroutine, first: models.Op) -> bool:
        optimize_subroutine(replaced_word_writes_elimination, main, *_declared, storage=_PAGES)
        return first in main.body[0].ops

    # one of the ways ends the subroutine with the first write in place
    assert is_kept(*_branches([call(_write_page, PAGE, _AT, _value)], [], return_()))
    # both ways write the word again
    assert not is_kept(
        *_branches(
            [call(_write_page, PAGE, _AT, _value)],
            [call(_write_page, PAGE, _AT, _other_value)],
            return_(),
        )
    )
    # a program that fails leaves nothing of what it wrote
    assert not is_kept(*_branches([call(_write_page, PAGE, _AT, _value)], [], _fail()))


def test_write_is_kept_without_promise_if_nothing_is_known_of_its_box() -> None:
    ops = [
        call(_write, _slot, word(1)),
        call(_write, _slot, word(2)),
    ]
    assert _optimized(ops, storage=STORAGE) == [str(op) for op in ops]
    assert _optimized(ops, storage=_PAGES) == [str(op) for op in ops]
    assert _optimized(ops, storage=_PROMISED) == [str(ops[1])]


def test_promised_write_is_kept_if_something_can_tell_or_if_it_is_not_replaced() -> None:
    # a read of a word that may be the same one
    ops = [
        call(_write, _slot, word(1)),
        assign(_bytes("read"), call(_read, _other_slot)),
        call(_write, _slot, word(2)),
    ]
    assert _optimized(ops, storage=_PROMISED) == [str(op) for op in ops]
    # a write to a slot that may be another one
    ops = [
        call(_write, _slot, word(1)),
        call(_write, _other_slot, word(2)),
    ]
    assert _optimized(ops, storage=_PROMISED) == [str(op) for op in ops]
    # a write of several words replaces none for certain
    ops = [
        call(_write, _slot, word(1)),
        call(_write, _slot, _value),
    ]
    assert _optimized(ops, storage=_PROMISED) == [str(op) for op in ops]
    # a write to another slot in between does not read the word
    ops = [
        call(_write, _slot, word(1)),
        call(_write, _other_slot, word(2)),
        call(_write, _slot, word(3)),
    ]
    assert _optimized(ops, storage=_PROMISED) == [str(ops[1]), str(ops[2])]


def _loop(slot_ops: list[models.Op], slot: models.Value) -> tuple[models.Subroutine, models.Op]:
    """A loop that writes a word each time around, and a write of the same register after
    the loop."""
    count = register("count")
    start = register("start")
    following = register("following")
    in_loop = call(_write, slot, word(1))
    entry = block(0, [assign(start, uint64(0))], return_())
    loop = block(
        1,
        [*slot_ops, in_loop, assign(following, intrinsic(AVMOp.add, count, uint64(1)))],
        return_(),
    )
    after = block(2, [call(_write, slot, word(2))], return_())
    goto(entry, loop)
    loop.terminator = models.ConditionalBranch(
        condition=_flag, non_zero=loop, zero=after, source_location=None
    )
    loop.add_predecessor(loop)
    after.add_predecessor(loop)
    loop.phis.append(
        models.Phi(
            register=count,
            args=[
                models.PhiArgument(value=start, through=entry),
                models.PhiArgument(value=following, through=loop),
            ],
        )
    )
    main = subroutine("main", [entry, loop, after], parameters=_parameters)
    return main, in_loop


def test_write_in_a_loop_is_replaced_by_the_next_time_around_only_at_the_same_slot() -> None:
    # the slot is the same each time around: every write is replaced by the next one, and
    # the last one by the write after the loop
    main, in_loop = _loop([], _slot)
    optimize_subroutine(replaced_word_writes_elimination, main, *_declared, storage=_PROMISED)
    assert in_loop not in main.body[1].ops
    # the slot is named by a register that is assigned anew each time around: the next
    # time around writes another word
    count = register("count")
    near = _bytes("near")
    main, in_loop = _loop([assign(near, call(_add, _slot, count))], near)
    optimize_subroutine(replaced_word_writes_elimination, main, *_declared, storage=_PROMISED)
    assert in_loop in main.body[1].ops


def test_chain_of_read_change_write_is_left_with_one_read_and_one_write() -> None:
    def changed(name: str, read: models.Register, position: int) -> models.Assignment:
        return assign(
            _bytes(name),
            intrinsic(AVMOp.replace2, read, bytes_(b"\x01"), immediates=[position]),
        )

    read_1, read_2, read_3 = _bytes("read_1"), _bytes("read_2"), _bytes("read_3")
    change_1 = changed("changed_1", read_1, 31)
    change_2 = changed("changed_2", read_2, 30)
    change_3 = changed("changed_3", read_3, 29)
    ops = [
        assign(read_1, call(_read, _slot)),
        change_1,
        call(_write, _slot, change_1.targets[0]),
        assign(read_2, call(_read, _slot)),
        change_2,
        call(_write, _slot, change_2.targets[0]),
        assign(read_3, call(_read, _slot)),
        change_3,
        call(_write, _slot, change_3.targets[0]),
        assign(_bytes("read_4"), call(_read, _slot)),
    ]
    main = subroutine("main", [block(0, ops, return_())], parameters=_parameters)
    for optimization in (repeated_state_reads_elimination, replaced_word_writes_elimination):
        optimize_subroutine(optimization, main, *_declared, storage=_PROMISED)
    assert [str(op) for op in main.body[0].ops] == [
        str(ops[0]),
        str(change_1),
        "let read_2#0: bytes = changed_1#0",
        str(change_2),
        "let read_3#0: bytes = changed_2#0",
        str(change_3),
        str(ops[8]),
        "let read_4#0: bytes = changed_3#0",
    ]
