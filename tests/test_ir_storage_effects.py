import attrs

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.effects import (
    ANY_BOX_CHANGE,
    NO_BOX_CHANGE,
    BoxChanges,
    analyse_subroutine_effects,
)
from puya.ir.optimize.slot_storage import ANY_SLOT, SlotWrite
from puya.ir.types_ import PrimitiveIRType
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    call,
    intrinsic,
    parameter,
    program,
    register,
    return_,
    subroutine,
    uint64,
)
from tests.utils.slot_storage import (
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


def _bytes(name: str) -> models.Register:
    return register(name, PrimitiveIRType.bytes)


def _sub(name: str, *ops: models.Op) -> models.Subroutine:
    return subroutine(name, [block(0, ops, return_())])


def _box_get(slot: models.Value) -> models.Assignment:
    return models.Assignment(
        targets=[_bytes("result"), register("exists", PrimitiveIRType.bool)],
        source=intrinsic(AVMOp.box_get, slot),
        source_location=None,
    )


def test_subroutine_has_its_declared_role_if_it_does_no_more() -> None:
    summaries = analyse_subroutine_effects(
        program(_sub("main"), *_declared, read_word("read_word.copy")), STORAGE
    )
    assert summaries.storage is not None
    assert summaries.storage.roles == {
        "read_page_word": "read_page_word",
        "write_page_word": "write_page_word",
        "read_word": "read_word",
        # a program can hold several copies of a subroutine, with qualified ids
        "read_word.copy": "read_word",
        "write_words": "write_words",
        "slot_add": "slot_add",
    }


def test_subroutine_that_does_more_than_its_declared_role_has_none() -> None:
    slot = parameter("slot", PrimitiveIRType.bytes)
    distance = parameter("distance")
    read = _box_get(slot)
    # declared to read a word, and it also logs
    logging_reader = subroutine(
        "read_word",
        [block(0, [read, intrinsic(AVMOp.log, read.targets[0])], return_(read.targets[0]))],
        parameters=[slot],
        returns=[PrimitiveIRType.bytes],
    )
    # declared to compute a slot, and it reads a box
    read = _box_get(slot)
    reading_adder = subroutine(
        "slot_add",
        [block(0, [read], return_(read.targets[0]))],
        parameters=[slot, distance],
        returns=[PrimitiveIRType.bytes],
    )
    summaries = analyse_subroutine_effects(
        program(_sub("main"), logging_reader, _write, reading_adder), STORAGE
    )
    assert summaries.storage is not None
    assert summaries.storage.roles == {"write_words": "write_words"}


def test_pages_are_boxes_only_if_page_subroutines_use_boxes_of_the_program() -> None:
    storage = attrs.evolve(STORAGE, pages_are_boxes=True)
    summaries = analyse_subroutine_effects(program(_sub("main"), _read_page, _write_page), storage)
    assert summaries.storage is not None
    assert summaries.storage.pages_are_boxes
    # a page reader that looks at a box of an application given by id
    page = parameter("page", PrimitiveIRType.bytes)
    at = parameter("at")
    result = _bytes("result")
    other_app_reader = subroutine(
        "read_page_word",
        [
            block(
                0,
                [
                    models.Assignment(
                        targets=[register("length"), register("exists", PrimitiveIRType.bool)],
                        source=intrinsic(AVMOp.app_box_len, uint64(7), page),
                        source_location=None,
                    ),
                    assign(result, intrinsic(AVMOp.box_extract, page, at, uint64(32))),
                ],
                return_(result),
            )
        ],
        parameters=[page, at],
        returns=[PrimitiveIRType.bytes],
    )
    summaries = analyse_subroutine_effects(
        program(_sub("main"), other_app_reader, _write_page), storage
    )
    assert summaries.storage is not None
    assert summaries.storage.roles["read_page_word"] == "read_page_word"
    assert not summaries.storage.pages_are_boxes


def test_summary_says_which_words_a_subroutine_may_write() -> None:
    constant_writer = _sub("constant_writer", call(_write, word(5), word(1)))
    slot = parameter("slot", PrimitiveIRType.bytes)
    unknown_writer = subroutine(
        "unknown_writer",
        [block(0, [call(_write, slot, word(1))], return_())],
        parameters=[slot],
    )
    # writes the word at the hash of something that ends in "balances"
    preimage = _bytes("preimage")
    hashed = _bytes("hashed")
    hashed_writer = subroutine(
        "hashed_writer",
        [
            block(
                0,
                [
                    assign(preimage, intrinsic(AVMOp.concat, slot, bytes_(b"balances"))),
                    assign(hashed, intrinsic(AVMOp.keccak256, preimage)),
                    call(_write, hashed, word(1)),
                ],
                return_(),
            )
        ],
        parameters=[slot],
    )
    box_writer = _sub("box_writer", intrinsic(AVMOp.box_put, bytes_(b"key"), bytes_(b"value")))
    reader = _sub("reader", assign(_bytes("read"), call(_read, word(5))))
    indirect_writer = _sub("indirect_writer", call(constant_writer), call(reader))
    both = _sub("both", call(constant_writer), call(box_writer))
    # writes a word and then fails, which undoes the write
    failing_writer = subroutine(
        "failing_writer",
        [
            block(
                0,
                [call(_write, word(5), word(1))],
                models.Fail(error_message=None, explicit=True, source_location=None),
            )
        ],
    )
    summaries = analyse_subroutine_effects(
        program(
            _sub("main"),
            *_declared,
            constant_writer,
            unknown_writer,
            hashed_writer,
            box_writer,
            reader,
            indirect_writer,
            both,
            failing_writer,
        ),
        STORAGE,
    )
    slot_five = SlotWrite(constant=5)
    assert summaries.box_changes(constant_writer) == BoxChanges(words=frozenset({slot_five}))
    assert summaries.box_changes(unknown_writer) == BoxChanges(words=frozenset({ANY_SLOT}))
    assert summaries.box_changes(hashed_writer) == BoxChanges(
        words=frozenset({SlotWrite(ending=b"balances")})
    )
    assert summaries.box_changes(box_writer) == ANY_BOX_CHANGE
    assert summaries.box_changes(reader) == NO_BOX_CHANGE
    assert summaries.box_changes(indirect_writer) == BoxChanges(words=frozenset({slot_five}))
    assert summaries.box_changes(both) == BoxChanges(words=frozenset({slot_five}), other=True)
    assert summaries.box_changes(failing_writer) == NO_BOX_CHANGE
    # without a declaration, a subroutine that may change a box may change any
    summaries = analyse_subroutine_effects(program(_sub("main"), *_declared, constant_writer))
    assert summaries.storage is None
    assert summaries.box_changes(constant_writer) == ANY_BOX_CHANGE
    assert summaries.box_changes(_read) == NO_BOX_CHANGE
