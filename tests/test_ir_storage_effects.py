import attrs

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.effects import (
    ANY_BOX_CHANGE,
    ANY_BOX_READ,
    NO_BOX_CHANGE,
    NO_BOX_READ,
    BoxChanges,
    BoxReads,
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

    def writes(*words: SlotWrite, other: bool = False) -> BoxChanges:
        # a write that is not a word write of a page may delete a box or make one shorter
        return BoxChanges(words=frozenset(words), other=other, shrinks=True)

    assert summaries.box_changes(constant_writer) == writes(slot_five)
    assert summaries.box_changes(unknown_writer) == writes(ANY_SLOT)
    assert summaries.box_changes(hashed_writer) == writes(SlotWrite(ending=b"balances"))
    assert summaries.box_changes(box_writer) == ANY_BOX_CHANGE
    assert summaries.box_changes(reader) == NO_BOX_CHANGE
    assert summaries.box_changes(indirect_writer) == writes(slot_five)
    assert summaries.box_changes(both) == writes(slot_five, other=True)
    assert summaries.box_changes(failing_writer) == NO_BOX_CHANGE
    # what each of them may read of boxes
    assert summaries.box_reads(reader) == BoxReads(words=frozenset({slot_five}))
    assert summaries.box_reads(indirect_writer) == BoxReads(words=frozenset({slot_five}))
    # a write through a declared subroutine reads no word
    assert summaries.box_reads(constant_writer) == NO_BOX_READ
    # an op that changes a box fails or not by what the box is
    assert summaries.box_reads(box_writer) == ANY_BOX_READ
    assert summaries.box_reads(both) == ANY_BOX_READ
    # without a declaration, a subroutine that may change a box may change any
    summaries = analyse_subroutine_effects(program(_sub("main"), *_declared, constant_writer))
    assert summaries.storage is None
    assert summaries.box_changes(constant_writer) == ANY_BOX_CHANGE
    assert summaries.box_changes(_read) == NO_BOX_CHANGE
    assert summaries.box_reads(constant_writer) == ANY_BOX_READ


def test_summary_says_which_words_a_subroutine_may_read() -> None:
    slot = parameter("slot", PrimitiveIRType.bytes)
    reads_any = subroutine(
        "reads_any",
        [block(0, [assign(_bytes("read"), call(_read, slot))], return_())],
        parameters=[slot],
    )
    reads_page = _sub("reads_page", assign(_bytes("read"), call(_read_page, PAGE, uint64(64))))
    writes_page = _sub("writes_page", call(_write_page, PAGE, uint64(64), word(1)))
    reads_box = _sub("reads_box", _box_get(bytes_(b"key")))
    computes = _sub(
        "computes", assign(register("sum"), intrinsic(AVMOp.add, uint64(1), uint64(2)))
    )
    summaries = analyse_subroutine_effects(
        program(_sub("main"), *_declared, reads_any, reads_page, writes_page, reads_box, computes),
        STORAGE,
    )
    assert summaries.box_reads(reads_any) == BoxReads(words=frozenset({ANY_SLOT}))
    assert summaries.box_reads(reads_page) == BoxReads(words=frozenset({SlotWrite(constant=2)}))
    assert summaries.box_reads(reads_box) == ANY_BOX_READ
    assert summaries.box_reads(computes) == NO_BOX_READ
    # a word write of a page makes no box shorter
    assert summaries.box_changes(writes_page) == BoxChanges(
        words=frozenset({SlotWrite(constant=2)})
    )
    assert summaries.box_reads(writes_page) == NO_BOX_READ
