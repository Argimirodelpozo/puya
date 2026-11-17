"""Subroutines that stand in for the ones a frontend declares as slot storage, for tests of
the optimizations that follow the words a program keeps in boxes."""

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.types_ import PrimitiveIRType
from puya.options import SlotStorage
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    intrinsic,
    parameter,
    register,
    return_,
    subroutine,
    uint64,
)

STORAGE = SlotStorage(
    read_page_word="read_page_word",
    write_page_word="write_page_word",
    read_word="read_word",
    write_words="write_words",
    slot_add="slot_add",
)

PAGE = bytes_(bytes(32))
"""the page whose first word is the word of slot 0"""


def word(number: int) -> models.BytesConstant:
    """32 bytes that hold a number: a slot, or a word to store."""
    return bytes_(number.to_bytes(32, "big"))


def read_page_word() -> models.Subroutine:
    page = parameter("page", PrimitiveIRType.bytes)
    at = parameter("at")
    result = register("result", PrimitiveIRType.bytes)
    return subroutine(
        "read_page_word",
        [
            block(
                0,
                [assign(result, intrinsic(AVMOp.box_extract, page, at, uint64(32)))],
                return_(result),
            )
        ],
        parameters=[page, at],
        returns=[PrimitiveIRType.bytes],
    )


def write_page_word() -> models.Subroutine:
    page = parameter("page", PrimitiveIRType.bytes)
    at = parameter("at")
    value = parameter("value", PrimitiveIRType.bytes)
    return subroutine(
        "write_page_word",
        [
            block(
                0,
                [
                    assign(
                        register("created", PrimitiveIRType.bool),
                        intrinsic(AVMOp.box_create, page, uint64(2048)),
                    ),
                    intrinsic(AVMOp.box_replace, page, at, value),
                ],
                return_(),
            )
        ],
        parameters=[page, at, value],
    )


def read_word(name: str = "read_word") -> models.Subroutine:
    slot = parameter("slot", PrimitiveIRType.bytes)
    result = register("result", PrimitiveIRType.bytes)
    exists = register("exists", PrimitiveIRType.bool)
    return subroutine(
        name,
        [
            block(
                0,
                [
                    models.Assignment(
                        targets=[result, exists],
                        source=intrinsic(AVMOp.box_get, slot),
                        source_location=None,
                    )
                ],
                return_(result),
            )
        ],
        parameters=[slot],
        returns=[PrimitiveIRType.bytes],
    )


def write_words() -> models.Subroutine:
    slot = parameter("slot", PrimitiveIRType.bytes)
    words = parameter("words", PrimitiveIRType.bytes)
    return subroutine(
        "write_words",
        [block(0, [intrinsic(AVMOp.box_put, slot, words)], return_())],
        parameters=[slot, words],
    )


def slot_add() -> models.Subroutine:
    slot = parameter("slot", PrimitiveIRType.bytes)
    distance = parameter("distance")
    as_bytes = register("as_bytes", PrimitiveIRType.bytes)
    result = register("result", PrimitiveIRType.bytes)
    return subroutine(
        "slot_add",
        [
            block(
                0,
                [
                    assign(as_bytes, intrinsic(AVMOp.itob, distance)),
                    assign(result, intrinsic(AVMOp.add_bytes, slot, as_bytes)),
                ],
                return_(result),
            )
        ],
        parameters=[slot, distance],
        returns=[PrimitiveIRType.bytes],
    )
