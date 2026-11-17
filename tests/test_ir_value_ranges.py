import math

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.effects import Effect, effects_of
from puya.ir.optimize.value_bounds import ANY_LENGTH, ANY_NUMBER, ANY_UINT64, ValueBounds
from puya.ir.types_ import PrimitiveIRType
from tests.utils.ir import (
    assign,
    biguint,
    block,
    bytes_,
    intrinsic,
    parameter,
    register,
    return_,
    subroutine,
    uint64,
)

_MAX_UINT64 = 2**64 - 1
_WORD = 2**256
_BYTES = PrimitiveIRType.bytes
_BOOL = PrimitiveIRType.bool


def _bytes(name: str) -> models.Register:
    return register(name, _BYTES)


def _condition(name: str) -> models.Register:
    return register(name, _BOOL)


def _may_fail(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    return Effect.fails in effects_of(op, bounds=bounds)


def test_numbers_of_definitions() -> None:
    number = parameter("number")
    data = parameter("data", _BYTES)
    word = _bytes("word")
    converted = _bytes("converted")
    total = _bytes("total")
    wrapped = _bytes("wrapped")
    product = _bytes("product")
    quotient = _bytes("quotient")
    difference = _bytes("difference")
    root = _bytes("root")
    masked = _bytes("masked")
    zeros = _bytes("zeros")
    padded = _bytes("padded")
    inverted = _bytes("inverted")
    chosen = _bytes("chosen")
    ops = [
        assign(word, intrinsic(AVMOp.extract, data, immediates=[0, 32])),
        assign(converted, intrinsic(AVMOp.itob, number)),
        assign(total, intrinsic(AVMOp.add_bytes, word, converted)),
        assign(wrapped, intrinsic(AVMOp.mod_bytes, total, biguint(_WORD))),
        assign(product, intrinsic(AVMOp.mul_bytes, converted, converted)),
        assign(quotient, intrinsic(AVMOp.div_floor_bytes, wrapped, biguint(10))),
        assign(difference, intrinsic(AVMOp.sub_bytes, wrapped, converted)),
        assign(root, intrinsic(AVMOp.bsqrt, wrapped)),
        assign(masked, intrinsic(AVMOp.bitwise_and_bytes, word, bytes_(b"\xff" * 8))),
        assign(zeros, intrinsic(AVMOp.bzero, uint64(32))),
        assign(padded, intrinsic(AVMOp.bitwise_or_bytes, converted, zeros)),
        assign(inverted, intrinsic(AVMOp.bitwise_not_bytes, converted)),
        assign(chosen, intrinsic(AVMOp.select, converted, biguint(_WORD), number, types=[_BYTES])),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))

    # nothing is known about a parameter, and constants are what they are
    assert bounds.number(data) == ANY_NUMBER
    assert bounds.number(biguint(1000)) == (1000, 1000)
    assert bounds.number(bytes_(b"\x00\x00\x01\x00")) == (256, 256)
    # the length bounds the number
    assert bounds.number(word) == (0, _WORD - 1)
    assert bounds.number(converted) == ANY_UINT64
    # byte math
    assert bounds.number(total) == (0, _WORD - 1 + _MAX_UINT64)
    assert bounds.number(wrapped) == (0, _WORD - 1)
    assert bounds.number(product) == (0, _MAX_UINT64**2)
    assert bounds.number(quotient) == (0, (_WORD - 1) // 10)
    assert bounds.number(difference) == (0, _WORD - 1)
    assert bounds.number(root) == (0, math.isqrt(_WORD - 1))
    # and the length of its results follows from their number
    assert bounds.length(total) == (0, 33)
    assert bounds.length(wrapped) == (0, 32)
    assert bounds.length(product) == (0, 16)
    assert bounds.length(root) == (0, 16)
    # ops on the bits of byte arrays, which keep the length of the longer operand
    assert bounds.number(masked) == (0, _MAX_UINT64)
    assert bounds.length(masked) == (32, 32)
    assert bounds.number(zeros) == (0, 0)
    assert bounds.number(padded) == ANY_UINT64
    assert bounds.length(padded) == (32, 32)
    assert bounds.number(inverted) == ANY_UINT64
    assert bounds.number(chosen) == (0, _WORD)


def test_numbers_of_parts() -> None:
    number = parameter("number")
    data = parameter("data", _BYTES)
    converted = _bytes("converted")
    zeros = _bytes("zeros")
    after_zeros = _bytes("after_zeros")
    before_zeros = _bytes("before_zeros")
    last = _bytes("last")
    first = _bytes("first")
    rest = _bytes("rest")
    middle = _bytes("middle")
    last_number = register("last_number")
    first_number = register("first_number")
    last_byte = register("last_byte")
    of_unknown = _bytes("of_unknown")
    ops = [
        assign(converted, intrinsic(AVMOp.itob, number)),
        assign(zeros, intrinsic(AVMOp.bzero, uint64(24))),
        assign(after_zeros, intrinsic(AVMOp.concat, zeros, converted)),
        assign(before_zeros, intrinsic(AVMOp.concat, converted, zeros)),
        assign(last, intrinsic(AVMOp.extract, after_zeros, immediates=[24, 8])),
        assign(first, intrinsic(AVMOp.extract, after_zeros, immediates=[0, 24])),
        assign(rest, intrinsic(AVMOp.extract, after_zeros, immediates=[16, 0])),
        assign(middle, intrinsic(AVMOp.extract3, before_zeros, uint64(4), uint64(8))),
        assign(last_number, intrinsic(AVMOp.extract_uint64, after_zeros, uint64(24))),
        assign(first_number, intrinsic(AVMOp.extract_uint64, after_zeros, uint64(0))),
        assign(last_byte, intrinsic(AVMOp.getbyte, zeros, uint64(23))),
        assign(of_unknown, intrinsic(AVMOp.extract, data, immediates=[24, 8])),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    # zeros before a number do not change it, zeros after it move it up
    assert bounds.number(after_zeros) == ANY_UINT64
    assert bounds.length(after_zeros) == (32, 32)
    assert bounds.number(before_zeros) == (0, _MAX_UINT64 << 192)
    # the last bytes of a byte array whose other bytes are zero are the same number
    assert bounds.number(last) == ANY_UINT64
    assert bounds.number(rest) == ANY_UINT64
    assert bounds.uint64(last_number) == ANY_UINT64
    # and the bytes before them are the number without its last bytes
    assert bounds.number(first) == (0, 0)
    assert bounds.uint64(first_number) == (0, 0)
    assert bounds.uint64(last_byte) == (0, 0)
    # a part that may have bytes that are not zero before it is bounded by its length
    assert bounds.number(middle) == ANY_UINT64
    assert bounds.number(of_unknown) == ANY_UINT64

    # a narrower number in the same bytes
    is_small = _condition("is_small")
    bits = register("bits")
    ops += [
        assign(bits, intrinsic(AVMOp.bitlen, after_zeros)),
        assign(is_small, intrinsic(AVMOp.lte, bits, uint64(8))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    given = bounds.given(non_zero=[is_small])
    assert given.number(after_zeros) == (0, 255)
    assert given.number(last) == (0, 255)
    assert given.uint64(last_number) == (0, 255)
    assert given.number(first) == (0, 0)


def test_number_and_length_are_separate() -> None:
    number = parameter("number")
    data = parameter("data", _BYTES)
    zeros = _bytes("zeros")
    product = _bytes("product")
    small = _bytes("small")
    is_small = _condition("is_small")
    ops = [
        assign(zeros, intrinsic(AVMOp.bzero, uint64(32))),
        assign(product, intrinsic(AVMOp.mul_bytes, zeros, data)),
        assign(small, intrinsic(AVMOp.mod_bytes, data, biguint(256))),
        assign(is_small, intrinsic(AVMOp.lt_bytes, data, biguint(256))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    # 32 zero bytes are the number 0, and too long to be converted to a uint64
    assert bounds.number(zeros) == (0, 0)
    assert bounds.length(zeros) == (32, 32)
    assert _may_fail(intrinsic(AVMOp.btoi, zeros), bounds)
    # the result of byte math has no leading zero bytes, so 0 is the empty byte array
    assert bounds.number(product) == (0, 0)
    assert bounds.length(product) == (0, 0)
    assert not _may_fail(intrinsic(AVMOp.btoi, product), bounds)
    assert bounds.length(small) == (0, 1)
    assert not _may_fail(intrinsic(AVMOp.btoi, small), bounds)
    # a small number in a byte array of unknown length says nothing about the length
    given = bounds.given(non_zero=[is_small])
    assert given.number(data) == (0, 255)
    assert given.length(data) == (0, 64)
    assert _may_fail(intrinsic(AVMOp.btoi, data), given)
    assert given.uint64(register("unknown")) == ANY_UINT64


def test_which_values_have_no_leading_zero_bytes() -> None:
    number = parameter("number")
    data = parameter("data", _BYTES)
    converted = _bytes("converted")
    total = _bytes("total")
    copy = _bytes("copy")
    either = _bytes("either")
    either_or_converted = _bytes("either_or_converted")
    padded = _bytes("padded")
    ops = [
        assign(converted, intrinsic(AVMOp.itob, number)),
        assign(total, intrinsic(AVMOp.add_bytes, data, converted)),
        assign(copy, total),
        assign(either, intrinsic(AVMOp.select, total, biguint(5), number, types=[_BYTES])),
        assign(
            either_or_converted,
            intrinsic(AVMOp.select, total, converted, number, types=[_BYTES]),
        ),
        assign(padded, intrinsic(AVMOp.bitwise_or_bytes, total, bytes_(bytes(32)))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    assert bounds.is_minimal(total)
    assert bounds.is_minimal(copy)
    assert bounds.is_minimal(either)
    assert bounds.is_minimal(biguint(0))
    assert bounds.is_minimal(bytes_(b"\x01\x00"))
    # not known for a parameter, the 8 bytes of a uint64, a padded value or a choice of those
    assert not bounds.is_minimal(data)
    assert not bounds.is_minimal(converted)
    assert not bounds.is_minimal(either_or_converted)
    assert not bounds.is_minimal(padded)
    assert not bounds.is_minimal(bytes_(b"\x00\x01"))


def test_conversions_carry_bounds_both_ways() -> None:
    number = parameter("number")
    data = parameter("data", _BYTES)
    small = register("small")
    converted = _bytes("converted")
    back = register("back")
    read = register("read")
    read_is_small = _condition("read_is_small")
    converted_is_small = _condition("converted_is_small")
    bits = register("bits")
    fits_byte = _condition("fits_byte")
    ops = [
        assign(small, intrinsic(AVMOp.mod, number, uint64(1000))),
        assign(converted, intrinsic(AVMOp.itob, small)),
        assign(back, intrinsic(AVMOp.btoi, converted)),
        assign(read, intrinsic(AVMOp.btoi, data)),
        assign(read_is_small, intrinsic(AVMOp.lt, read, uint64(100))),
        assign(converted_is_small, intrinsic(AVMOp.lt_bytes, converted, biguint(10))),
        assign(bits, intrinsic(AVMOp.bitlen, data)),
        assign(fits_byte, intrinsic(AVMOp.lte, bits, uint64(8))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    # the bytes of a uint64 are the same number, and so is the uint64 of some bytes
    assert bounds.number(converted) == (0, 999)
    assert bounds.uint64(back) == (0, 999)
    assert bounds.uint64(read) == ANY_UINT64
    # a bound on the uint64 that bytes were converted to is a bound on the bytes
    given = bounds.given(non_zero=[read_is_small])
    assert given.number(data) == (0, 99)
    # a bound on the bytes that a uint64 was converted to is a bound on the uint64
    given = bounds.given(non_zero=[converted_is_small])
    assert given.uint64(small) == (0, 9)
    assert given.uint64(back) == (0, 9)
    # a bound on the number of bits is a bound on the number
    given = bounds.given(non_zero=[fits_byte])
    assert given.number(data) == (0, 255)
    assert given.uint64(bits) == (0, 8)
    given = bounds.given(zero=[fits_byte])
    assert given.number(data) == (256, ANY_NUMBER[1])


def test_bounds_given_comparisons_of_numbers() -> None:
    a = parameter("a", _BYTES)
    b = parameter("b", _BYTES)
    data = parameter("data", _BYTES)
    b_word = _bytes("b_word")
    below = _condition("below")
    at_most = _condition("at_most")
    above_ten = _condition("above_ten")
    is_seven = _condition("is_seven")
    same_bytes = _condition("same_bytes")
    not_seven = _condition("not_seven")
    ops = [
        assign(b_word, intrinsic(AVMOp.extract, data, immediates=[0, 32])),
        assign(below, intrinsic(AVMOp.lt_bytes, a, b_word)),
        assign(at_most, intrinsic(AVMOp.lte_bytes, a, b)),
        assign(above_ten, intrinsic(AVMOp.gt_bytes, a, biguint(10))),
        assign(is_seven, intrinsic(AVMOp.eq_bytes, a, biguint(7))),
        assign(same_bytes, intrinsic(AVMOp.eq, a, b_word)),
        assign(not_seven, intrinsic(AVMOp.neq_bytes, a, biguint(7))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[a, b, data]))
    given = bounds.given(non_zero=[below])
    assert given.number(a) == (0, _WORD - 2)
    assert given.number(b_word) == (1, _WORD - 1)
    assert given.not_above(a, b_word)
    assert not given.not_above(b_word, a)
    # the comparison gave a result, so its operands were not too long for byte math
    assert given.length(a) == (0, 64)
    # the other way around where the comparison does not hold
    given = bounds.given(zero=[below])
    assert given.not_above(b_word, a)
    assert given.number(a) == ANY_NUMBER[:1] + (256**64 - 1,)
    # a value that is at most another one of which nothing is known
    given = bounds.given(non_zero=[at_most])
    assert given.not_above(a, b)
    assert not _may_fail(intrinsic(AVMOp.sub_bytes, b, a), given)
    assert _may_fail(intrinsic(AVMOp.sub_bytes, a, b), given)
    given = bounds.given(non_zero=[above_ten])
    assert given.number(a)[0] == 11
    given = bounds.given(zero=[above_ten])
    assert given.number(a) == (0, 10)
    given = bounds.given(non_zero=[is_seven])
    assert given.number(a) == (7, 7)
    assert bounds.given(zero=[not_seven]).number(a) == (7, 7)
    # that two numbers differ says nothing about either
    assert bounds.given(non_zero=[not_seven]).number(a)[0] == 0
    # the same bytes are the same number and have the same length
    given = bounds.given(non_zero=[same_bytes])
    assert given.length(a) == (32, 32)
    assert given.number(a) == (0, _WORD - 1)
    assert bounds.given(zero=[same_bytes]).length(a) == ANY_LENGTH
    # the bounds they were derived from are as they were
    assert bounds.number(a) == ANY_NUMBER


def test_failures_of_byte_math_that_numbers_rule_out() -> None:
    a = parameter("a", _BYTES)
    b = parameter("b", _BYTES)
    number = parameter("number")
    a_word = _bytes("a_word")
    b_word = _bytes("b_word")
    converted = _bytes("converted")
    successor = _bytes("successor")
    b_at_most_a = _condition("b_at_most_a")
    b_not_zero = _condition("b_not_zero")
    ops = [
        assign(a_word, intrinsic(AVMOp.extract, a, immediates=[0, 32])),
        assign(b_word, intrinsic(AVMOp.extract, b, immediates=[0, 32])),
        assign(converted, intrinsic(AVMOp.itob, number)),
        assign(successor, intrinsic(AVMOp.add_bytes, converted, biguint(1))),
        assign(b_at_most_a, intrinsic(AVMOp.lte_bytes, b_word, a_word)),
        assign(b_not_zero, intrinsic(AVMOp.neq_bytes, b_word, biguint(0))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[a, b, number]))
    subtraction = intrinsic(AVMOp.sub_bytes, a_word, b_word)
    division = intrinsic(AVMOp.div_floor_bytes, a_word, b_word)
    modulo = intrinsic(AVMOp.mod_bytes, a_word, b_word)
    assert _may_fail(subtraction, bounds)
    assert _may_fail(division, bounds)
    assert _may_fail(modulo, bounds)
    # the bounds of the operands are enough
    assert not _may_fail(intrinsic(AVMOp.sub_bytes, successor, biguint(1)), bounds)
    assert not _may_fail(intrinsic(AVMOp.sub_bytes, biguint(_WORD), a_word), bounds)
    assert not _may_fail(intrinsic(AVMOp.div_floor_bytes, a_word, successor), bounds)
    assert not _may_fail(intrinsic(AVMOp.mod_bytes, a_word, biguint(_WORD)), bounds)
    # or a condition that says which operand is larger, or that the divisor is not 0
    given = bounds.given(non_zero=[b_at_most_a])
    assert not _may_fail(subtraction, given)
    assert _may_fail(intrinsic(AVMOp.sub_bytes, b_word, a_word), given)
    assert _may_fail(division, given)
    given = bounds.given(zero=[b_not_zero])
    assert _may_fail(division, given)
    # that two numbers differ is not a bound, but an operation that has completed is
    given = bounds.given(completed=[division])
    assert not _may_fail(modulo, given)
    given = bounds.given(completed=[subtraction])
    assert given.not_above(b_word, a_word)
    assert not _may_fail(subtraction, given)
    # an operand that may be too long for byte math fails whatever its number
    assert _may_fail(intrinsic(AVMOp.sub_bytes, a, biguint(0)), bounds)
    assert _may_fail(intrinsic(AVMOp.mod_bytes, a, biguint(_WORD)), bounds)
    assert _may_fail(intrinsic(AVMOp.div_floor_bytes, a_word, bytes_(bytes(65) + b"\x01")), bounds)


def test_outcomes_that_bounds_fix() -> None:
    number = parameter("number")
    data = parameter("data", _BYTES)
    small = register("small")
    word = _bytes("word")
    converted = _bytes("converted")
    total = _bytes("total")
    bits = register("bits")
    word_bits = register("word_bits")
    is_small = _condition("is_small")
    ops = [
        assign(small, intrinsic(AVMOp.mod, number, uint64(100))),
        assign(word, intrinsic(AVMOp.extract, data, immediates=[0, 32])),
        assign(converted, intrinsic(AVMOp.itob, number)),
        assign(total, intrinsic(AVMOp.add_bytes, converted, converted)),
        assign(bits, intrinsic(AVMOp.bitlen, total)),
        assign(word_bits, intrinsic(AVMOp.bitlen, word)),
        assign(is_small, intrinsic(AVMOp.lt, number, uint64(10))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    assert bounds.outcome(intrinsic(AVMOp.lt, small, uint64(100))) is True
    assert bounds.outcome(intrinsic(AVMOp.gte, small, uint64(100))) is False
    assert bounds.outcome(intrinsic(AVMOp.lte, small, uint64(98))) is None
    assert bounds.outcome(intrinsic(AVMOp.eq, small, uint64(100))) is False
    assert bounds.outcome(intrinsic(AVMOp.neq, small, uint64(100))) is True
    assert bounds.outcome(intrinsic(AVMOp.eq, small, uint64(5))) is None
    # the sum of two uint64 values fits in 65 bits, and a word in 256
    assert bounds.uint64(bits) == (0, 65)
    assert bounds.outcome(intrinsic(AVMOp.lte, bits, uint64(256))) is True
    assert bounds.outcome(intrinsic(AVMOp.lte, bits, uint64(64))) is None
    assert bounds.outcome(intrinsic(AVMOp.lte, word_bits, uint64(256))) is True
    assert bounds.outcome(intrinsic(AVMOp.lt_bytes, total, biguint(2**65))) is True
    assert bounds.outcome(intrinsic(AVMOp.gte_bytes, word, biguint(_WORD))) is False
    assert bounds.outcome(intrinsic(AVMOp.lt_bytes, word, biguint(_WORD - 1))) is None
    # byte arrays of other lengths are not the same bytes, though they may be the same number
    assert bounds.outcome(intrinsic(AVMOp.eq, word, converted)) is False
    assert bounds.outcome(intrinsic(AVMOp.neq, word, converted)) is True
    assert bounds.outcome(intrinsic(AVMOp.eq_bytes, word, converted)) is None
    # a value compared with itself, and what conditions say
    assert bounds.outcome(intrinsic(AVMOp.lte, number, number)) is True
    assert bounds.outcome(intrinsic(AVMOp.lt_bytes, word, word)) is False
    given = bounds.given(non_zero=[is_small])
    assert given.outcome(intrinsic(AVMOp.lt, number, uint64(10))) is True
    assert given.outcome(intrinsic(AVMOp.not_, is_small)) is False
    assert given.outcome(intrinsic(AVMOp.lt_bytes, converted, biguint(10))) is True
    assert given.uint64(is_small) == (1, 1)
    # the bounds of a comparison are its outcome where that is fixed
    always = _condition("always")
    ops.append(assign(always, intrinsic(AVMOp.lte, bits, uint64(256))))
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[number, data]))
    assert bounds.uint64(always) == (1, 1)
    assert bounds.uint64(is_small) == (0, 1)
