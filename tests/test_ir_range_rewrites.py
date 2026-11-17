import copy
from collections.abc import Callable, Sequence
from unittest.mock import Mock

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.main import SubroutineOptimization
from puya.ir.optimize.range_rewrites import (
    range_cheaper_forms,
    range_constant_outcomes,
    range_operand_results,
    range_rewrites,
)
from puya.ir.types_ import PrimitiveIRType
from tests.utils.ir import (
    assert_,
    assign,
    biguint,
    block,
    branch,
    bytes_,
    goto,
    intrinsic,
    optimize_block,
    parameter,
    phi,
    register,
    return_,
    subroutine,
    uint64,
)

_WORD = 2**256
_BYTES = PrimitiveIRType.bytes
_BOOL = PrimitiveIRType.bool

_a = parameter("a")
_b = parameter("b")
_data = parameter("data", _BYTES)
_other = parameter("other", _BYTES)
_PARAMETERS = [_a, _b, _data, _other]


def _bytes(name: str) -> models.Register:
    return register(name, _BYTES)


def _condition(name: str) -> models.Register:
    return register(name, _BOOL)


def _use(*values: models.Value) -> list[models.Op]:
    """Ops that read values as they are, bytes and all."""
    return [intrinsic(AVMOp.app_global_put, bytes_(b"key"), value) for value in values]


def _constant_outcomes(ops: Sequence[models.Op]) -> list[str]:
    return _rewritten(range_constant_outcomes, ops)


def _operand_results(ops: Sequence[models.Op]) -> list[str]:
    return _rewritten(range_operand_results, ops)


def _cheaper_forms(ops: Sequence[models.Op]) -> list[str]:
    return _rewritten(range_cheaper_forms, ops)


def _rewritten(
    rewrite: Callable[[IROptimizationContext, models.Subroutine], bool], ops: Sequence[models.Op]
) -> list[str]:
    # on a copy: a rewrite changes the ops it is given, which are compared with the result
    return optimize_block(rewrite, copy.deepcopy(list(ops)), parameters=_PARAMETERS)


def _unchanged(ops: Sequence[models.Op]) -> list[str]:
    return [str(op) for op in ops]


_a_word = _bytes("a_word")
_b_word = _bytes("b_word")
_a_bytes = _bytes("a_bytes")
_b_bytes = _bytes("b_bytes")
# two words of which nothing else is known, and the 8 bytes of two uint64 values
_WORDS = [
    assign(_a_word, intrinsic(AVMOp.extract, _data, immediates=[0, 32])),
    assign(_b_word, intrinsic(AVMOp.extract, _other, immediates=[0, 32])),
]
_CONVERTED = [
    assign(_a_bytes, intrinsic(AVMOp.itob, _a)),
    assign(_b_bytes, intrinsic(AVMOp.itob, _b)),
]


def test_comparison_that_ranges_decide() -> None:
    small = register("small")
    is_small = _condition("is_small")
    is_tiny = _condition("is_tiny")
    ops = [
        assign(small, intrinsic(AVMOp.mod, _a, uint64(100))),
        assign(is_small, intrinsic(AVMOp.lt, small, uint64(100))),
        assert_(is_small),
        assign(is_tiny, intrinsic(AVMOp.lt, small, uint64(10))),
        assert_(is_tiny),
        *_use(small),
    ]
    # the comparison that always holds is that result, and the assert of it is gone. The
    # modulo that makes it hold stays, and so does the comparison that may not hold
    assert _constant_outcomes(ops) == [
        "let small#0: uint64 = (% a#0 100u)",
        "let is_small#0: bool = 1u",
        "let is_tiny#0: bool = (< small#0 10u)",
        "(assert is_tiny#0)",
        "(app_global_put 0x6b6579 small#0)",
    ]


def test_check_of_addition_that_cannot_overflow() -> None:
    total = _bytes("total")
    bits = register("bits")
    fits = _condition("fits")

    def checked_addition(x: models.Value, y: models.Value, limit: int) -> list[models.Op]:
        return [
            *_WORDS,
            *_CONVERTED,
            assign(total, intrinsic(AVMOp.add_bytes, x, y)),
            assign(bits, intrinsic(AVMOp.bitlen, total)),
            assign(fits, intrinsic(AVMOp.lte, bits, uint64(limit))),
            assert_(fits),
            *_use(total),
        ]

    # the sum of two uint64 values fits in 256 bits
    assert _constant_outcomes(checked_addition(_a_bytes, _b_bytes, 256))[-3:] == [
        "let bits#0: uint64 = (bitlen total#0)",
        "let fits#0: bool = 1u",
        "(app_global_put 0x6b6579 total#0)",
    ]
    # but not in 64 bits, and the sum of two words may not fit in 256 bits
    for ops in (
        checked_addition(_a_bytes, _b_bytes, 64),
        checked_addition(_a_word, _b_word, 256),
    ):
        assert _constant_outcomes(ops) == _unchanged(ops)


def test_comparison_that_can_fail_stays() -> None:
    zeros = _bytes("zeros")
    is_small = _condition("is_small")
    ops = [
        assign(zeros, intrinsic(AVMOp.bzero, uint64(100))),
        assign(is_small, intrinsic(AVMOp.lt_bytes, zeros, biguint(5))),
        *_use(is_small),
    ]
    # 100 zero bytes are the number 0, which is below 5, but byte math does not take them
    assert _constant_outcomes(ops) == _unchanged(ops)
    # 64 zero bytes it does take
    ops[0] = assign(zeros, intrinsic(AVMOp.bzero, uint64(64)))
    assert _constant_outcomes(ops)[1] == "let is_small#0: bool = 1u"


def _diamond(
    entry_ops: Sequence[models.Op],
    condition: models.Value,
    then_ops: Sequence[models.Op],
    otherwise_ops: Sequence[models.Op],
    merge_ops: Sequence[models.Op],
) -> models.Subroutine:
    entry = block(0, entry_ops, return_())
    then = block(1, then_ops, return_())
    otherwise = block(2, otherwise_ops, return_())
    merge = block(3, merge_ops, return_())
    branch(entry, condition, non_zero=then, zero=otherwise)
    goto(then, merge)
    goto(otherwise, merge)
    return subroutine("main", [entry, then, otherwise, merge], parameters=_PARAMETERS)


def _block_ops(sub: models.Subroutine, index: int) -> list[str]:
    return [str(op) for op in sub.body[index].ops]


def test_outcomes_that_a_branch_decides() -> None:
    a_below_b = _condition("a_below_b")
    again = _condition("again")
    b_at_most_a = _condition("b_at_most_a")
    at_merge = _condition("at_merge")
    difference = register("difference")
    sub = _diamond(
        [assign(a_below_b, intrinsic(AVMOp.lt, _a, _b))],
        a_below_b,
        [assert_(a_below_b), assign(again, intrinsic(AVMOp.lte, _a, _b)), *_use(again)],
        [assign(b_at_most_a, intrinsic(AVMOp.lte, _b, _a)), *_use(b_at_most_a)],
        [assign(at_merge, intrinsic(AVMOp.lte, _a, _b)), *_use(at_merge)],
    )
    assert range_constant_outcomes(Mock(), sub)
    sub.validate_with_ssa()
    # where the branch was taken its condition holds, and what follows from it
    assert _block_ops(sub, 1) == [
        "let again#0: bool = 1u",
        "(app_global_put 0x6b6579 again#0)",
    ]
    # where it was not taken the opposite holds
    assert _block_ops(sub, 2)[0] == "let b_at_most_a#0: bool = 1u"
    # where both ways meet neither is known
    assert _block_ops(sub, 3)[0] == "let at_merge#0: bool = (<= a#0 b#0)"
    # the branch itself is as it was: nothing decides it
    assert str(sub.body[0].terminator) == "goto a_below_b#0 ? block@1 : block@2"

    # a branch on what an assert has established is decided
    sub = _diamond(
        [assign(a_below_b, intrinsic(AVMOp.lt, _a, _b)), assert_(a_below_b)],
        a_below_b,
        [assign(difference, intrinsic(AVMOp.sub, _b, _a)), *_use(difference)],
        [],
        [],
    )
    assert range_constant_outcomes(Mock(), sub)
    assert str(sub.body[0].terminator) == "goto 1u ? block@1 : block@2"
    # the assert that decides it stays
    assert _block_ops(sub, 0)[-1] == "(assert a_below_b#0)"
    # and so does an operation that the condition keeps from failing
    assert _block_ops(sub, 1)[0] == "let difference#0: uint64 = (- b#0 a#0)"


def _counter_loop(limit: models.Value, entry_ops: Sequence[models.Op]) -> models.Subroutine:
    #     header: i = phi(start, next); branch on i b< limit
    #     body:   assert i b< limit; next = i b+ 1; assert bitlen(next) <= 256
    start = _bytes("start")
    i = _bytes("i")
    following = _bytes("next")
    bits = register("bits")
    in_range = _condition("in_range")
    fits = _condition("fits")
    entry = block(0, [*entry_ops, assign(start, intrinsic(AVMOp.itob, uint64(0)))], return_())
    header = block(1, [assign(in_range, intrinsic(AVMOp.lt_bytes, i, limit))], return_())
    body = block(
        2,
        [
            assert_(in_range),
            assign(following, intrinsic(AVMOp.add_bytes, i, biguint(1))),
            assign(bits, intrinsic(AVMOp.bitlen, following)),
            assign(fits, intrinsic(AVMOp.lte, bits, uint64(256))),
            assert_(fits),
        ],
        return_(),
    )
    after = block(3, _use(i), return_())
    goto(entry, header)
    branch(header, in_range, non_zero=body, zero=after)
    goto(body, header)
    header.phis.append(phi(i, (start, entry), (following, body)))
    return subroutine("main", [entry, header, body, after], parameters=_PARAMETERS)


def test_checks_of_a_loop_counter() -> None:
    # below a limit of 32 bytes, the counter can be stepped without leaving 256 bits
    sub = _counter_loop(_a_word, _WORDS)
    assert range_constant_outcomes(Mock(), sub)
    sub.validate_with_ssa()
    assert _block_ops(sub, 2) == [
        "let next#0: bytes = (b+ i#0 1b)",
        "let bits#0: uint64 = (bitlen next#0)",
        "let fits#0: bool = 1u",
    ]
    # below a limit of which nothing is known, only the repeated check goes
    sub = _counter_loop(_data, [])
    assert range_constant_outcomes(Mock(), sub)
    assert _block_ops(sub, 2) == [
        "let next#0: bytes = (b+ i#0 1b)",
        "let bits#0: uint64 = (bitlen next#0)",
        "let fits#0: bool = (<= bits#0 256u)",
        "(assert fits#0)",
    ]


def test_modulo_of_value_below_the_modulus() -> None:
    small = register("small")
    again = register("again")
    total = _bytes("total")
    wrapped = _bytes("wrapped")
    ops = [
        assign(small, intrinsic(AVMOp.mod, _a, uint64(100))),
        assign(again, intrinsic(AVMOp.mod, small, uint64(256))),
        *_CONVERTED,
        assign(total, intrinsic(AVMOp.add_bytes, _a_bytes, _b_bytes)),
        assign(wrapped, intrinsic(AVMOp.mod_bytes, total, biguint(_WORD))),
        *_use(again, wrapped),
    ]
    result = _operand_results(ops)
    assert result[1] == "let again#0: uint64 = small#0"
    # the sum has no leading zero bytes, like the result of the modulo: the same bytes
    assert result[5] == "let wrapped#0: bytes = total#0"
    # the first modulo is needed
    assert result[0] == "let small#0: uint64 = (% a#0 100u)"


def test_modulo_of_value_that_may_reach_the_modulus_stays() -> None:
    total = _bytes("total")
    wrapped = _bytes("wrapped")
    narrow = _bytes("narrow")
    ops = [
        *_WORDS,
        assign(total, intrinsic(AVMOp.add_bytes, _a_word, _b_word)),
        assign(wrapped, intrinsic(AVMOp.mod_bytes, total, biguint(_WORD))),
        assign(narrow, intrinsic(AVMOp.mod_bytes, _a_word, biguint(2**128))),
        *_use(wrapped, narrow),
    ]
    assert _operand_results(ops) == _unchanged(ops)


def test_modulo_of_value_with_leading_zero_bytes() -> None:
    wrapped = _bytes("wrapped")
    total = _bytes("total")
    joined = _bytes("joined")
    is_small = _condition("is_small")
    bits = register("bits")
    modulo = intrinsic(AVMOp.mod_bytes, _a_word, biguint(_WORD))
    as_number: list[models.Op] = [
        assign(total, intrinsic(AVMOp.add_bytes, wrapped, biguint(1))),
        assign(is_small, intrinsic(AVMOp.lt_bytes, wrapped, _b_word)),
        assign(bits, intrinsic(AVMOp.bitlen, wrapped)),
        *_use(total, is_small, bits),
    ]
    # a word is below 2^256, but it is 32 bytes where the result of the modulo has as many
    # as it takes: something that reads the bytes of the result can tell
    for reads_bytes in (
        _use(wrapped),
        [assign(bits, intrinsic(AVMOp.len_, wrapped)), *_use(bits)],
        [assign(joined, intrinsic(AVMOp.concat, wrapped, _b_word)), *_use(joined)],
        [assign(bits, intrinsic(AVMOp.btoi, wrapped)), *_use(bits)],
    ):
        ops = [*_WORDS, assign(wrapped, modulo), *as_number[:2], *reads_bytes]
        assert _operand_results(ops) == _unchanged(ops)
    # byte math, its comparisons and bitlen read a number, and cannot
    ops = [*_WORDS, assign(wrapped, modulo), *as_number]
    assert _operand_results(ops)[2] == "let wrapped#0: bytes = a_word#0"


def test_modulo_that_can_fail_stays() -> None:
    long = _bytes("long")
    wrapped = _bytes("wrapped")
    total = _bytes("total")
    ops = [
        *_CONVERTED,
        # 68 bytes that are a number below 2^64
        assign(long, intrinsic(AVMOp.concat, bytes_(bytes(60)), _a_bytes)),
        assign(wrapped, intrinsic(AVMOp.mod_bytes, long, biguint(_WORD))),
        assign(total, intrinsic(AVMOp.add_bytes, wrapped, biguint(1))),
        *_use(total),
    ]
    assert _operand_results(ops) == _unchanged(ops)


def test_masks_that_cover_the_value() -> None:
    low = register("low")
    again = register("again")
    lower = register("lower")
    masked = _bytes("masked")
    partly = _bytes("partly")
    longer = _bytes("longer")
    ops = [
        assign(low, intrinsic(AVMOp.bitwise_and, _a, uint64(255))),
        assign(again, intrinsic(AVMOp.bitwise_and, low, uint64(1023))),
        assign(lower, intrinsic(AVMOp.bitwise_and, low, uint64(127))),
        *_WORDS,
        *_CONVERTED,
        assign(masked, intrinsic(AVMOp.bitwise_and_bytes, _a_word, bytes_(b"\xff" * 32))),
        assign(partly, intrinsic(AVMOp.bitwise_and_bytes, _a_word, bytes_(b"\xff" * 8))),
        # all the bits of the 8 bytes, but the result has the 32 bytes of the mask
        assign(longer, intrinsic(AVMOp.bitwise_and_bytes, _a_bytes, bytes_(b"\xff" * 32))),
        *_use(again, lower, masked, partly, longer),
    ]
    result = _operand_results(ops)
    assert result[1] == "let again#0: uint64 = low#0"
    assert result[7] == "let masked#0: bytes = a_word#0"
    # the others are as they were
    assert [result[i] for i in (0, 2, 8, 9)] == [str(ops[i]) for i in (0, 2, 8, 9)]


def test_pads_of_values_that_are_long_enough() -> None:
    zeros = _bytes("zeros")
    padded = _bytes("padded")
    padded_short = _bytes("padded_short")
    total = _bytes("total")
    ops = [
        *_WORDS,
        *_CONVERTED,
        assign(zeros, intrinsic(AVMOp.bzero, uint64(32))),
        assign(padded, intrinsic(AVMOp.bitwise_or_bytes, zeros, _a_word)),
        assign(padded_short, intrinsic(AVMOp.bitwise_or_bytes, _a_bytes, zeros)),
        *_use(padded, padded_short),
    ]
    result = _operand_results(ops)
    # a word is a word already, and 8 bytes are not
    assert result[5] == "let padded#0: bytes = a_word#0"
    assert result[6] == str(ops[6])
    # unless nothing reads more than a number in the result
    ops = [
        *_CONVERTED,
        assign(zeros, intrinsic(AVMOp.bzero, uint64(32))),
        assign(padded_short, intrinsic(AVMOp.bitwise_or_bytes, _a_bytes, zeros)),
        assign(total, intrinsic(AVMOp.add_bytes, padded_short, _b_bytes)),
        *_use(total),
    ]
    assert _operand_results(ops)[3] == "let padded_short#0: bytes = a_bytes#0"
    # zeros that byte math does not take: the pad makes the addition fail
    ops[2] = assign(zeros, intrinsic(AVMOp.bzero, uint64(100)))
    assert _operand_results(ops) == _unchanged(ops)


def test_parts_that_are_the_whole() -> None:
    names = ["rest", "all", "first", "also_all", "sized", "start"]
    rest, everything, first, also_all, sized, start = map(_bytes, names)
    ops = [
        *_WORDS,
        assign(rest, intrinsic(AVMOp.extract, _data, immediates=[0, 0])),
        assign(everything, intrinsic(AVMOp.extract, _a_word, immediates=[0, 32])),
        assign(first, intrinsic(AVMOp.extract, _a_word, immediates=[0, 8])),
        assign(also_all, intrinsic(AVMOp.substring, _a_word, immediates=[0, 32])),
        assign(sized, intrinsic(AVMOp.extract3, _a_word, uint64(0), uint64(32))),
        # may be all of it, or fail
        assign(start, intrinsic(AVMOp.extract, _data, immediates=[0, 32])),
        *_use(rest, everything, first, also_all, sized, start),
    ]
    result = _operand_results(ops)
    assert result[2:8] == [
        "let rest#0: bytes = data#0",
        "let all#0: bytes = a_word#0",
        str(ops[4]),
        "let also_all#0: bytes = a_word#0",
        "let sized#0: bytes = a_word#0",
        str(ops[7]),
    ]


def _wrapping_subtraction(
    a: models.Value, b: models.Value, before: Sequence[models.Op]
) -> list[models.Op]:
    raised = _bytes("raised")
    lowered = _bytes("lowered")
    result = _bytes("result")
    return [
        *_WORDS,
        *before,
        assign(raised, intrinsic(AVMOp.add_bytes, a, biguint(_WORD))),
        assign(lowered, intrinsic(AVMOp.sub_bytes, raised, b)),
        assign(result, intrinsic(AVMOp.mod_bytes, lowered, biguint(_WORD))),
        *_use(result),
    ]


def test_subtraction_that_does_not_wrap() -> None:
    b_at_most_a = _condition("b_at_most_a")
    known = [
        assign(b_at_most_a, intrinsic(AVMOp.lte_bytes, _b_word, _a_word)),
        assert_(b_at_most_a),
    ]
    # (a + 2^256 - b) % 2^256 where b <= a < 2^256
    ops = _wrapping_subtraction(_a_word, _b_word, known)
    result = _cheaper_forms(ops)
    assert result[-2] == "let result#0: bytes = (b- a_word#0 b_word#0)"
    # what is left over is for dead code elimination
    assert result[:-2] == _unchanged(ops)[:-2]

    # which of the two is larger is not known
    ops = _wrapping_subtraction(_a_word, _b_word, [])
    assert _cheaper_forms(ops) == _unchanged(ops)
    # the other one is
    ops = _wrapping_subtraction(_b_word, _a_word, known)
    assert _cheaper_forms(ops) == _unchanged(ops)
    # the first operand may be 2^256 or more, which the modulo takes off again
    known = [assign(b_at_most_a, intrinsic(AVMOp.lte_bytes, _b_word, _data)), assert_(b_at_most_a)]
    ops = _wrapping_subtraction(_data, _b_word, known)
    assert _cheaper_forms(ops) == _unchanged(ops)


def test_comparison_of_converted_uint64_values() -> None:
    below = _condition("below")
    at_least_ten = _condition("at_least_ten")
    huge = _condition("huge")
    mixed = _condition("mixed")
    ops = [
        *_WORDS,
        *_CONVERTED,
        assign(below, intrinsic(AVMOp.lt_bytes, _a_bytes, _b_bytes)),
        assign(at_least_ten, intrinsic(AVMOp.gte_bytes, _a_bytes, biguint(10))),
        # not a uint64
        assign(huge, intrinsic(AVMOp.lt_bytes, _a_bytes, biguint(2**64))),
        assign(mixed, intrinsic(AVMOp.lt_bytes, _a_bytes, _b_word)),
        *_use(below, at_least_ten, huge, mixed),
    ]
    result = _cheaper_forms(ops)
    assert result[4:8] == [
        "let below#0: bool = (< a#0 b#0)",
        "let at_least_ten#0: bool = (>= a#0 10u)",
        str(ops[6]),
        str(ops[7]),
    ]


def test_byte_math_on_converted_uint64_values() -> None:
    small_a = register("small_a")
    small_b = register("small_b")
    total = _bytes("total")
    is_big = _condition("is_big")
    back = register("back")

    def math(op: AVMOp, x: models.Value, y: models.Value, uses: Sequence[models.Op]) -> list[str]:
        return _cheaper_forms(
            [
                assign(small_a, intrinsic(AVMOp.mod, _a, uint64(1000))),
                assign(small_b, intrinsic(AVMOp.mod, _b, uint64(1000))),
                assign(_a_bytes, intrinsic(AVMOp.itob, x)),
                assign(_b_bytes, intrinsic(AVMOp.itob, y)),
                assign(total, intrinsic(op, _a_bytes, _b_bytes)),
                *uses,
            ]
        )

    as_number: list[models.Op] = [
        assign(is_big, intrinsic(AVMOp.gt_bytes, total, _data)),
        assign(back, intrinsic(AVMOp.btoi, total)),
        *_use(is_big, back),
    ]
    unchanged = "let total#0: bytes = (b+ a_bytes#0 b_bytes#0)"
    # the sum of two small numbers is a uint64, and its bytes are only read as a number
    assert math(AVMOp.add_bytes, small_a, small_b, as_number)[4:6] == [
        "let uint64%total#0: uint64 = (+ small_a#0 small_b#0)",
        "let total#0: bytes = (itob uint64%total#0)",
    ]
    # the sum of any two uint64 values may not be one
    assert math(AVMOp.add_bytes, _a, _b, as_number)[4] == unchanged
    # the result has 8 bytes then, which shows where its bytes are read
    assert math(AVMOp.add_bytes, small_a, small_b, _use(total))[4] == unchanged
    # not worth it where both conversions are needed anyway
    assert (
        math(AVMOp.add_bytes, small_a, small_b, [*as_number, *_use(_a_bytes, _b_bytes)])[4]
        == unchanged
    )
    # subtraction, division and modulo fail for the same operands either way
    assert math(AVMOp.sub_bytes, _a, _b, as_number)[4] == (
        "let uint64%total#0: uint64 = (- a#0 b#0)"
    )
    assert math(AVMOp.div_floor_bytes, _a, _b, as_number)[4] == (
        "let uint64%total#0: uint64 = (/ a#0 b#0)"
    )
    assert math(AVMOp.mod_bytes, _a, _b, as_number)[4] == (
        "let uint64%total#0: uint64 = (% a#0 b#0)"
    )
    assert math(AVMOp.mul_bytes, small_a, small_b, as_number)[4] == (
        "let uint64%total#0: uint64 = (* small_a#0 small_b#0)"
    )
    assert math(AVMOp.mul_bytes, _a, small_b, as_number)[4] == (
        "let total#0: bytes = (b* a_bytes#0 b_bytes#0)"
    )


def test_last_bytes_as_uint64() -> None:
    low = _bytes("low")
    number = register("number")

    def last_bytes(whole: models.Value, modulus: int) -> str:
        return _cheaper_forms(
            [
                *_WORDS,
                assign(low, intrinsic(AVMOp.mod_bytes, whole, biguint(modulus))),
                assign(number, intrinsic(AVMOp.btoi, low)),
                *_use(number),
            ]
        )[3]

    assert last_bytes(_a_word, 2**64) == "let number#0: uint64 = (extract_uint64 a_word#0 24u)"
    assert last_bytes(_a_word, 2**32) == "let number#0: uint64 = (extract_uint32 a_word#0 28u)"
    assert last_bytes(_a_word, 2**16) == "let number#0: uint64 = (extract_uint16 a_word#0 30u)"
    assert last_bytes(_a_word, 2**8) == "let number#0: uint64 = (getbyte a_word#0 31u)"
    # no op for 3 bytes, and no position for the last bytes of a byte array of unknown length
    unchanged = "let number#0: uint64 = (btoi low#0)"
    assert last_bytes(_a_word, 2**24) == unchanged
    assert last_bytes(_data, 2**64) == unchanged


def test_groups_of_rewrites() -> None:
    small = register("small")
    is_small = _condition("is_small")
    again = register("again")
    below = _condition("below")
    ops = [
        *_CONVERTED,
        assign(small, intrinsic(AVMOp.mod, _a, uint64(100))),
        assign(is_small, intrinsic(AVMOp.lt, small, uint64(100))),
        assign(again, intrinsic(AVMOp.mod, small, uint64(256))),
        assign(below, intrinsic(AVMOp.lt_bytes, _a_bytes, _b_bytes)),
        *_use(is_small, again, below),
    ]
    rewritten = {
        "range_constant_outcomes": (3, "let is_small#0: bool = 1u"),
        "range_operand_results": (4, "let again#0: uint64 = small#0"),
        "range_cheaper_forms": (5, "let below#0: bool = (< a#0 b#0)"),
    }

    def rewrite(**switches: bool) -> list[str]:
        def with_switches(_: IROptimizationContext, subroutine: models.Subroutine) -> bool:
            context = Mock()
            context.options.optimizations_override = switches
            return range_rewrites(context, subroutine)

        return _rewritten(with_switches, ops)

    # one pass over the subroutine does the rewrites of all groups
    expected = _unchanged(ops)
    for index, replacement in rewritten.values():
        expected[index] = replacement
    assert rewrite() == expected
    # each group is switched off by its name
    for group in rewritten:
        expected = _unchanged(ops)
        for index, replacement in (rewritten[other] for other in rewritten if other != group):
            expected[index] = replacement
        assert rewrite(**{group: False}) == expected
    assert rewrite(**dict.fromkeys(rewritten, False)) == _unchanged(ops)


def test_optimization_that_depends_on_the_subroutine_alone() -> None:
    runs = list[str]()

    def looks(_: IROptimizationContext, subroutine: models.Subroutine) -> bool:
        runs.append(subroutine.id)
        return False

    def modifies(_: IROptimizationContext, _subroutine: models.Subroutine) -> bool:
        return True

    context = Mock(modifications={})
    sub = subroutine("main", [block(0, [], return_())])
    other = subroutine("other", [block(0, [], return_())])
    local = SubroutineOptimization.from_function(looks, local=True)
    # it is not run again on a subroutine that it left as it was
    assert not local.optimize(context, sub)
    assert not local.optimize(context, sub)
    assert runs == ["main"]
    # until an optimization modifies that subroutine
    assert SubroutineOptimization.from_function(modifies).optimize(context, other)
    assert not local.optimize(context, sub)
    assert runs == ["main"]
    assert SubroutineOptimization.from_function(modifies).optimize(context, sub)
    assert not local.optimize(context, sub)
    assert runs == ["main", "main"]
    # an optimization that may depend on more is run every time
    every_time = SubroutineOptimization.from_function(looks)
    assert not every_time.optimize(context, sub)
    assert not every_time.optimize(context, sub)
    assert runs == ["main"] * 4
