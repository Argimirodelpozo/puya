import pytest

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.effects import (
    _INTRINSIC_EFFECTS,
    EVERYTHING,
    OBSERVABLE,
    Effect,
    analyse_subroutine_effects,
    changed_by,
    effects_of,
    observers_of,
)
from puya.ir.optimize.sink_single_use_intrinsics import _is_unconditionally_movable
from puya.ir.optimize.value_bounds import ANY_LENGTH, ANY_UINT64, ValueBounds
from puya.ir.types_ import PrimitiveIRType, SizedBytesType, SlotType
from tests.utils.ir import (
    assign,
    block,
    bytes_,
    call,
    goto,
    intrinsic,
    parameter,
    program,
    register,
    return_,
    subroutine,
    uint64,
)

_MAX_UINT64 = 2**64 - 1

_a = register("a")
_b = register("b")
_data = register("data", PrimitiveIRType.bytes)
_key = bytes_(b"key")


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        # values that cannot fail
        (intrinsic(AVMOp.itob, _a), Effect.none),
        (intrinsic(AVMOp.len_, _data), Effect.none),
        (intrinsic(AVMOp.sha256, _data), Effect.none),
        (intrinsic(AVMOp.addw, _a, _b), Effect.none),
        (intrinsic(AVMOp.bitwise_or_bytes, _data, _data), Effect.none),
        # values that can
        (intrinsic(AVMOp.add, _a, _b), Effect.fails),
        (intrinsic(AVMOp.sub, _a, _b), Effect.fails),
        (intrinsic(AVMOp.mul, _a, _b), Effect.fails),
        (intrinsic(AVMOp.exp, _a, _b), Effect.fails),
        (intrinsic(AVMOp.div_floor, _a, _b), Effect.fails),
        (intrinsic(AVMOp.shl, _a, _b), Effect.fails),
        (intrinsic(AVMOp.btoi, _data), Effect.fails),
        (intrinsic(AVMOp.concat, _data, _data), Effect.fails),
        (intrinsic(AVMOp.extract, _data, immediates=[0, 8]), Effect.fails),
        (intrinsic(AVMOp.substring3, _data, _a, _b), Effect.fails),
        (intrinsic(AVMOp.add_bytes, _data, _data), Effect.fails),
        (intrinsic(AVMOp.sub_bytes, _data, _data), Effect.fails),
        # what is fixed while the program runs
        (intrinsic(AVMOp.txn, immediates=["Sender"]), Effect.reads_txn),
        (intrinsic(AVMOp.txn, immediates=["FirstValidTime"]), Effect.reads_txn | Effect.fails),
        (intrinsic(AVMOp.txn, immediates=["LastLog"]), Effect.reads_txn | Effect.fails),
        (intrinsic(AVMOp.txn, immediates=["ApprovalProgram"]), Effect.reads_txn | Effect.fails),
        (
            intrinsic(AVMOp.txna, immediates=["ApplicationArgs", 1]),
            Effect.reads_txn | Effect.fails,
        ),
        (intrinsic(AVMOp.gtxns, _a, immediates=["Amount"]), Effect.reads_txn | Effect.fails),
        (intrinsic(AVMOp.global_, immediates=["GroupSize"]), Effect.reads_txn),
        (
            intrinsic(AVMOp.global_, immediates=["LatestTimestamp"]),
            Effect.reads_txn | Effect.fails,
        ),
        (intrinsic(AVMOp.global_, immediates=["OpcodeBudget"]), Effect.reads_budget),
        # scratch space
        (
            intrinsic(AVMOp.load, immediates=[7], types=[PrimitiveIRType.uint64]),
            Effect.reads_scratch,
        ),
        (
            intrinsic(AVMOp.loads, _a, types=[PrimitiveIRType.uint64]),
            Effect.reads_scratch | Effect.fails,
        ),
        (intrinsic(AVMOp.store, _a, immediates=[7]), Effect.writes_scratch),
        (intrinsic(AVMOp.stores, _a, _b), Effect.writes_scratch | Effect.fails),
        # application state
        (
            intrinsic(AVMOp.app_global_get, _key, types=[PrimitiveIRType.uint64]),
            Effect.reads_global,
        ),
        (
            intrinsic(
                AVMOp.app_global_get_ex,
                uint64(0),
                _key,
                types=[PrimitiveIRType.uint64, PrimitiveIRType.bool],
            ),
            Effect.reads_global,
        ),
        (
            intrinsic(
                AVMOp.app_global_get_ex,
                _a,
                _key,
                types=[PrimitiveIRType.uint64, PrimitiveIRType.bool],
            ),
            Effect.reads_global | Effect.reads_ledger | Effect.fails,
        ),
        (intrinsic(AVMOp.app_global_put, _key, _a), Effect.writes_global | Effect.fails),
        (intrinsic(AVMOp.app_global_del, _key), Effect.writes_global | Effect.fails),
        # boxes: contents, size and existence are separate
        (intrinsic(AVMOp.box_len, _key), Effect.reads_box_size | Effect.fails),
        (intrinsic(AVMOp.box_get, _key), Effect.reads_box | Effect.reads_box_size | Effect.fails),
        (
            intrinsic(AVMOp.box_replace, _key, _a, _data),
            Effect.reads_box_size | Effect.writes_box | Effect.fails,
        ),
        (
            intrinsic(AVMOp.box_put, _key, _data),
            Effect.reads_box_size | Effect.writes_box | Effect.allocates_box | Effect.fails,
        ),
        (
            intrinsic(AVMOp.box_resize, _key, _a),
            Effect.reads_box_size | Effect.allocates_box | Effect.fails,
        ),
        (
            intrinsic(AVMOp.box_del, _key),
            Effect.reads_box_size | Effect.deletes_box | Effect.fails,
        ),
        # the ledger
        (intrinsic(AVMOp.min_balance, _a), Effect.reads_ledger | Effect.fails),
        # logs
        (intrinsic(AVMOp.log, _data), Effect.logs | Effect.fails),
        # inner transactions
        (intrinsic(AVMOp.itxn_begin), Effect.writes_itxn | Effect.fails),
        (
            intrinsic(AVMOp.itxn, immediates=["CreatedApplicationID"]),
            Effect.reads_itxn | Effect.fails,
        ),
    ],
    ids=str,
)
def test_intrinsic_effects(op: models.Intrinsic, expected: Effect) -> None:
    assert effects_of(op) == expected


def test_every_avm_op_is_classified() -> None:
    unclassified = [op.code for op in AVMOp if op not in _INTRINSIC_EFFECTS]
    # An op that is not classified may do everything, which is safe but stops optimizations
    # around it. Classify a new op in puya.ir.optimize.effects, or list it here if not enough
    # is known about it.
    assert unclassified == []


def test_inner_transactions_can_see_and_change_state() -> None:
    effects = effects_of(intrinsic(AVMOp.itxn_submit))
    # another application can read the state and boxes of this one
    assert effects & Effect.reads_global
    assert effects & Effect.reads_local
    assert effects & Effect.reads_box
    # and with shared box access it can change the boxes
    assert effects & Effect.writes_box
    assert effects & Effect.allocates_box
    assert effects & Effect.deletes_box
    # it cannot call this application, so the state of this application stays
    assert not effects & Effect.writes_global
    assert not effects & Effect.writes_local
    assert not effects & Effect.writes_scratch


@pytest.mark.parametrize(
    ("op", "may_fail"),
    [
        # uint64 arithmetic
        (intrinsic(AVMOp.add, uint64(1), uint64(2)), False),
        (intrinsic(AVMOp.add, uint64(_MAX_UINT64), uint64(1)), True),
        (intrinsic(AVMOp.sub, uint64(2), uint64(2)), False),
        (intrinsic(AVMOp.sub, uint64(1), uint64(2)), True),
        (intrinsic(AVMOp.mul, uint64(2**32), uint64(2**32 - 1)), False),
        (intrinsic(AVMOp.mul, uint64(2**32), uint64(2**32)), True),
        (intrinsic(AVMOp.div_floor, _a, uint64(2)), False),
        (intrinsic(AVMOp.div_floor, _a, uint64(0)), True),
        (intrinsic(AVMOp.mod, _a, uint64(0)), True),
        (intrinsic(AVMOp.exp, uint64(2), uint64(63)), False),
        (intrinsic(AVMOp.exp, uint64(2), uint64(64)), True),
        (intrinsic(AVMOp.exp, uint64(0), uint64(0)), True),
        (intrinsic(AVMOp.exp, _a, uint64(0)), True),
        (intrinsic(AVMOp.exp, uint64(1), _a), False),
        (intrinsic(AVMOp.expw, uint64(2), uint64(127)), False),
        (intrinsic(AVMOp.expw, uint64(2), uint64(128)), True),
        (intrinsic(AVMOp.divw, uint64(0), _a, uint64(1)), False),
        (intrinsic(AVMOp.divw, uint64(1), _a, uint64(1)), True),
        (intrinsic(AVMOp.divmodw, _a, _a, uint64(0), uint64(1)), False),
        (intrinsic(AVMOp.divmodw, _a, _a, uint64(0), uint64(0)), True),
        (intrinsic(AVMOp.shl, _a, uint64(63)), False),
        (intrinsic(AVMOp.shl, _a, uint64(64)), True),
        (intrinsic(AVMOp.shr, _a, uint64(64)), True),
        # byte arrays
        (intrinsic(AVMOp.btoi, bytes_(b"12345678")), False),
        (intrinsic(AVMOp.btoi, bytes_(b"123456789")), True),
        (intrinsic(AVMOp.bzero, uint64(4096)), False),
        (intrinsic(AVMOp.bzero, uint64(4097)), True),
        (intrinsic(AVMOp.concat, bytes_(b"a" * 4000), bytes_(b"b" * 96)), False),
        (intrinsic(AVMOp.concat, bytes_(b"a" * 4000), bytes_(b"b" * 97)), True),
        (intrinsic(AVMOp.extract, bytes_(b"abcd"), immediates=[1, 3]), False),
        (intrinsic(AVMOp.extract, bytes_(b"abcd"), immediates=[2, 3]), True),
        (intrinsic(AVMOp.extract, bytes_(b"abcd"), immediates=[4, 0]), False),
        (intrinsic(AVMOp.extract, bytes_(b"abcd"), immediates=[5, 0]), True),
        (intrinsic(AVMOp.extract3, bytes_(b"abcd"), uint64(1), uint64(3)), False),
        (intrinsic(AVMOp.extract3, bytes_(b"abcd"), uint64(1), uint64(4)), True),
        (intrinsic(AVMOp.substring, bytes_(b"abcd"), immediates=[1, 4]), False),
        (intrinsic(AVMOp.substring, bytes_(b"abcd"), immediates=[3, 2]), True),
        (intrinsic(AVMOp.substring3, bytes_(b"abcd"), uint64(2), uint64(5)), True),
        (intrinsic(AVMOp.extract_uint16, bytes_(b"abcd"), uint64(2)), False),
        (intrinsic(AVMOp.extract_uint16, bytes_(b"abcd"), uint64(3)), True),
        (intrinsic(AVMOp.extract_uint64, bytes_(b"abcd"), uint64(0)), True),
        (intrinsic(AVMOp.getbyte, bytes_(b"abcd"), uint64(3)), False),
        (intrinsic(AVMOp.getbyte, bytes_(b"abcd"), uint64(4)), True),
        (intrinsic(AVMOp.setbyte, bytes_(b"abcd"), uint64(0), uint64(255)), False),
        (intrinsic(AVMOp.setbyte, bytes_(b"abcd"), uint64(0), uint64(256)), True),
        (intrinsic(AVMOp.getbit, _a, uint64(63)), False),
        (intrinsic(AVMOp.getbit, _a, uint64(64)), True),
        (intrinsic(AVMOp.getbit, bytes_(b"ab"), uint64(15)), False),
        (intrinsic(AVMOp.getbit, bytes_(b"ab"), uint64(16)), True),
        (intrinsic(AVMOp.setbit, _a, uint64(0), uint64(2), types=[PrimitiveIRType.uint64]), True),
        (intrinsic(AVMOp.replace2, bytes_(b"abcd"), bytes_(b"xy"), immediates=[2]), False),
        (intrinsic(AVMOp.replace2, bytes_(b"abcd"), bytes_(b"xy"), immediates=[3]), True),
        (intrinsic(AVMOp.replace3, bytes_(b"abcd"), uint64(3), bytes_(b"xy")), True),
        # byte math takes at most 64 bytes
        (intrinsic(AVMOp.add_bytes, bytes_(b"a" * 64), bytes_(b"b" * 64)), False),
        (intrinsic(AVMOp.add_bytes, bytes_(b"a" * 65), bytes_(b"b")), True),
        (intrinsic(AVMOp.eq_bytes, bytes_(b"a" * 65), bytes_(b"b")), True),
        (intrinsic(AVMOp.sub_bytes, bytes_(b"\x02"), bytes_(b"\x01")), False),
        (intrinsic(AVMOp.sub_bytes, bytes_(b"\x01"), bytes_(b"\x02")), True),
        (intrinsic(AVMOp.div_floor_bytes, bytes_(b"\x01"), bytes_(b"\x00")), True),
        (intrinsic(AVMOp.mod_bytes, bytes_(b"\x01"), bytes_(b"\x02")), False),
        # but the bitwise ops take any length
        (intrinsic(AVMOp.bitwise_and_bytes, bytes_(b"a" * 65), bytes_(b"b")), False),
        # comparing values of different types fails
        (intrinsic(AVMOp.eq, _a, _b), False),
        (intrinsic(AVMOp.eq, _a, _data), True),
        # scratch slots
        (intrinsic(AVMOp.loads, uint64(255), types=[PrimitiveIRType.uint64]), False),
        (intrinsic(AVMOp.loads, uint64(256), types=[PrimitiveIRType.uint64]), True),
        # state keys are at most 64 bytes
        (intrinsic(AVMOp.app_global_get, bytes_(b"k" * 65), types=[PrimitiveIRType.uint64]), True),
    ],
    ids=str,
)
def test_failure_with_constant_operands(op: models.Intrinsic, *, may_fail: bool) -> None:
    assert bool(effects_of(op) & Effect.fails) == may_fail


def test_failure_with_bounded_operands() -> None:
    param = parameter("param")
    arg = parameter("arg", PrimitiveIRType.bytes)
    small = register("small")
    word = register("word", PrimitiveIRType.bytes)
    padded = register("padded", PrimitiveIRType.bytes)
    length = register("length")
    sum_ = register("sum")
    difference = register("difference")
    ops = [
        assign(small, intrinsic(AVMOp.mod, param, uint64(32))),
        assign(word, intrinsic(AVMOp.itob, param)),
        assign(padded, intrinsic(AVMOp.concat, word, bytes_(b"\x00" * 24))),
        assign(length, intrinsic(AVMOp.len_, padded)),
        assign(sum_, intrinsic(AVMOp.add, small, length)),
        assign(difference, intrinsic(AVMOp.sub, length, small)),
    ]
    sub = subroutine("bounded", [block(0, ops, return_())], parameters=[param, arg], returns=[])
    bounds = ValueBounds.of(sub)

    assert bounds.uint64(param) == ANY_UINT64
    assert bounds.length(arg) == ANY_LENGTH
    assert bounds.uint64(small) == (0, 31)
    assert bounds.length(word) == (8, 8)
    assert bounds.length(padded) == (32, 32)
    assert bounds.uint64(length) == (32, 32)
    assert bounds.uint64(sum_) == (32, 63)
    assert bounds.uint64(difference) == (1, 32)

    def may_fail(op: models.Intrinsic) -> bool:
        return Effect.fails in effects_of(op, bounds=bounds)

    # the operands rule a failure out
    assert not may_fail(intrinsic(AVMOp.add, small, length))
    assert not may_fail(intrinsic(AVMOp.sub, length, small))
    assert not may_fail(intrinsic(AVMOp.shl, param, small))
    assert not may_fail(intrinsic(AVMOp.btoi, word))
    assert not may_fail(intrinsic(AVMOp.extract, padded, immediates=[24, 8]))
    assert not may_fail(intrinsic(AVMOp.extract3, padded, small, uint64(1)))
    assert not may_fail(intrinsic(AVMOp.getbyte, padded, small))
    assert not may_fail(intrinsic(AVMOp.add_bytes, padded, word))
    assert not may_fail(intrinsic(AVMOp.bzero, small))
    # the operands do not
    assert may_fail(intrinsic(AVMOp.add, param, small))
    assert may_fail(intrinsic(AVMOp.sub, small, length))
    assert may_fail(intrinsic(AVMOp.shl, param, param))
    assert may_fail(intrinsic(AVMOp.btoi, padded))
    assert may_fail(intrinsic(AVMOp.btoi, arg))
    assert may_fail(intrinsic(AVMOp.extract, padded, immediates=[25, 8]))
    assert may_fail(intrinsic(AVMOp.extract3, padded, small, uint64(2)))
    assert may_fail(intrinsic(AVMOp.getbyte, word, small))
    # without bounds, nothing is known about a register
    assert effects_of(intrinsic(AVMOp.add, small, length)) == Effect.fails


def test_declared_type_of_register_is_not_evidence() -> None:
    claimed = parameter("claimed", SizedBytesType(8))
    sub = subroutine("typed", [block(0, [], return_())], parameters=[claimed])
    bounds = ValueBounds.of(sub)
    assert bounds.length(claimed) == ANY_LENGTH
    assert effects_of(intrinsic(AVMOp.btoi, claimed), bounds=bounds) == Effect.fails


def test_bounds_of_value_defined_by_loop() -> None:
    # i#1 = phi(0, i#2); i#2 = i#1 + 1
    start = register("start")
    i1 = models.Register(name="i", version=1, ir_type=PrimitiveIRType.uint64, source_location=None)
    i2 = models.Register(name="i", version=2, ir_type=PrimitiveIRType.uint64, source_location=None)
    entry = block(0, [assign(start, uint64(0))], return_())
    loop = block(1, [assign(i2, intrinsic(AVMOp.add, i1, uint64(1)))], return_())
    goto(entry, loop)
    goto(loop, loop)
    loop.phis.append(
        models.Phi(
            register=i1,
            args=[
                models.PhiArgument(value=start, through=entry),
                models.PhiArgument(value=i2, through=loop),
            ],
        )
    )
    bounds = ValueBounds.of(subroutine("loop", [entry, loop]))
    assert bounds.uint64(i1) == ANY_UINT64
    assert bounds.uint64(i2) == (1, _MAX_UINT64)


def test_bounds_given_conditions() -> None:
    a = parameter("a")
    b = parameter("b")
    data = parameter("data", PrimitiveIRType.bytes)
    length = register("length")
    is_word = register("is_word", PrimitiveIRType.bool)
    a_below_b = register("a_below_b", PrimitiveIRType.bool)
    b_is_small = register("b_is_small", PrimitiveIRType.bool)
    both = register("both", PrimitiveIRType.bool)
    neither = register("neither", PrimitiveIRType.bool)
    ops = [
        assign(length, intrinsic(AVMOp.len_, data)),
        assign(is_word, intrinsic(AVMOp.eq, length, uint64(8))),
        assign(a_below_b, intrinsic(AVMOp.lt, a, b)),
        assign(b_is_small, intrinsic(AVMOp.lte, b, uint64(100))),
        assign(both, intrinsic(AVMOp.and_, a_below_b, b_is_small)),
        assign(neither, intrinsic(AVMOp.not_, both)),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())], parameters=[a, b, data]))

    given = bounds.given(non_zero=[is_word])
    assert given.length(data) == (8, 8)
    assert given.uint64(length) == (8, 8)
    assert Effect.fails not in effects_of(intrinsic(AVMOp.btoi, data), bounds=given)

    given = bounds.given(non_zero=[both])
    assert given.uint64(b) == (1, 100)
    assert given.uint64(a) == (0, 99)
    assert Effect.fails not in effects_of(intrinsic(AVMOp.sub, b, a), bounds=given)
    assert Effect.fails in effects_of(intrinsic(AVMOp.sub, a, b), bounds=given)

    # a condition that is 0 says the opposite
    given = bounds.given(zero=[a_below_b])
    assert Effect.fails not in effects_of(intrinsic(AVMOp.sub, a, b), bounds=given)
    assert bounds.given(zero=[neither]).uint64(a) == (0, 99)
    # one of the two does not hold, but it is not known which
    assert bounds.given(non_zero=[neither]).uint64(a) == ANY_UINT64

    # the bounds they were derived from are as they were
    assert bounds.length(data) == ANY_LENGTH
    assert bounds.uint64(a) == ANY_UINT64


def test_bounds_given_completed_operations() -> None:
    a = parameter("a")
    b = parameter("b")
    data = parameter("data", PrimitiveIRType.bytes)
    other = parameter("other", PrimitiveIRType.bytes)
    bounds = ValueBounds.of(
        subroutine("s", [block(0, [], return_())], parameters=[a, b, data, other])
    )

    def given(*completed: models.Intrinsic) -> ValueBounds:
        return bounds.given(completed=completed)

    # reading bytes says how long the byte array is at least
    assert given(intrinsic(AVMOp.extract, data, immediates=[24, 8])).length(data) == (32, 4096)
    assert given(intrinsic(AVMOp.extract_uint64, data, uint64(24))).length(data) == (32, 4096)
    assert given(intrinsic(AVMOp.extract_uint16, data, uint64(2))).length(data) == (4, 4096)
    assert given(intrinsic(AVMOp.getbyte, data, uint64(31))).length(data) == (32, 4096)
    assert given(intrinsic(AVMOp.extract3, data, uint64(4), uint64(4))).length(data) == (8, 4096)
    assert given(intrinsic(AVMOp.substring, data, immediates=[2, 6])).length(data) == (6, 4096)
    # where the position is not known, only what is known about it counts
    assert given(intrinsic(AVMOp.extract_uint64, data, a)).length(data) == (8, 4096)
    # other operations say how long it is at most
    assert given(intrinsic(AVMOp.btoi, data)).length(data) == (0, 8)
    assert given(intrinsic(AVMOp.add_bytes, data, other)).length(other) == (0, 64)
    assert given(intrinsic(AVMOp.concat, data, bytes_(b"1234"))).length(data) == (0, 4092)
    # and uint64 operands are bounded by what does not fail
    assert given(intrinsic(AVMOp.div_floor, a, b)).uint64(b) == (1, _MAX_UINT64)
    assert given(intrinsic(AVMOp.shl, a, b)).uint64(b) == (0, 63)
    assert given(intrinsic(AVMOp.bzero, a)).uint64(a) == (0, 4096)
    assert given(intrinsic(AVMOp.sub, a, b)).not_above(b, a)
    assert not given(intrinsic(AVMOp.sub, a, b)).not_above(a, b)
    assert given(intrinsic(AVMOp.add, a, uint64(64))).uint64(a) == (0, _MAX_UINT64 - 64)
    assert given(intrinsic(AVMOp.mul, a, uint64(2**32))).uint64(a) == (0, 2**32 - 1)
    # an operation that fails for no reason that is known here says nothing
    assert given(intrinsic(AVMOp.add, a, b)).uint64(a) == ANY_UINT64
    assert given(intrinsic(AVMOp.exp, a, b)).uint64(a) == ANY_UINT64

    # each is found not to fail once the same operation has completed
    for op in (
        intrinsic(AVMOp.extract, data, immediates=[24, 8]),
        intrinsic(AVMOp.extract_uint32, data, uint64(7)),
        intrinsic(AVMOp.getbyte, data, uint64(31)),
        intrinsic(AVMOp.substring, data, immediates=[2, 6]),
        intrinsic(AVMOp.btoi, data),
        intrinsic(AVMOp.add_bytes, data, other),
        intrinsic(AVMOp.concat, data, bytes_(b"1234")),
        intrinsic(AVMOp.div_floor, a, b),
        intrinsic(AVMOp.shr, a, b),
        intrinsic(AVMOp.bzero, a),
        intrinsic(AVMOp.sub, a, b),
        intrinsic(AVMOp.add, a, uint64(32)),
        intrinsic(AVMOp.mul, a, uint64(32)),
        intrinsic(AVMOp.loads, a, types=[PrimitiveIRType.uint64]),
    ):
        assert Effect.fails in effects_of(op, bounds=bounds), str(op)
        assert Effect.fails not in effects_of(op, bounds=given(op)), str(op)


def test_reads_of_what_is_fixed_give_the_same_value() -> None:
    first = register("first", PrimitiveIRType.bytes)
    second = register("second", PrimitiveIRType.bytes)
    other = register("other", PrimitiveIRType.bytes)
    budget = register("budget")
    budget_again = register("budget_again")
    length = register("length")
    is_long = register("is_long", PrimitiveIRType.bool)
    ops = [
        assign(first, intrinsic(AVMOp.txna, immediates=["ApplicationArgs", 1])),
        assign(second, intrinsic(AVMOp.txna, immediates=["ApplicationArgs", 1])),
        assign(other, intrinsic(AVMOp.txna, immediates=["ApplicationArgs", 2])),
        assign(budget, intrinsic(AVMOp.global_, immediates=["OpcodeBudget"])),
        assign(budget_again, intrinsic(AVMOp.global_, immediates=["OpcodeBudget"])),
        assign(length, intrinsic(AVMOp.len_, first)),
        assign(is_long, intrinsic(AVMOp.gte, length, uint64(64))),
    ]
    bounds = ValueBounds.of(subroutine("s", [block(0, ops, return_())]))
    assert bounds.same_value(second) == bounds.same_value(first)
    assert bounds.same_value(other) != bounds.same_value(first)
    # the opcode budget is different each time
    assert bounds.same_value(budget_again) != bounds.same_value(budget)

    # what is known about one read holds for the other
    given = bounds.given(non_zero=[is_long])
    assert given.length(first) == (64, 4096)
    assert given.length(second) == (64, 4096)
    assert given.length(other) == ANY_LENGTH
    given = bounds.given(completed=[intrinsic(AVMOp.extract, second, immediates=[32, 32])])
    assert Effect.fails not in effects_of(
        intrinsic(AVMOp.extract, first, immediates=[0, 64]), bounds=given
    )


def test_sum_within_length() -> None:
    start = parameter("start")
    size = parameter("size")
    data = parameter("data", PrimitiveIRType.bytes)
    other = parameter("other", PrimitiveIRType.bytes)
    end = register("end")
    length = register("length")
    fits = register("fits", PrimitiveIRType.bool)
    ops = [
        assign(end, intrinsic(AVMOp.add, start, size)),
        assign(length, intrinsic(AVMOp.len_, data)),
        assign(fits, intrinsic(AVMOp.lte, end, length)),
    ]
    bounds = ValueBounds.of(
        subroutine("s", [block(0, ops, return_())], parameters=[start, size, data, other])
    )
    part = intrinsic(AVMOp.extract3, data, start, size)
    assert Effect.fails in effects_of(part, bounds=bounds)
    given = bounds.given(non_zero=[fits])
    assert given.within(data, start, size)
    assert given.within(data, size, start)
    assert Effect.fails not in effects_of(part, bounds=given)
    # it is this sum and this byte array that the condition is about
    assert not given.within(data, start, start)
    assert not given.within(other, start, size)
    assert Effect.fails in effects_of(intrinsic(AVMOp.extract3, other, start, size), bounds=given)
    assert Effect.fails in effects_of(intrinsic(AVMOp.extract3, data, size, size), bounds=given)
    # and it has to hold
    assert not bounds.given(zero=[fits]).within(data, start, size)
    # bounds on the operands are enough as well
    assert bounds.within(bytes_(b"1234"), uint64(2), uint64(2))
    assert not bounds.within(bytes_(b"1234"), uint64(2), uint64(3))


def test_within_what_a_completed_operation_took() -> None:
    index = parameter("index")
    size = parameter("size")
    data = parameter("data", PrimitiveIRType.bytes)
    other = parameter("other", PrimitiveIRType.bytes)
    bounds = ValueBounds.of(
        subroutine("s", [block(0, [], return_())], parameters=[index, size, data, other])
    )

    def may_fail(op: models.Intrinsic, *completed: models.Intrinsic) -> bool:
        return Effect.fails in effects_of(op, bounds=bounds.given(completed=completed))

    number = intrinsic(AVMOp.extract_uint64, data, index)
    word = intrinsic(AVMOp.extract3, data, index, uint64(8))
    # taking 8 bytes at an index fails exactly when reading a uint64 there does
    assert may_fail(word)
    assert not may_fail(word, number)
    assert not may_fail(number, word)
    # and whenever taking less from there does
    assert not may_fail(intrinsic(AVMOp.extract_uint32, data, index), number)
    assert not may_fail(intrinsic(AVMOp.getbyte, data, index), number)
    assert not may_fail(intrinsic(AVMOp.extract3, data, index, uint64(7)), number)
    # but not when taking more, or from another byte array, or at another index
    assert may_fail(intrinsic(AVMOp.extract3, data, index, uint64(9)), number)
    assert may_fail(number, intrinsic(AVMOp.extract_uint32, data, index))
    assert may_fail(intrinsic(AVMOp.extract3, other, index, uint64(8)), number)
    assert may_fail(intrinsic(AVMOp.extract3, data, size, uint64(8)), number)
    # a size that is not a constant counts for itself, and for what it is at least
    part = intrinsic(AVMOp.extract3, data, index, size)
    assert not may_fail(part, part)
    assert may_fail(number, part)
    assert not may_fail(intrinsic(AVMOp.substring3, data, uint64(0), index), part)


def test_bounds_given_conditions_do_not_reach_through_phi() -> None:
    # previous#1 = phi(0, current#0), where current#0 is read anew each time around the loop
    start = register("start")
    previous = register("previous")
    current = register("current")
    is_small = register("is_small", PrimitiveIRType.bool)
    entry = block(0, [assign(start, uint64(0))], return_())
    loop = block(
        1,
        [
            assign(current, intrinsic(AVMOp.btoi, bytes_(b"12345678"))),
            assign(is_small, intrinsic(AVMOp.lt, current, uint64(10))),
        ],
        return_(),
    )
    goto(entry, loop)
    goto(loop, loop)
    loop.phis.append(
        models.Phi(
            register=previous,
            args=[
                models.PhiArgument(value=start, through=entry),
                models.PhiArgument(value=current, through=loop),
            ],
        )
    )
    bounds = ValueBounds.of(subroutine("loop", [entry, loop]))
    given = bounds.given(non_zero=[is_small])
    assert given.uint64(current) == (0, 9)
    # the value from the last time around was not necessarily small
    assert given.uint64(previous) == ANY_UINT64


def test_typed_operations() -> None:
    slot = register("slot", SlotType(PrimitiveIRType.uint64))
    assert effects_of(models.ReadSlot(slot=slot, source_location=None)) == Effect.reads_scratch
    assert (
        effects_of(models.WriteSlot(slot=slot, value=_a, source_location=None))
        == Effect.writes_scratch
    )
    read = models.BoxRead(
        key=_key,
        value_type=PrimitiveIRType.bytes,
        exists_assertion_message="exists",
        source_location=None,
    )
    assert effects_of(read) == Effect.reads_box | Effect.reads_box_size | Effect.fails
    write = models.BoxWrite(key=_key, value=_data, delete_first=False, source_location=None)
    assert Effect.deletes_box not in effects_of(write)
    assert Effect.allocates_box in effects_of(write)
    delete_and_write = models.BoxWrite(
        key=_key, value=_data, delete_first=True, source_location=None
    )
    assert Effect.deletes_box in effects_of(delete_and_write)
    # an assert of a constant that holds cannot fail
    assert (
        effects_of(models.Assert(condition=_a, message=None, explicit=True, source_location=None))
        == Effect.fails
    )
    assert (
        effects_of(
            models.Assert(condition=uint64(1), message=None, explicit=True, source_location=None)
        )
        == Effect.none
    )
    # copies do nothing
    assert effects_of(assign(_a, _b)) == Effect.none
    # a call may do anything if nothing is known about its target
    callee = subroutine("callee", [block(0, [], return_())])
    assert effects_of(call(callee)) == EVERYTHING


def test_which_writes_change_which_reads() -> None:
    assert changed_by(Effect.reads_txn) == Effect.none
    assert changed_by(Effect.reads_scratch) == Effect.writes_scratch
    assert changed_by(Effect.reads_global) == Effect.writes_global
    # the size of a box is not changed by a write to its contents
    assert not changed_by(Effect.reads_box_size) & Effect.writes_box
    assert changed_by(Effect.reads_box) & Effect.writes_box
    # creating, resizing or deleting a box moves the minimum balance of the application account
    assert changed_by(Effect.reads_ledger) & Effect.allocates_box
    assert changed_by(Effect.reads_ledger) & Effect.deletes_box
    assert observers_of(Effect.allocates_box) == (
        Effect.reads_box | Effect.reads_box_size | Effect.reads_ledger
    )
    assert observers_of(Effect.writes_global) == Effect.reads_global
    # nothing an operation reads is lost when it is removed
    for read in (Effect.reads_scratch, Effect.reads_box, Effect.reads_ledger, Effect.reads_txn):
        assert not read & OBSERVABLE
    assert Effect.fails & OBSERVABLE
    assert Effect.exits & OBSERVABLE


def test_subroutine_summaries() -> None:
    param = parameter("param")
    result = register("result")
    # no effect, cannot fail
    harmless = subroutine(
        "harmless",
        [block(0, [assign(result, intrinsic(AVMOp.mod, param, uint64(7)))], return_(result))],
        parameters=[param],
        returns=[PrimitiveIRType.uint64],
    )
    # can fail
    checked = subroutine(
        "checked",
        [block(0, [assign(result, intrinsic(AVMOp.add, param, uint64(1)))], return_(result))],
        parameters=[param],
        returns=[PrimitiveIRType.uint64],
    )
    # writes a box, through another subroutine
    value = bytes_(b"value")
    writer = subroutine("writer", [block(0, [intrinsic(AVMOp.box_put, _key, value)], return_())])
    indirect_writer = subroutine("indirect_writer", [block(0, [call(writer)], return_())])
    # reads scratch space
    reader = subroutine(
        "reader",
        [
            block(
                0,
                [
                    assign(
                        result,
                        intrinsic(AVMOp.load, immediates=[3], types=[PrimitiveIRType.uint64]),
                    )
                ],
                return_(result),
            )
        ],
        returns=[PrimitiveIRType.uint64],
    )
    # ends the program
    exiting = subroutine(
        "exiting",
        [block(0, [], models.ProgramExit(result=uint64(1), source_location=None))],
    )
    # logs and then fails: nothing of the log is left
    failing = subroutine(
        "failing",
        [
            block(
                0,
                [intrinsic(AVMOp.log, value)],
                models.Fail(error_message=None, explicit=True, source_location=None),
            )
        ],
    )
    main = subroutine(
        "main",
        [
            block(
                0,
                [
                    assign(register("r1"), call(harmless, uint64(1))),
                    assign(register("r2"), call(checked, uint64(1))),
                    call(indirect_writer),
                    assign(register("r3"), call(reader)),
                    call(exiting),
                    call(failing),
                ],
                models.ProgramExit(result=uint64(1), source_location=None),
            )
        ],
        returns=[PrimitiveIRType.uint64],
    )
    summaries = analyse_subroutine_effects(
        program(main, harmless, checked, writer, indirect_writer, reader, exiting, failing)
    )
    assert summaries.of(harmless) == Effect.none
    assert summaries.of(checked) == Effect.fails
    box_put = effects_of(intrinsic(AVMOp.box_put, _key, value))
    assert summaries.of(writer) == box_put
    assert summaries.of(indirect_writer) == box_put
    assert summaries.of(reader) == Effect.reads_scratch
    assert summaries.of(exiting) == Effect.exits
    assert summaries.of(failing) == Effect.fails
    assert summaries.of(main) == (Effect.fails | box_put | Effect.reads_scratch | Effect.exits)
    # a subroutine that was not part of the program may do everything
    assert summaries.of(subroutine("unknown", [block(0, [], return_())])) == EVERYTHING


def test_summary_of_recursion_and_loops() -> None:
    # even and odd call each other, odd writes scratch space
    even = subroutine("even", [block(0, [], return_())])
    odd = subroutine(
        "odd",
        [block(0, [intrinsic(AVMOp.store, uint64(1), immediates=[1]), call(even)], return_())],
    )
    even.body[0].ops.append(call(odd))
    # calls itself
    recursive = subroutine("recursive", [block(0, [], return_())])
    recursive.body[0].ops.append(call(recursive))
    # loops, with no other effect
    entry = block(0, [], return_())
    body = block(1, [], return_())
    exit_ = block(2, [], return_())
    goto(entry, body)
    again = parameter("again")
    body.terminator = models.ConditionalBranch(
        condition=again, non_zero=body, zero=exit_, source_location=None
    )
    body.add_predecessor(body)
    exit_.add_predecessor(body)
    looping = subroutine("looping", [entry, body, exit_], parameters=[again])
    caller = subroutine("caller", [block(0, [call(even), call(looping, uint64(0))], return_())])
    main = subroutine(
        "main",
        [
            block(
                0,
                [call(caller), call(recursive)],
                models.ProgramExit(result=uint64(1), source_location=None),
            )
        ],
        returns=[PrimitiveIRType.uint64],
    )
    summaries = analyse_subroutine_effects(program(main, even, odd, recursive, looping, caller))
    # each of the two has the effects of both, and recursion that does not end fails
    assert summaries.of(even) == Effect.writes_scratch | Effect.fails
    assert summaries.of(odd) == Effect.writes_scratch | Effect.fails
    assert summaries.of(recursive) == Effect.fails
    # as does a loop that does not end
    assert summaries.of(looping) == Effect.fails
    assert summaries.of(caller) == Effect.writes_scratch | Effect.fails


def test_movable_intrinsics_cannot_fail() -> None:
    movable = [
        intrinsic(AVMOp.bzero, uint64(32)),
        intrinsic(AVMOp.itob, uint64(1)),
        intrinsic(AVMOp.txn, immediates=["Sender"]),
        intrinsic(AVMOp.global_, immediates=["CurrentApplicationAddress"]),
    ]
    not_movable = [
        intrinsic(AVMOp.bzero, uint64(4097)),
        intrinsic(AVMOp.bzero, _a),
        intrinsic(AVMOp.txn, immediates=["FirstValidTime"]),
        intrinsic(AVMOp.txn, immediates=["CreatedApplicationID"]),
        intrinsic(AVMOp.global_, immediates=["LatestTimestamp"]),
        intrinsic(AVMOp.global_, immediates=["OpcodeBudget"]),
    ]
    for op in movable:
        assert _is_unconditionally_movable(op), str(op)
        assert not effects_of(op) & OBSERVABLE, str(op)
    for op in not_movable:
        assert not _is_unconditionally_movable(op), str(op)
