from collections.abc import Sequence
from unittest.mock import Mock

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.main import SubroutineOptimization
from puya.ir.optimize.range_rewrites import range_constant_outcomes
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
    return optimize_block(range_constant_outcomes, ops, parameters=_PARAMETERS)


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
