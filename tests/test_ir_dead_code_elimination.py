from unittest.mock import Mock

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.dead_code_elimination import remove_unused_variables
from puya.ir.optimize.intrinsic_simplification import intrinsic_simplifier
from puya.ir.types_ import PrimitiveIRType
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


def _checked() -> models.Subroutine:
    param = parameter("param")
    result = register("result")
    return subroutine(
        "checked",
        [
            block(
                0,
                [assign(result, intrinsic(AVMOp.add, param, uint64(1)))],
                return_(result),
            )
        ],
        parameters=[param],
        returns=[PrimitiveIRType.uint64],
    )


def _exists_flag(name: str, key: models.Value = _key) -> tuple[models.Op, models.Register]:
    length = register(f"{name}_length")
    exists = register(f"{name}_exists", PrimitiveIRType.bool)
    read = models.Assignment(
        targets=[length, exists], source=intrinsic(AVMOp.box_len, key), source_location=None
    )
    return read, exists


def test_unused_operation_that_can_fail_is_kept() -> None:
    unused = [
        assign(register("sum"), intrinsic(AVMOp.add, _a, _b)),
        assign(register("difference"), intrinsic(AVMOp.sub, _a, _b)),
        assign(register("product"), intrinsic(AVMOp.mul, _a, _b)),
        assign(register("power"), intrinsic(AVMOp.exp, _a, _b)),
        assign(register("quotient"), intrinsic(AVMOp.div_floor, _a, _b)),
        assign(register("part", PrimitiveIRType.bytes), intrinsic(AVMOp.extract3, _data, _a, _b)),
        assign(register("double", PrimitiveIRType.bytes), intrinsic(AVMOp.concat, _data, _data)),
    ]
    remaining = optimize_block(remove_unused_variables, unused, parameters=_parameters)
    assert remaining == [str(op) for op in unused]


def test_unused_operation_is_removed_if_another_fails_whenever_it_does() -> None:
    word = assign(
        register("word", PrimitiveIRType.bytes),
        intrinsic(AVMOp.extract, _data, immediates=[24, 8]),
    )
    number = assign(register("number"), intrinsic(AVMOp.extract_uint64, _data, uint64(24)))
    byte = assign(register("byte"), intrinsic(AVMOp.getbyte, _data, uint64(31)))
    longer = assign(
        register("longer", PrimitiveIRType.bytes), intrinsic(AVMOp.concat, _data, bytes_(b"12"))
    )
    much_longer = assign(
        register("much_longer", PrimitiveIRType.bytes),
        intrinsic(AVMOp.concat, _data, bytes_(b"1234")),
    )
    use = [
        intrinsic(AVMOp.app_global_put, _key, number.targets[0]),
        intrinsic(AVMOp.app_global_put, _key, byte.targets[0]),
        intrinsic(AVMOp.log, much_longer.targets[0]),
    ]

    def optimized(*ops: models.Op) -> list[str]:
        return optimize_block(remove_unused_variables, [*ops, *use], parameters=_parameters)

    kept = [str(op) for op in (number, byte, much_longer, *use)]
    # an operation that follows it needs the same bytes, or more
    assert optimized(word, number, byte, longer, much_longer) == kept
    # as does an operation that ran before it
    assert optimized(number, byte, much_longer, word, longer) == kept
    # an unused operation that asks for more than any other stays
    wider = assign(
        register("wider", PrimitiveIRType.bytes),
        intrinsic(AVMOp.extract, _data, immediates=[24, 9]),
    )
    assert str(wider) in optimized(wider, number, byte, much_longer)


def test_unused_operation_is_removed_if_operation_on_what_it_is_computed_from_fails() -> None:
    # (a + 32) + 32 next to a + 64
    first = assign(register("first"), intrinsic(AVMOp.add, _a, uint64(32)))
    second = assign(register("second"), intrinsic(AVMOp.add, first.targets[0], uint64(32)))
    direct = assign(register("direct"), intrinsic(AVMOp.add, _a, uint64(64)))
    use = [
        intrinsic(AVMOp.app_global_put, _key, first.targets[0]),
        intrinsic(AVMOp.app_global_put, _key, direct.targets[0]),
    ]
    remaining = optimize_block(
        remove_unused_variables, [first, second, direct, *use], parameters=_parameters
    )
    assert remaining == [str(op) for op in (first, direct, *use)]
    # but not next to a + 63
    short = assign(register("short"), intrinsic(AVMOp.add, _a, uint64(63)))
    use[1] = intrinsic(AVMOp.app_global_put, _key, short.targets[0])
    ops = [first, second, short, *use]
    assert optimize_block(remove_unused_variables, ops, parameters=_parameters) == [
        str(op) for op in ops
    ]


def test_only_one_of_two_unused_operations_that_fail_together_is_removed() -> None:
    first = assign(register("first"), intrinsic(AVMOp.btoi, _data))
    second = assign(register("second"), intrinsic(AVMOp.btoi, _data))
    remaining = optimize_block(remove_unused_variables, [first, second], parameters=_parameters)
    assert remaining == [str(second)]


def test_unused_operation_is_kept_if_program_may_end_before_the_one_that_fails_with_it() -> None:
    exiting = subroutine(
        "exiting", [block(0, [], models.ProgramExit(result=uint64(1), source_location=None))]
    )
    word = assign(
        register("word", PrimitiveIRType.bytes),
        intrinsic(AVMOp.extract, _data, immediates=[24, 8]),
    )
    number = assign(register("number"), intrinsic(AVMOp.extract_uint64, _data, uint64(24)))
    use = intrinsic(AVMOp.app_global_put, _key, number.targets[0])
    ops = [word, call(exiting), number, use]
    remaining = optimize_block(remove_unused_variables, ops, exiting, parameters=_parameters)
    assert remaining == [str(op) for op in ops]


def test_unused_operation_that_cannot_fail_is_removed() -> None:
    small = register("small")
    word = register("word", PrimitiveIRType.bytes)
    unused = [
        assign(register("encoded", PrimitiveIRType.bytes), intrinsic(AVMOp.itob, _a)),
        assign(register("length"), intrinsic(AVMOp.len_, _data)),
        assign(register("hash", PrimitiveIRType.bytes), intrinsic(AVMOp.sha256, _data)),
        assign(
            register("sender", PrimitiveIRType.bytes), intrinsic(AVMOp.txn, immediates=["Sender"])
        ),
        assign(register("constant_sum"), intrinsic(AVMOp.add, uint64(1), uint64(2))),
        assign(register("halved"), intrinsic(AVMOp.div_floor, _a, uint64(2))),
        assign(
            register("state"),
            intrinsic(AVMOp.app_global_get, _key, types=[PrimitiveIRType.uint64]),
        ),
        assign(
            register("slot"), intrinsic(AVMOp.load, immediates=[1], types=[PrimitiveIRType.uint64])
        ),
    ]
    used = [
        assign(small, intrinsic(AVMOp.mod, _a, uint64(8))),
        assign(word, intrinsic(AVMOp.itob, _b)),
        # what is known about the operands rules a failure out
        assign(register("bounded_sum"), intrinsic(AVMOp.add, small, small)),
        assign(register("byte"), intrinsic(AVMOp.getbyte, word, small)),
        assign(register("decoded"), intrinsic(AVMOp.btoi, word)),
    ]
    remaining = optimize_block(remove_unused_variables, [*unused, *used], parameters=_parameters)
    # the operations that only fed removed operations are left for the next round
    assert remaining == [str(op) for op in used[:2]]


def test_unused_state_read_that_can_fail_is_kept() -> None:
    unused = [
        # a box has to be available to the transaction
        _exists_flag("box")[0],
        # as does an account
        assign(register("balance"), intrinsic(AVMOp.balance, _a)),
        # there may be no inner transaction to read from
        assign(
            register("created"),
            intrinsic(AVMOp.itxn, immediates=["CreatedApplicationID"]),
        ),
    ]
    remaining = optimize_block(remove_unused_variables, unused, parameters=_parameters)
    assert remaining == [str(op) for op in unused]


def test_unused_call_is_removed_if_target_does_nothing_else() -> None:
    harmless = _harmless()
    checked = _checked()
    # nothing but a loop, which may not end
    again = parameter("again")
    loop = block(0, [], return_())
    loop.terminator = models.ConditionalBranch(
        condition=again, non_zero=loop, zero=loop, source_location=None
    )
    loop.add_predecessor(loop)
    looping = subroutine("looping", [loop], parameters=[again], returns=[PrimitiveIRType.uint64])
    remaining = optimize_block(
        remove_unused_variables,
        [
            assign(register("r1"), call(harmless, _a)),
            assign(register("r2"), call(checked, _a)),
            assign(register("r3"), call(looping, _a)),
        ],
        harmless,
        checked,
        looping,
        parameters=_parameters,
    )
    assert remaining == [
        "let r2#0: uint64 = checked(a#0)",
        "let r3#0: uint64 = looping(a#0)",
    ]


def _is(name: str, op: AVMOp, a: models.Value, b: models.Value) -> models.Assignment:
    return assign(register(name, PrimitiveIRType.bool), intrinsic(op, a, b))


def test_unused_operation_is_removed_if_earlier_assert_rules_failure_out() -> None:
    length = assign(register("length"), intrinsic(AVMOp.len_, _data))
    is_word = _is("is_word", AVMOp.eq, length.targets[0], uint64(8))
    check = models.Assert(
        condition=is_word.targets[0], message="a word", explicit=True, source_location=None
    )
    decode = assign(register("number"), intrinsic(AVMOp.btoi, _data))

    before = [length, is_word, check, decode]
    assert optimize_block(remove_unused_variables, before, parameters=_parameters) == [
        str(op) for op in before[:3]
    ]
    # an assert that runs after the operation does not stop it from failing first
    after = [length, is_word, decode, check]
    assert optimize_block(remove_unused_variables, after, parameters=_parameters) == [
        str(op) for op in after
    ]


def test_unused_operation_is_removed_if_branch_taken_rules_failure_out() -> None:
    enough = _is("enough", AVMOp.gte, _a, _b)
    entry = block(0, [enough], return_())
    can_pay = block(1, [assign(register("left"), intrinsic(AVMOp.sub, _a, _b))], return_())
    cannot_pay = block(2, [assign(register("short"), intrinsic(AVMOp.sub, _a, _b))], return_())
    entry.terminator = models.ConditionalBranch(
        condition=enough.targets[0], non_zero=can_pay, zero=cannot_pay, source_location=None
    )
    can_pay.add_predecessor(entry)
    cannot_pay.add_predecessor(entry)
    sub = subroutine("branches", [entry, can_pay, cannot_pay], parameters=_parameters)
    remove_unused_variables(Mock(effects=None), sub)
    assert can_pay.ops == []
    assert [str(op) for op in cannot_pay.ops] == ["let short#0: uint64 = (- a#0 b#0)"]


def test_unused_operation_is_kept_if_condition_holds_on_one_way_in_only() -> None:
    enough = _is("enough", AVMOp.gte, _a, _b)
    entry = block(0, [enough], return_())
    checked = block(1, [], return_())
    join = block(2, [assign(register("left"), intrinsic(AVMOp.sub, _a, _b))], return_())
    entry.terminator = models.ConditionalBranch(
        condition=enough.targets[0], non_zero=checked, zero=join, source_location=None
    )
    checked.add_predecessor(entry)
    join.add_predecessor(entry)
    checked.terminator = models.Goto(target=join, source_location=None)
    join.add_predecessor(checked)
    sub = subroutine("join", [entry, checked, join], parameters=_parameters)
    remove_unused_variables(Mock(effects=None), sub)
    assert [str(op) for op in join.ops] == ["let left#0: uint64 = (- a#0 b#0)"]


def test_intrinsic_with_ignored_results_is_kept_if_it_can_fail() -> None:
    small = assign(register("small"), intrinsic(AVMOp.mod, _a, uint64(8)))
    kept = [
        intrinsic(AVMOp.add, _a, _b),
        intrinsic(AVMOp.btoi, _data),
        intrinsic(AVMOp.box_len, _key),
        intrinsic(AVMOp.log, _data),
        intrinsic(AVMOp.app_global_put, _key, small.targets[0]),
    ]
    removed = [
        intrinsic(AVMOp.itob, _a),
        intrinsic(AVMOp.add, small.targets[0], small.targets[0]),
        intrinsic(AVMOp.app_global_get, _key, types=[PrimitiveIRType.uint64]),
    ]
    main = subroutine(
        "main", [block(0, [small, *removed, *kept], return_())], parameters=_parameters
    )
    intrinsic_simplifier(Mock(effects=None, expand_all_bytes=False), main)
    assert [str(op) for op in main.body[0].ops] == [str(op) for op in (small, *kept)]
