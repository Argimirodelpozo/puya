from collections import Counter

import pytest

from puya.errors import InternalError
from puya.mir import models as mir
from puya.teal import models
from puya.teal.main import _ensure_explicit_checks_kept
from puya.teal.optimize.main import _replace_callsubs_with_branch

_CHECK = models.Assert(explicit=True, error_message="only the creator", source_location=None)
_KEY = ("caller() -> void:", _CHECK)


def _call(target: str) -> models.CallSub:
    return models.CallSub(target=target, consumes=0, produces=0, source_location=None)


def _subroutine(*ops: models.TealOp) -> models.TealSubroutine:
    block = models.TealBlock(
        label="caller", ops=list(ops), x_stack_in=(), entry_stack_height=0, exit_stack_height=0
    )
    return models.TealSubroutine(
        is_main=False,
        signature=mir.Signature(name="caller", parameters=(), returns=()),
        blocks=[block],
        source_location=None,
    )


def _op_codes(ops: list[models.TealOp]) -> list[str]:
    return [op.op_code for op in ops]


def test_ops_after_a_call_that_never_returns_are_discarded_and_reported() -> None:
    sub = _subroutine(
        _call("halt"),
        models.Int(1, source_location=None),
        _CHECK,
        models.RetSub(consumes=0, source_location=None),
    )
    discarded = _replace_callsubs_with_branch(sub, {"halt"})
    assert [op.teal() for op in sub.blocks[0].ops] == ["b halt"]
    assert _op_codes(discarded) == ["int", "assert", "retsub"]
    assert discarded[1] == _CHECK


def test_ops_before_such_a_call_and_after_other_calls_stay() -> None:
    sub = _subroutine(
        _call("helper"),
        models.Int(1, source_location=None),
        _CHECK,
        _call("halt"),
        models.RetSub(consumes=0, source_location=None),
    )
    discarded = _replace_callsubs_with_branch(sub, {"halt"})
    assert _op_codes(sub.blocks[0].ops) == ["callsub", "int", "assert", "b"]
    assert _op_codes(discarded) == ["retsub"]


def test_explicit_check_removed_as_unreachable_is_not_lost() -> None:
    _ensure_explicit_checks_kept(
        initial=Counter({_KEY: 1}), final=Counter(), unreachable=Counter({_KEY: 1})
    )


def test_explicit_check_duplicated_by_inlining_is_kept() -> None:
    _ensure_explicit_checks_kept(
        initial=Counter({_KEY: 1}), final=Counter({_KEY: 2}), unreachable=Counter()
    )


def test_explicit_check_removed_from_reachable_code_is_an_error() -> None:
    with pytest.raises(InternalError, match="explicit condition check"):
        _ensure_explicit_checks_kept(
            initial=Counter({_KEY: 1}), final=Counter(), unreachable=Counter()
        )
    # one of two was unreachable, the other is gone
    with pytest.raises(InternalError, match="explicit condition check"):
        _ensure_explicit_checks_kept(
            initial=Counter({_KEY: 2}), final=Counter(), unreachable=Counter({_KEY: 1})
        )


def test_unreachable_check_of_another_subroutine_does_not_count() -> None:
    other = ("other() -> void:", _CHECK)
    with pytest.raises(InternalError, match="explicit condition check"):
        _ensure_explicit_checks_kept(
            initial=Counter({_KEY: 1}), final=Counter(), unreachable=Counter({other: 1})
        )
