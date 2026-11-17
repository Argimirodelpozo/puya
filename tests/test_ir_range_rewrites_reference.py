"""Checks the rewrites that value ranges allow against runs of made-up programs.

Each program is run on random inputs as it is and as the rewrites leave it, with a reference
implementation of the AVM ops: both have to fail for the same inputs, and to return the same
values otherwise.
"""

import copy
import random
from collections import Counter
from unittest.mock import Mock

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.range_rewrites import (
    range_cheaper_forms,
    range_constant_outcomes,
    range_operand_results,
)
from tests.utils.ir_programs import Program, describe, run

_REWRITES = (range_constant_outcomes, range_operand_results, range_cheaper_forms)


def _ops(subroutine: models.Subroutine) -> Counter[str]:
    """How many there are of each AVM op, of asserts and of branches on a register."""
    ops = Counter[str]()
    for block in subroutine.body:
        for op in block.ops:
            match op:
                case models.Assert():
                    ops["assert"] += 1
                case models.Assignment(source=models.Intrinsic(op=avm_op)):
                    ops[avm_op.code] += 1
        match block.terminator:
            case models.ConditionalBranch(condition=models.Register()):
                ops["branch"] += 1
    return ops


def check_programs(
    rng: random.Random, programs: int, runs: int
) -> tuple[Counter[str], Counter[str]]:
    """Makes up programs and runs them with and without the rewrites. Gives how the runs
    ended, and how many ops of each kind the rewrites took away."""
    outcomes = Counter[str]()
    removed = Counter[str]()
    for _ in range(programs):
        program = Program(
            rng, num_variables=rng.choice([2, 3]), num_statements=rng.randrange(6, 18)
        )
        original = program.subroutine
        rewritten = copy.deepcopy(original)
        context = Mock()
        changed = True
        while changed:
            changed = False
            for rewrite in _REWRITES:
                changed = rewrite(context, rewritten) or changed
        rewritten.validate_with_ssa()
        removed += _ops(original) - _ops(rewritten)
        for _ in range(runs):
            inputs = program.random_inputs()
            expected = run(original, inputs)
            if expected == "runs on":
                # the rewritten program may get further with the same number of ops
                continue
            outcomes[expected if isinstance(expected, str) else "completes"] += 1
            actual = run(rewritten, inputs, max_ops=10_000)
            assert actual == expected, (
                f"inputs: {inputs}\noriginal:\n{describe(original)}\n"
                f"rewritten:\n{describe(rewritten)}"
            )
    return outcomes, removed


def test_range_rewrites_agree_with_reference() -> None:
    outcomes, removed = check_programs(random.Random("range rewrites"), programs=100, runs=6)
    # programs failed and programs completed
    assert outcomes["fails"]
    assert outcomes["completes"]
    # each group of rewrites took something away: asserts and decided branches, operations
    # that gave back an operand, and byte math that had a cheaper form
    assert removed["assert"]
    assert removed["branch"]
    assert removed[AVMOp.mod_bytes.code]
    assert removed[AVMOp.bitwise_or_bytes.code]
    assert removed[AVMOp.lt_bytes.code]
    assert removed[AVMOp.btoi.code]
