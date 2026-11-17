import functools
import itertools
import typing
from collections.abc import Sequence

import attrs

from puya import log
from puya.teal import models
from puya.teal._util import preserve_stack_manipulations

TealOpSequence = tuple[models.TealOpUInt8, ...]
logger = log.get_logger(__name__)


class InvalidOpSequenceError(Exception):
    pass


@attrs.define
class TealStack:
    stack: list[int]

    @classmethod
    def from_stack_size(cls, stack_size: int) -> "TealStack":
        return cls(stack=list(range(stack_size)))

    def apply(self, ops: Sequence[models.TealOpUInt8]) -> "TealStack":
        stack = TealStack(self.stack.copy())
        for op in ops:
            n = op.n
            if n:
                index = len(stack.stack) - n - 1
                if index < 0 or index >= len(stack.stack):
                    raise InvalidOpSequenceError
                match op.op_code:
                    case "cover":
                        stack.stack.insert(index, stack.stack.pop())
                    case "uncover":
                        stack.stack.append(stack.stack.pop(index))
                    case _:
                        raise InvalidOpSequenceError
        return stack


@functools.cache
def simplify_rotation_ops(original_ops: TealOpSequence) -> TealOpSequence | None:
    num_rot_ops = len(original_ops)
    max_rot_op_n = 0
    for o in original_ops:
        max_rot_op_n = max(max_rot_op_n, o.n)

    original_stack = TealStack.from_stack_size(max_rot_op_n + 1)

    expected = original_stack.apply(original_ops)
    # entire sequence can be removed!
    if expected == original_stack:
        return ()

    possible_rotation_ops = get_possible_rotation_ops(max_rot_op_n)
    original_stack_result = original_stack.apply(original_ops)

    # TODO: use a non-bruteforce approach and/or capture common simplifications as data
    for num_rotation_ops in range(num_rot_ops):
        for maybe_ops in itertools.permutations(possible_rotation_ops, num_rotation_ops):
            try:
                stack = original_stack.apply(maybe_ops)
            except InvalidOpSequenceError:
                continue
            if expected == stack:
                assert original_stack_result == original_stack.apply(maybe_ops)
                return tuple(attrs.evolve(op, source_location=None) for op in maybe_ops)
    return None


@functools.cache
def get_possible_rotation_ops(n: int) -> TealOpSequence:
    possible_ops = list[models.TealOpUInt8]()
    for i in range(1, n + 1):
        possible_ops.append(models.Cover(i, source_location=None))
        possible_ops.append(models.Uncover(i, source_location=None))
    return tuple(possible_ops)


ROTATION_SIMPLIFY_OPS = frozenset(
    [
        "cover",
        "uncover",
    ]
)


def repeated_rotation_ops_search(block: models.TealBlock) -> None:
    ops = block.ops
    index = 0
    while index < len(ops):
        if ops[index].op_code not in ROTATION_SIMPLIFY_OPS:
            index += 1
            continue
        stop = index + 1
        while stop < len(ops) and ops[stop].op_code in ROTATION_SIMPLIFY_OPS:
            stop += 1
        index = stop - _simplify_run(ops, index, stop)


def _simplify_run(ops: list[models.TealOp], start: int, stop: int, window_size: int = 5) -> int:
    """Replaces the first window of a run of rotation ops that has a shorter equivalent,
    and returns by how many ops the run got shorter.

    The window is replaced in the ops of the block and not in a copy of the run: the stack
    manipulations of a window that cancels out go to the op before it, or to the op after
    it, and for a whole run that cancels out that op is outside the run.
    """
    for window_start in range(start, stop - 1):
        window = slice(window_start, min(window_start + window_size + 1, stop))
        simplified = simplify_rotation_ops(typing.cast(TealOpSequence, tuple(ops[window])))
        if simplified is None:
            continue
        if not simplified and window.start == 0 and window.stop == len(ops):
            # the block is nothing but this window: no op is left to take its stack
            # manipulations
            continue
        logger.debug(
            f"Replaced '{'; '.join(op.teal() for op in ops[start:stop])}'"
            f" with '{'; '.join(op.teal() for op in simplified)}',"
            f" reducing by {window.stop - window.start - len(simplified)} ops by search"
        )
        num_ops = len(ops)
        preserve_stack_manipulations(ops, window, simplified)
        return num_ops - len(ops)
    return 0
