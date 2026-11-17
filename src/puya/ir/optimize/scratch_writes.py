"""With `LinearMemory` declared (see `puya.options`), a run of word writes to neighbouring
addresses becomes one range write of the words joined together: one call, one page lookup.

A run is a sequence of word writes in one block, to distinct words at known distances from one
base, with nothing between them that reads or writes scratch space, calls a subroutine or asserts
a condition. The words keep their values; only the moment they reach memory moves, to the last
write of the run, and nothing in between can tell. One difference remains: a write to an address
outside memory fails at the end of the run instead of at its own position, so if a pure operation
in between also fails, that failure is the one reported.
"""

from collections.abc import Sequence

import attrs

from puya import log
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.dead_code_elimination import PURE_AVM_OPS
from puya.ir.optimize.scratch_reads import _WORD, _Address, _Memory, local_objects
from puya.ir.register_read_collector import RegisterReadCollector

logger = log.get_logger(__name__)

# two words joined and padded cost what the second call cost
_MIN_WORDS = 3
# instructions at a call site of a word write: the address, the value, the call
_CALL_COST = 5
# a range write takes one stack value
_MAX_WORDS = 4096 // _WORD
_PURE_SOURCES = (
    models.Value,
    models.ValueTuple,
    models.DecodeBytes,
    models.BytesEncode,
    models.ExtractValue,
    models.ReplaceValue,
    models.ArrayLength,
)


def merge_word_writes(context: IROptimizationContext, subroutine: models.Subroutine) -> bool:
    declared = context.linear_memory
    subroutines = context.program_subroutines
    if declared is None or subroutines is None or declared.write_range is None:
        return False
    write_range = subroutines.get(declared.write_range)
    if write_range is None or len(write_range.parameters) != 2:
        return False
    definitions = dict[models.Register, models.ValueProvider]()
    for block in subroutine.body:
        for op in block.ops:
            if isinstance(op, models.Assignment) and len(op.targets) == 1:
                definitions[op.targets[0]] = op.source
    merger = _Merger(
        context=context,
        memory=_Memory(declared, definitions),
        write_range=write_range,
        versions=_next_versions(subroutine),
    )
    modified = False
    for block in subroutine.body:
        modified |= merger.merge(block)
    return modified


def remove_unread_word_writes(
    context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    """A word of a local object (see `scratch_reads`) that no read in the subroutine can see does
    not need its writes, and an allocation whose pointer is not used does not need to happen.
    """
    declared = context.linear_memory
    if declared is None or not declared.fresh_allocations:
        return False
    memory = _Memory(declared, _definitions(subroutine))
    local = local_objects(subroutine, memory)
    if not local:
        return False
    read = dict[models.Value, list[int]]()
    for block in subroutine.body:
        for op in block.ops:
            node = op.source if isinstance(op, models.Assignment) else op
            if isinstance(node, models.InvokeSubroutine) and memory.role(node.target) in (
                "read",
                "read64",
            ):
                address = memory.address(node.args[0])
                read.setdefault(address.base, []).append(address.offset)
    modified = False
    for block in subroutine.body:
        for op in block.ops.copy():
            if not isinstance(op, models.InvokeSubroutine) or len(op.args) != 2:
                continue
            role = memory.role(op.target)
            length: int | None
            if role == "write":
                length = _WORD
            elif role == "write_range":
                words = memory.words(op.args[1])
                length = None if words is None else _WORD * len(words)  # None: to the end
            else:
                continue
            address = memory.address(op.args[0])
            if address.base in local and not any(
                (length is None or at < address.offset + length) and address.offset < at + _WORD
                for at in read.get(address.base, ())
            ):
                logger.debug(f"removing unread write {op}", location=op.source_location)
                block.ops.remove(op)
                modified = True
    if declared.removable_allocations:
        collector = RegisterReadCollector()
        collector.visit_all_blocks(subroutine.body)
        used = set(collector.used_registers)
        for block in subroutine.body:
            for op in block.ops.copy():
                if (
                    isinstance(op, models.Assignment)
                    and isinstance(op.source, models.InvokeSubroutine)
                    and memory.role(op.source.target) == "alloc"
                    and not any(target in used for target in op.targets)
                ):
                    logger.debug(f"removing unused allocation {op}", location=op.source_location)
                    block.ops.remove(op)
                    modified = True
    return modified


def _definitions(subroutine: models.Subroutine) -> dict[models.Register, models.ValueProvider]:
    definitions = dict[models.Register, models.ValueProvider]()
    for block in subroutine.body:
        for op in block.ops:
            if isinstance(op, models.Assignment) and len(op.targets) == 1:
                definitions[op.targets[0]] = op.source
    return definitions


def _next_versions(subroutine: models.Subroutine) -> dict[str, int]:
    versions = dict[str, int]()
    for register in subroutine.get_assigned_registers():
        if register.name.startswith(_NAME_PREFIX):
            versions[register.name] = max(versions.get(register.name, 0), register.version + 1)
    return versions


_NAME_PREFIX = f"merged{models.TMP_VAR_INDICATOR}"


@attrs.frozen
class _Write:
    call: models.InvokeSubroutine
    address: _Address


@attrs.define
class _Merger:
    context: IROptimizationContext
    memory: _Memory
    write_range: models.Subroutine
    versions: dict[str, int]
    _shared: bool | None = None

    def _range_writer_is_shared(self) -> bool:
        """A merge adds a call to the range writer, which then has to stay a subroutine. If nothing
        calls it (unused, or already inlined at its one caller), the merge would pay for a copy of
        its body.
        """
        if self._shared is None:
            subroutines = self.context.program_subroutines or {}
            self._shared = any(
                _called(op) is self.write_range
                for sub in subroutines.values()
                for block in sub.body
                for op in block.ops
            )
            if self._shared:
                # from here on its calls stay calls
                self.write_range.inline = False
                self.context.inlineable_calls -= {
                    pair
                    for pair in self.context.inlineable_calls
                    if pair[1] == self.write_range.id
                }
        return self._shared

    def merge(self, block: models.BasicBlock) -> bool:
        runs = list[list[_Write]]()
        run = list[_Write]()
        for op in block.ops:
            if (
                isinstance(op, models.InvokeSubroutine)
                and self.memory.role(op.target) == "write"
                and len(op.args) == 2
            ):
                address = self.memory.address(op.args[0])
                if run and (
                    address.base != run[0].address.base
                    or any(abs(address.offset - w.address.offset) < _WORD for w in run)
                ):
                    runs.append(run)
                    run = []
                run.append(_Write(call=op, address=address))
            elif not _transparent(op):
                runs.append(run)
                run = []
        runs.append(run)
        modified = False
        for run in runs:
            run.sort(key=lambda w: w.address.offset)
            group = run[:1]
            for write in run[1:]:
                if (
                    write.address.offset == group[-1].address.offset + _WORD
                    and len(group) < _MAX_WORDS
                ):
                    group.append(write)
                else:
                    modified |= self._emit(block, group)
                    group = [write]
            modified |= self._emit(block, group)
        return modified

    def _emit(self, block: models.BasicBlock, group: Sequence[_Write]) -> bool:
        if len(group) < _MIN_WORDS:
            return False
        calls = [w.call for w in group]
        padded = sum(not self.memory.is_word(call.args[1]) for call in calls)
        if _joined_cost(words=len(calls), padded=padded) >= _CALL_COST * len(calls):
            return False
        if not self._range_writer_is_shared():
            return False
        last = max(calls, key=block.ops.index)
        location = last.source_location
        new_ops = list[models.Op]()

        def assign(name: str, source: models.Intrinsic) -> models.Register:
            name = _NAME_PREFIX + name
            version = self.versions.get(name, 0)
            self.versions[name] = version + 1
            (ir_type,) = source.types
            register = models.Register(
                name=name, version=version, ir_type=ir_type, source_location=location
            )
            new_ops.append(
                models.Assignment(targets=[register], source=source, source_location=location)
            )
            return register

        zero_word: models.Register | None = None
        words = list[models.Value]()
        for call in calls:
            value = call.args[1]
            if not self.memory.is_word(value):
                # what a word write does to its value before it stores it
                if zero_word is None:
                    zero_word = assign(
                        "zero",
                        models.Intrinsic(
                            op=AVMOp.bzero,
                            args=[models.UInt64Constant(value=_WORD, source_location=location)],
                            source_location=location,
                        ),
                    )
                value = assign(
                    "word",
                    models.Intrinsic(
                        op=AVMOp.bitwise_or_bytes,
                        args=[zero_word, value],
                        source_location=location,
                    ),
                )
            words.append(value)
        joined = words[0]
        for word in words[1:]:
            joined = assign(
                "words",
                models.Intrinsic(op=AVMOp.concat, args=[joined, word], source_location=location),
            )
        new_ops.append(
            models.InvokeSubroutine(
                target=self.write_range, args=[calls[0].args[0], joined], source_location=location
            )
        )
        logger.debug(f"merging {len(calls)} word writes into one range write", location=location)
        at = block.ops.index(last)
        block.ops[at : at + 1] = new_ops
        for call in calls:
            if call is not last:
                block.ops.remove(call)
        return True


def _called(op: models.Op) -> models.Subroutine | None:
    node = op.source if isinstance(op, models.Assignment) else op
    return node.target if isinstance(node, models.InvokeSubroutine) else None


def _joined_cost(*, words: int, padded: int) -> int:
    """Instructions that replace the call sites of a run: one call, the joins, and for every value
    not known to be a word its padding, with the zero word made once."""
    return _CALL_COST + (words - 1) + (words - padded) + 3 * padded + (2 if padded else 0)


def _transparent(op: models.Op) -> bool:
    """True for an assignment that neither reads nor writes scratch space and calls nothing."""
    if not isinstance(op, models.Assignment):
        return False
    source = op.source
    if isinstance(source, models.Intrinsic):
        return source.op.code in PURE_AVM_OPS
    return isinstance(source, _PURE_SOURCES)
