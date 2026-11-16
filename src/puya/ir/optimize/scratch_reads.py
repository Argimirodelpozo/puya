"""Scratch space is state that outlives a subroutine call, so the optimizer has to know who
writes it.

`analyse_scratch_effects` says, for every subroutine, whether it can write scratch space and
whether its result is a function of its arguments and of scratch space alone.

`repeated_scratch_reads_elimination` then follows what is known about scratch space through a
subroutine:

* a `load` after a `store` of a constant to the same slot takes the constant (any other value
  is cheaper to load again than to keep on the stack);
* a repeated call to a subroutine that only reads scratch space takes the earlier result, while
  nothing has written scratch space in between;
* with `LinearMemory` declared (see `puya.options`), a word read takes the value an earlier word
  write put at the same address, and survives writes to other words at a known distance.

What is known at the start of a block is what every predecessor agrees on; a block reached by a
back edge starts with nothing.
"""

import functools
import json
import os
import typing
from collections.abc import Iterator, Mapping, Sequence

import attrs

from puya import log
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext, ScratchEffects
from puya.ir.optimize.dead_code_elimination import PURE_AVM_OPS
from puya.options import LinearMemory

logger = log.get_logger(__name__)

_WORD = 32
_ZERO_WORD = b"\x00" * _WORD


def keep_declared_accessors(context: IROptimizationContext, program: models.Program) -> None:
    """A declared word accessor has to stay a call for its calls to be recognised."""
    # experiment switch: subroutines to inline at every call (ids, or prefixes ending in *)
    forced = [n for n in os.environ.get("PUYA_INLINE_SUBROUTINES", "").split(",") if n]
    for sub in program.subroutines:
        if sub.inline is None and any(
            sub.id.startswith(n[:-1]) if n.endswith("*") else sub.id == n for n in forced
        ):
            sub.inline = True
    declared = _declared_memory(context.options.linear_memory)
    if declared is None:
        return
    names = {declared.read_word, declared.read_word_uint64, declared.write_word}
    if declared.fresh_allocations:
        names.add(declared.alloc)
    for sub in program.subroutines:
        if sub.id in names and sub.inline is None:
            sub.inline = False


def analyse_scratch_effects(context: IROptimizationContext, program: models.Program) -> None:
    local = dict[str, ScratchEffects]()
    callees = dict[str, set[str]]()
    for sub in program.all_subroutines:
        local[sub.id], callees[sub.id] = _local_effects(sub)
    effects = dict(local)
    changed = True
    while changed:
        changed = False
        for sub_id, called in callees.items():
            current = effects[sub_id]
            writes = current.writes or any(effects[c].writes for c in called)
            repeatable = current.repeatable and all(effects[c].repeatable for c in called)
            if (writes, repeatable) != (current.writes, current.repeatable):
                effects[sub_id] = ScratchEffects(writes=writes, repeatable=repeatable)
                changed = True
    context.scratch_effects = effects
    context.program_subroutines = {sub.id: sub for sub in program.all_subroutines}


_PURE_SOURCES = (
    models.Value,
    models.ValueTuple,
    models.DecodeBytes,
    models.BytesEncode,
    models.ExtractValue,
    models.ReplaceValue,
    models.ArrayLength,
    models.ReadSlot,
)
_SCRATCH_READS = frozenset((AVMOp.load.code, AVMOp.loads.code))
_SCRATCH_WRITES = frozenset((AVMOp.store.code, AVMOp.stores.code))


def _local_effects(sub: models.Subroutine) -> tuple[ScratchEffects, set[str]]:
    writes = False
    repeatable = True
    callees = set[str]()
    for block in sub.body:
        for op in block.ops:
            node: object = op.source if isinstance(op, models.Assignment) else op
            if isinstance(node, models.InvokeSubroutine):
                callees.add(node.target.id)
            elif isinstance(node, models.Intrinsic):
                code = node.op.code
                if code in _SCRATCH_WRITES:
                    writes = True
                elif code not in PURE_AVM_OPS and code not in _SCRATCH_READS:
                    repeatable = False
            elif isinstance(node, models.WriteSlot | models.NewSlot):
                writes = True
            elif not isinstance(node, (*_PURE_SOURCES, models.Assert)):
                repeatable = False
    return ScratchEffects(writes=writes, repeatable=repeatable), callees


# what is known: a key says which read, the values are what that read gives
_Key = tuple[object, ...]
_Known = dict[_Key, tuple[models.Value, ...]]


def repeated_scratch_reads_elimination(
    context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    effects = context.scratch_effects
    if effects is None:
        return False
    definitions = dict[models.Register, models.ValueProvider]()
    for block in subroutine.body:
        for op in block.ops:
            if isinstance(op, models.Assignment) and len(op.targets) == 1:
                definitions[op.targets[0]] = op.source
    memory = _Memory(_declared_memory(context.options.linear_memory), definitions)
    known_at_end = dict[models.BasicBlock, _Known]()
    modified = False
    for block in _reverse_post_order(subroutine.entry):
        predecessors = block.predecessors
        if predecessors and all(p in known_at_end for p in predecessors):
            first, *rest = (known_at_end[p] for p in predecessors)
            known = {k: v for k, v in first.items() if all(other.get(k) == v for other in rest)}
        else:
            known = {}
        visitor = _BlockVisitor(block=block, known=known, effects=effects, memory=memory)
        modified |= visitor.run()
        known_at_end[block] = known
    return modified


def _declared_memory(option: LinearMemory | None) -> LinearMemory | None:
    # PUYA_LINEAR_MEMORY (the option as JSON) stands in for the option while a frontend cannot
    # write it yet
    return option or _memory_from_environment(os.environ.get("PUYA_LINEAR_MEMORY"))


@functools.cache
def _memory_from_environment(declaration: str | None) -> LinearMemory | None:
    return LinearMemory(**json.loads(declaration)) if declaration else None


def _reverse_post_order(entry: models.BasicBlock) -> Sequence[models.BasicBlock]:
    order = list[models.BasicBlock]()
    seen = {entry}
    stack: list[tuple[models.BasicBlock, Iterator[models.BasicBlock]]] = [
        (entry, iter(entry.successors))
    ]
    while stack:
        block, successors = stack[-1]
        for successor in successors:
            if successor not in seen:
                seen.add(successor)
                stack.append((successor, iter(successor.successors)))
                break
        else:
            order.append(block)
            stack.pop()
    order.reverse()
    return order


@attrs.frozen
class _Address:
    base: models.Value | None
    """None for an address that is a constant"""
    offset: int


@attrs.define
class _Memory:
    """The declared word accessors, and what the definitions in a subroutine say about a value."""

    declared: LinearMemory | None
    definitions: Mapping[models.Register, models.ValueProvider]

    def role(self, callee: models.Subroutine) -> str | None:
        declared = self.declared
        if declared is None:
            return None
        if callee.id == declared.read_word:
            return "read"
        if callee.id == declared.read_word_uint64:
            return "read64"
        if callee.id == declared.write_word:
            return "write"
        if callee.id == declared.write_range:
            return "write_range"
        if callee.id == declared.alloc:
            return "alloc"
        return None

    def address(self, value: models.Value) -> _Address:
        offset = 0
        for _ in range(64):
            if isinstance(value, models.UInt64Constant):
                return _Address(base=None, offset=offset + value.value)
            match self._definition(value):
                case models.Intrinsic(
                    op=AVMOp.add, args=[models.UInt64Constant(value=n), rest]
                ) | (models.Intrinsic(op=AVMOp.add, args=[rest, models.UInt64Constant(value=n)])):
                    offset += n
                    value = rest
                case models.Intrinsic(op=AVMOp.sub, args=[rest, models.UInt64Constant(value=n)]):
                    offset -= n
                    value = rest
                case _:
                    break
        return _Address(base=value, offset=offset)

    def is_word(self, value: models.Value) -> bool:
        """True if the value is what a word write stores for it: itself, unpadded."""
        if self._length(value, depth=8) == _WORD:
            return True
        match self._definition(value):
            case models.Intrinsic(op=AVMOp.bitwise_or_bytes, args=[a, b]):
                # or-ing with 32 zero bytes pads to a word, and padding twice changes nothing
                return self._is_zero_word(a) or self._is_zero_word(b)
        return False

    def word_as_uint64(self, value: models.Value) -> models.Value | None:
        """The uint64 a word holds, if the word was made from one."""
        if isinstance(value, models.BytesConstant):
            raw = value.value
            if len(raw) == _WORD and not any(raw[:24]):
                return models.UInt64Constant(
                    value=int.from_bytes(raw[24:]), source_location=value.source_location
                )
            return None
        match self._definition(value):
            case models.Intrinsic(op=AVMOp.concat, args=[zeros, low]) if (
                self._zero_length(zeros) == _WORD - 8
            ):
                return self._itob_argument(low)
            case models.Intrinsic(op=AVMOp.bitwise_or_bytes, args=[a, b]):
                if self._is_zero_word(a):
                    return self._itob_argument(b)
                if self._is_zero_word(b):
                    return self._itob_argument(a)
        return None

    def words(self, value: models.Value, *, depth: int = 64) -> list[models.Value] | None:
        """The words a value is a concatenation of, if it is one."""
        if isinstance(value, models.BytesConstant):
            raw = value.value
            if not raw or len(raw) % _WORD:
                return None
            return [
                models.BytesConstant(
                    value=raw[at : at + _WORD],
                    encoding=value.encoding,
                    source_location=value.source_location,
                )
                for at in range(0, len(raw), _WORD)
            ]
        if self.is_word(value):
            return [value]
        match self._definition(value):
            case models.Intrinsic(op=AVMOp.concat, args=[a, b]) if depth:
                first = self.words(a, depth=depth - 1)
                rest = self.words(b, depth=depth - 1) if first else None
                return first + rest if first and rest else None
        return None

    def _itob_argument(self, value: models.Value) -> models.Value | None:
        match self._definition(value):
            case models.Intrinsic(op=AVMOp.itob, args=[number]):
                return number
        return None

    def _definition(self, value: models.Value) -> models.ValueProvider | None:
        for _ in range(8):
            if not isinstance(value, models.Register):
                return None
            source = self.definitions.get(value)
            if not isinstance(source, models.Register):
                return source
            value = source
        return None

    def _is_zero_word(self, value: models.Value) -> bool:
        return self._zero_length(value) == _WORD

    def _zero_length(self, value: models.Value) -> int | None:
        if isinstance(value, models.BytesConstant):
            return len(value.value) if not any(value.value) else None
        match self._definition(value):
            case models.Intrinsic(op=AVMOp.bzero, args=[models.UInt64Constant(value=n)]):
                return n
        return None

    def _length(self, value: models.Value, *, depth: int) -> int | None:
        if isinstance(value, models.BytesConstant):
            return len(value.value)
        if depth == 0 or not isinstance(value, models.Register):
            return None
        sized = value.ir_type.num_bytes
        if sized is not None:
            return sized
        source = self.definitions.get(value)
        if isinstance(source, models.Value):
            return self._length(source, depth=depth - 1)
        if isinstance(source, models.InvokeSubroutine):
            return _WORD if self.role(source.target) == "read" else None
        if not isinstance(source, models.Intrinsic):
            return None
        match source:
            case models.Intrinsic(op=AVMOp.bzero, args=[models.UInt64Constant(value=n)]):
                return n
            case models.Intrinsic(op=AVMOp.itob):
                return 8
            case models.Intrinsic(op=AVMOp.concat, args=[a, b]):
                a_length = self._length(a, depth=depth - 1)
                b_length = self._length(b, depth=depth - 1)
                return None if a_length is None or b_length is None else a_length + b_length
            case models.Intrinsic(op=AVMOp.extract3, args=[_, _, models.UInt64Constant(value=n)]):
                return n
            case models.Intrinsic(op=AVMOp.extract, immediates=[_, int(n)]) if n:
                return n
            case models.Intrinsic(op=(AVMOp.replace2 | AVMOp.replace3), args=[target, *_]):
                return self._length(target, depth=depth - 1)
            case models.Intrinsic(
                op=(AVMOp.sha256 | AVMOp.keccak256 | AVMOp.sha512_256 | AVMOp.sha3_256)
            ):
                return 32
        return None


@attrs.define
class _BlockVisitor:
    block: models.BasicBlock
    known: _Known
    effects: Mapping[str, ScratchEffects]
    memory: _Memory
    modified: bool = False

    def run(self) -> bool:
        for op in self.block.ops.copy():
            if isinstance(op, models.Assignment):
                self._assignment(op)
            elif isinstance(op, models.InvokeSubroutine):
                self._call(op, op, targets=())
            elif isinstance(op, models.Intrinsic):
                self._statement(op)
            elif isinstance(op, models.WriteSlot):
                self.known.clear()
        return self.modified

    def _assignment(self, ass: models.Assignment) -> None:
        match ass.source:
            case models.InvokeSubroutine() as call:
                self._call(ass, call, targets=tuple(ass.targets))
            case (
                models.Intrinsic(op=AVMOp.load, immediates=[int(slot)])
                | models.Intrinsic(op=AVMOp.loads, args=[models.UInt64Constant(value=slot)])
            ):
                stored = self.known.get(("load", slot))
                if stored is not None:
                    self._replace(ass, stored)
            case models.NewSlot():
                self.known.clear()

    def _statement(self, op: models.Intrinsic) -> None:
        match op:
            case models.Intrinsic(op=AVMOp.store, immediates=[int(slot)], args=[value]):
                self._stored(slot, value)
            case models.Intrinsic(
                op=AVMOp.stores, args=[models.UInt64Constant(value=slot), value]
            ):
                self._stored(slot, value)
            case models.Intrinsic(op=AVMOp.stores):
                self.known.clear()

    def _stored(self, slot: int, value: models.Value) -> None:
        # a store through a computed slot number might have been to this slot, and a subroutine
        # or a word might have read it
        for key in [k for k in self.known if k[0] != "load" or k[1] == slot]:
            del self.known[key]
        if isinstance(value, models.Constant):
            self.known["load", slot] = (value,)

    def _call(
        self,
        op: models.Assignment | models.InvokeSubroutine,
        call: models.InvokeSubroutine,
        *,
        targets: tuple[models.Register, ...],
    ) -> None:
        role = self.memory.role(call.target)
        if role == "read" and len(targets) == 1:
            self._word_read(op, call, wide=True)
        elif role == "read64" and len(targets) == 1:
            self._word_read(op, call, wide=False)
        elif role == "write" and not targets:
            address_value, value = call.args
            self._words_written(self.memory.address(address_value), [value])
        elif role == "write_range" and not targets:
            address_value, value = call.args
            words = self.memory.words(value)
            if words is None:
                self.known.clear()
            else:
                self._words_written(self.memory.address(address_value), words)
        elif role == "alloc" and self.memory.declared and self.memory.declared.fresh_allocations:
            # it moves the free-memory pointer: not a word behind a computed address
            self._forget(lambda key: key[0] == "word" and key[1] is not None)
        else:
            effects = self.effects[call.target.id]
            if effects.writes:
                self.known.clear()
            elif effects.repeatable:
                self._read(op, ("call", call.target.id, *call.args), targets=targets)

    def _word_read(
        self,
        op: models.Assignment | models.InvokeSubroutine,
        call: models.InvokeSubroutine,
        *,
        wide: bool,
    ) -> None:
        assert isinstance(op, models.Assignment)
        address = self.memory.address(call.args[0])
        written = self.known.get(("word", address.base, address.offset, "written"))
        if written is not None:
            (value,) = written
            replacement = value if wide else self.memory.word_as_uint64(value)
            if replacement is not None:
                self._replace(op, (replacement,))
                return
        self._read(op, ("word", address.base, address.offset, "wide" if wide else "uint64"))

    def _words_written(self, address: _Address, words: Sequence[models.Value]) -> None:
        end = address.offset + _WORD * len(words)

        def survives(key: _Key) -> bool:
            if key[0] != "word":
                return False
            _, base, offset, _ = key
            assert isinstance(offset, int)
            return base == address.base and (offset + _WORD <= address.offset or offset >= end)

        self._forget(survives)
        for index, word in enumerate(words):
            if self.memory.is_word(word):
                self.known["word", address.base, address.offset + _WORD * index, "written"] = (
                    word,
                )

    def _forget(self, survives: typing.Callable[[_Key], bool]) -> None:
        for key in [k for k in self.known if not survives(k)]:
            del self.known[key]

    def _read(
        self,
        op: models.Assignment | models.InvokeSubroutine,
        key: _Key,
        *,
        targets: tuple[models.Register, ...] | None = None,
    ) -> None:
        if targets is None:
            assert isinstance(op, models.Assignment)
            targets = tuple(op.targets)
        earlier = self.known.get(key)
        if earlier is None or (targets and not earlier):
            self.known[key] = targets
        elif isinstance(op, models.InvokeSubroutine) or len(earlier) == len(targets):
            self._replace(op, earlier)

    def _replace(
        self, op: models.Assignment | models.InvokeSubroutine, values: tuple[models.Value, ...]
    ) -> None:
        index = self.block.ops.index(op)
        if isinstance(op, models.InvokeSubroutine):
            # a call for its failure alone: the earlier identical call did not fail
            logger.debug(f"removing repeated call {op}", location=op.source_location)
            del self.block.ops[index]
            self.modified = True
            return
        if any(
            target.ir_type.maybe_avm_type != value.ir_type.maybe_avm_type
            for target, value in zip(op.targets, values, strict=True)
        ):
            return
        logger.debug(f"replacing {op} with known values {values}", location=op.source_location)
        self.block.ops[index : index + 1] = [
            models.Assignment(targets=[target], source=value, source_location=op.source_location)
            for target, value in zip(op.targets, values, strict=True)
        ]
        self.modified = True
