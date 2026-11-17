"""Boxes and application state outlive a subroutine call, so the optimizer has to know who changes
them.

`analyse_state_effects` says, for every subroutine, which kinds of state its outcome depends on,
which kinds it can change, and whether it is repeatable: free of effects, with an outcome fixed by
its arguments and by the state it reads.

`repeated_state_reads_elimination` then follows what is known through a subroutine:

* a repeated call to a repeatable subroutine takes the earlier result, while nothing has changed
  the state the callee reads;
* with `SlotStorage` declared (see `puya.options`), a word read takes the word an earlier read or
  write of the same slot left, and survives writes to slots that are certainly other slots: a
  constant against another constant, one base at another distance, and whatever the declaration
  promises to be apart.

A read that is removed could not have failed: the same slot, or the same call, succeeded earlier
on every path, and nothing since could have taken its box away.

`page_reads_merge` runs once the other optimizations are done, for a declaration with
`pages_are_boxes`: word reads of one page share one read of the page box, and a word write of a
page whose box is known to hold the word replaces its bytes in place.

What is known at the start of a block is what every predecessor agrees on. A program that keeps
nothing in boxes, and declares nothing, is left as it is.
"""

import functools
import json
import os
import typing
from collections.abc import Iterator, Mapping, Sequence

import attrs

from puya import log
from puya.avm import AVMType
from puya.errors import InternalError
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import (
    IROptimizationContext,
    SlotWrite,
    StateEffects,
)
from puya.ir.optimize.dead_code_elimination import PURE_AVM_OPS
from puya.ir.types_ import IRType, PrimitiveIRType
from puya.options import SlotStorage
from puya.parse import SourceLocation
from puya.utils import not_none

logger = log.get_logger(__name__)

_WORD = 32
_SLOTS = 2**256
_SMALL = 2**64
_PAGE_WORDS = 64

# kinds of state
_BOX = "box"
_APP = "app"
_CHAIN = "chain"
_BALANCE = "balance"
_ALL = frozenset((_BOX, _APP, _CHAIN, _BALANCE))

_STATE_READS: typing.Final[Mapping[str, str]] = {
    AVMOp.box_get.code: _BOX,
    AVMOp.box_len.code: _BOX,
    AVMOp.box_extract.code: _BOX,
    AVMOp.app_global_get.code: _APP,
    AVMOp.app_global_get_ex.code: _APP,
    AVMOp.app_local_get.code: _APP,
    AVMOp.app_local_get_ex.code: _APP,
    # what only an inner transaction can change
    AVMOp.app_params_get.code: _CHAIN,
    AVMOp.asset_params_get.code: _CHAIN,
    AVMOp.asset_holding_get.code: _CHAIN,
    AVMOp.app_opted_in.code: _CHAIN,
    # a minimum balance moves with the boxes of the account
    AVMOp.balance.code: _BALANCE,
    AVMOp.min_balance.code: _BALANCE,
    AVMOp.acct_params_get.code: _BALANCE,
}
_STATE_WRITES: typing.Final[Mapping[str, frozenset[str]]] = {
    **dict.fromkeys(
        (
            AVMOp.box_put.code,
            AVMOp.box_replace.code,
            AVMOp.box_del.code,
            AVMOp.box_create.code,
            AVMOp.box_resize.code,
            AVMOp.box_splice.code,
            # a box of another application of the same family can be a box of this one
            AVMOp.app_box_put.code,
            AVMOp.app_box_replace.code,
            AVMOp.app_box_del.code,
            AVMOp.app_box_create.code,
            AVMOp.app_box_resize.code,
            AVMOp.app_box_splice.code,
        ),
        frozenset((_BOX, _BALANCE)),
    ),
    **dict.fromkeys(
        (
            AVMOp.app_global_put.code,
            AVMOp.app_global_del.code,
            AVMOp.app_local_put.code,
            AVMOp.app_local_del.code,
        ),
        frozenset((_APP,)),
    ),
    # an inner transaction can run any program
    AVMOp.itxn_submit.code: _ALL,
}
_HASHES = frozenset(
    (AVMOp.keccak256.code, AVMOp.sha256.code, AVMOp.sha512_256.code, AVMOp.sha3_256.code)
)
_PURE_SOURCES = (
    models.Value,
    models.ValueTuple,
    models.DecodeBytes,
    models.BytesEncode,
    models.ExtractValue,
    models.ReplaceValue,
    models.ArrayLength,
    models.ArrayPop,
    models.ArrayConcat,
)

_READ_ROLES = frozenset(("read_page_word", "read_word", "read_slot"))
_WRITE_ROLES = frozenset(("write_page_word", "write_words", "write_slot"))
_PAGE_ROLES = frozenset(("read_page_word", "write_page_word"))
_ANY_SLOT = SlotWrite()
# a subroutine nothing is known about
_UNKNOWN = StateEffects(writes=_ALL, repeatable=False)


def declared_storage(option: SlotStorage | None) -> SlotStorage | None:
    # PUYA_SLOT_STORAGE (the option as JSON) stands in for the option while a frontend cannot
    # write it yet
    return option or _storage_from_environment(os.environ.get("PUYA_SLOT_STORAGE"))


@functools.cache
def _storage_from_environment(declaration: str | None) -> SlotStorage | None:
    return SlotStorage(**json.loads(declaration)) if declaration else None


def storage_roles(storage: SlotStorage | None, program: models.Program) -> Mapping[str, str]:
    """subroutine id -> the declared role of that subroutine"""
    if storage is None:
        return {}
    roles = dict[str, str]()
    for role in (*_READ_ROLES, *_WRITE_ROLES, "slot_add"):
        name = getattr(storage, role)
        if name is not None:
            for sub in program.subroutines:
                # a program that holds several copies of a subroutine qualifies their ids
                if sub.id == name or sub.id.startswith(f"{name}."):
                    roles[sub.id] = role
    return roles


def analyse_state_effects(context: IROptimizationContext, program: models.Program) -> None:
    storage = declared_storage(context.options.slot_storage)
    roles = storage_roles(storage, program)
    local = dict[str, StateEffects]()
    callees = dict[str, set[str]]()
    for sub in program.all_subroutines:
        role = roles.get(sub.id)
        if role is None:
            local[sub.id], callees[sub.id] = _local_effects(sub, _Values.of(sub, roles))
        elif role in _READ_ROLES:
            local[sub.id] = StateEffects(reads=frozenset((_BOX,)))
        elif role in _WRITE_ROLES:
            local[sub.id] = StateEffects(slot_writes=frozenset((_ANY_SLOT,)), repeatable=False)
        else:
            local[sub.id] = StateEffects()
    effects = dict(local)
    changed = True
    while changed:
        changed = False
        for sub_id, called in callees.items():
            current = merged = effects[sub_id]
            for callee in called:
                other = effects.get(callee, _UNKNOWN)
                merged = StateEffects(
                    reads=merged.reads | other.reads,
                    writes=merged.writes | other.writes,
                    slot_writes=merged.slot_writes | other.slot_writes,
                    repeatable=merged.repeatable and other.repeatable,
                )
            if merged != current:
                effects[sub_id] = merged
                changed = True
    # a program that keeps nothing in boxes is left as it is
    boxes = storage is not None or any(
        _BOX in e.reads or _BOX in e.writes or e.slot_writes for e in effects.values()
    )
    context.state_effects = effects if boxes else None
    context.storage_roles = roles
    # the page subroutines of a program that keeps its words in the boxes of another
    # application are declared like any others, but those pages are not boxes of this program
    context.pages_are_boxes = (
        storage is not None
        and storage.pages_are_boxes
        and all(
            _uses_own_boxes(sub) for sub in program.subroutines if roles.get(sub.id) in _PAGE_ROLES
        )
    )


def _uses_own_boxes(sub: models.Subroutine) -> bool:
    """True if the subroutine calls no other and uses no box of another application."""
    for block in sub.body:
        for op in block.ops:
            node = op.source if isinstance(op, models.Assignment) else op
            if isinstance(node, models.InvokeSubroutine) or (
                isinstance(node, models.Intrinsic) and node.op.code.startswith("app_box_")
            ):
                return False
    return True


def _local_effects(sub: models.Subroutine, values: "_Values") -> tuple[StateEffects, set[str]]:
    reads = set[str]()
    writes = set[str]()
    slot_writes = set[SlotWrite]()
    repeatable = True
    callees = set[str]()
    for block in sub.body:
        if isinstance(block.terminator, models.Fail):
            continue  # nothing it does outlives the failure
        for op in block.ops:
            node: object = op.source if isinstance(op, models.Assignment) else op
            if isinstance(node, models.InvokeSubroutine):
                role = values.roles.get(node.target.id)
                if role in _WRITE_ROLES:
                    written = values.written(role, node)
                    slot_writes.add(_ANY_SLOT if written is None else values.kind(*written))
                    repeatable = False
                elif role in _READ_ROLES:
                    reads.add(_BOX)
                elif role is None:
                    callees.add(node.target.id)
            elif isinstance(node, models.Intrinsic):
                code = node.op.code
                if code in _STATE_WRITES:
                    writes |= _STATE_WRITES[code]
                    repeatable = False
                elif code in _STATE_READS:
                    reads.add(_STATE_READS[code])
                elif not _is_pure(node):
                    repeatable = False
            elif isinstance(node, models.BoxRead):
                reads.add(_BOX)
            elif isinstance(node, models.BoxWrite):
                writes |= _STATE_WRITES[AVMOp.box_put.code]
                repeatable = False
            elif not isinstance(node, (*_PURE_SOURCES, models.Assert)):
                repeatable = False
    return (
        StateEffects(
            reads=frozenset(reads),
            writes=frozenset(writes),
            slot_writes=frozenset(slot_writes),
            repeatable=repeatable,
        ),
        callees,
    )


def _is_pure(op: models.Intrinsic) -> bool:
    code = op.op.code
    if code in PURE_AVM_OPS:
        return True
    # the fields of `global` are fixed while a program runs, except for the budget left
    return code == AVMOp.global_.code and op.immediates != ["OpcodeBudget"]


@attrs.frozen
class _Slot:
    base: models.Value | None
    """None for a slot that is a constant number"""
    offset: int
    """modulo 2**256"""


@attrs.frozen
class _Word:
    value: models.Value
    form: str
    """how the 32 bytes of the word follow from the value: it is them ("word"), it is them once
    left-padded with zero bytes ("pad"), or they are its first 32 bytes ("head")"""


@attrs.frozen
class _Values:
    """The declared roles, and what the definitions in a subroutine say about a value."""

    roles: Mapping[str, str]
    definitions: Mapping[models.Register, models.ValueProvider]
    pages: frozenset[bytes]
    """the constant pages the subroutine reads or writes words of"""
    versions: dict[str, int]
    """the highest version assigned of each register name"""

    @classmethod
    def of(cls, subroutine: models.Subroutine, roles: Mapping[str, str]) -> "_Values":
        definitions = dict[models.Register, models.ValueProvider]()
        pages = set[bytes]()
        versions = dict[str, int]()
        for block in subroutine.body:
            for register in block.get_assigned_registers():
                versions[register.name] = max(versions.get(register.name, 0), register.version)
            for op in block.ops:
                if isinstance(op, models.Assignment) and len(op.targets) == 1:
                    definitions[op.targets[0]] = op.source
                node = op.source if isinstance(op, models.Assignment) else op
                match node:
                    case models.InvokeSubroutine(
                        target=target, args=[models.BytesConstant(value=page), *_]
                    ) if roles.get(target.id) in _PAGE_ROLES and len(page) == _WORD:
                        pages.add(page)
        return cls(roles=roles, definitions=definitions, pages=frozenset(pages), versions=versions)

    def fresh(
        self, name: str, ir_type: IRType, location: SourceLocation | None
    ) -> models.Register:
        """A register no assignment of the subroutine has as its target."""
        version = self.versions[name] = self.versions.get(name, -1) + 1
        return models.Register(
            name=name, version=version, ir_type=ir_type, source_location=location
        )

    def page_position(self, slot: _Slot) -> tuple[bytes, int] | None:
        """The page a constant slot is in, of those the subroutine names, and its byte
        position there."""
        if slot.base is None:
            for page in self.pages:
                distance = (slot.offset - int.from_bytes(page)) % _SLOTS
                if distance < _PAGE_WORDS:
                    return page, distance * _WORD
        return None

    def is_zero(self, value: models.Value) -> bool:
        """True if the value is certainly the number zero."""
        value = self.copied(value)
        number = _number(value)
        if number is not None:
            return number == 0
        match self.definition(value):
            case models.Intrinsic(op=AVMOp.itob, args=[models.UInt64Constant(value=n)]):
                return n == 0
            case models.Intrinsic(op=AVMOp.bzero):
                return True
        return False

    def is_nonzero(self, value: models.Value, *, depth: int = 4) -> bool:
        """True if the value certainly has a byte that is not zero."""
        value = self.copied(value)
        number = _number(value)
        if number is not None:
            return number != 0
        if not depth:
            return False
        match self.definition(value):
            case models.Intrinsic(op=AVMOp.itob, args=[models.UInt64Constant(value=n)]):
                return n != 0
            case models.Intrinsic(op=AVMOp.bitwise_or_bytes, args=[a, b]):
                return self.is_nonzero(a, depth=depth - 1) or self.is_nonzero(b, depth=depth - 1)
        return False

    def copied(self, value: models.Value) -> models.Value:
        """What a register is a copy of; definitions hold no phi, so the chain ends."""
        while isinstance(value, models.Register):
            source = self.definitions.get(value)
            if not isinstance(source, models.Value):
                break
            value = source
        return value

    def definition(self, value: models.Value) -> models.ValueProvider | None:
        value = self.copied(value)
        return self.definitions.get(value) if isinstance(value, models.Register) else None

    def slot(self, value: models.Value) -> _Slot | None:
        """The slot a value names, as a base and a distance from it; None if the value is not
        one that is the same wherever it is used."""
        offset = 0
        while True:
            value = self.copied(value)
            number = _number(value)
            if number is not None:
                return _Slot(base=None, offset=(number + offset) % _SLOTS)
            match self.definition(value):
                case models.Intrinsic(op=AVMOp.add_bytes, args=[a, b]):
                    a_number = _number(self.copied(a))
                    b_number = _number(self.copied(b))
                    if a_number is not None:
                        offset += a_number
                        value = b
                    elif b_number is not None:
                        offset += b_number
                        value = a
                    else:
                        break
                case models.InvokeSubroutine(
                    target=target, args=[base, models.UInt64Constant(value=distance)]
                ) if self.roles.get(target.id) == "slot_add":
                    offset += distance
                    value = base
                case _:
                    break
        if not isinstance(value, models.Register):
            return None
        return _Slot(base=value, offset=offset % _SLOTS)

    def accessed(self, role: str, call: models.InvokeSubroutine) -> _Slot | None:
        """The slot a declared read or write is of, None if it cannot be told."""
        if role not in _PAGE_ROLES:
            return self.slot(call.args[0])
        match call.args:
            case [models.BytesConstant(value=page), models.UInt64Constant(value=at), *_] if (
                len(page) == _WORD and at % _WORD == 0
            ):
                return _Slot(base=None, offset=(int.from_bytes(page) + at // _WORD) % _SLOTS)
        return None

    def written(self, role: str, call: models.InvokeSubroutine) -> tuple[_Slot, int] | None:
        """The first slot a declared write is to, and how many slots it may reach."""
        slot = self.accessed(role, call)
        if slot is None:
            return None
        if role == "write_words" and not self.is_word(call.args[1]):
            return slot, _PAGE_WORDS
        return slot, 1

    def kind(self, slot: _Slot, count: int) -> SlotWrite:
        """A write as other subroutines see it: without the values of this one."""
        if slot.base is None:
            return SlotWrite(constant=slot.offset, count=count)
        ending = self.hashed(slot.base)
        if ending is None or slot.offset + count > _SMALL:
            return _ANY_SLOT
        return SlotWrite(ending=ending)

    def hashed(self, base: models.Value) -> bytes | None:
        """For the result of a hash op, the constant ending of the bytes hashed."""
        match self.definition(base):
            case models.Intrinsic(op=op, args=[hashed]) if op.code in _HASHES:
                return self._ending(hashed, depth=8)[0]
        return None

    def _ending(self, value: models.Value, *, depth: int) -> tuple[bytes, bool]:
        """The constant bytes a value ends in, and whether they are the whole value."""
        value = self.copied(value)
        if isinstance(value, models.BytesConstant):
            return value.value, True
        match self.definition(value):
            case models.Intrinsic(op=AVMOp.concat, args=[a, b]) if depth:
                b_ending, b_whole = self._ending(b, depth=depth - 1)
                if not b_whole:
                    return b_ending, False
                a_ending, a_whole = self._ending(a, depth=depth - 1)
                return a_ending + b_ending, a_whole
        return b"", False

    def is_word(self, value: models.Value, *, depth: int = 8) -> bool:
        """True if the value is certainly 32 bytes long."""
        value = self.copied(value)
        if isinstance(value, models.BytesConstant):
            return len(value.value) == _WORD
        if not depth:
            return False
        match self.definition(value):
            case models.InvokeSubroutine(target=target):
                return self.roles.get(target.id) in (*_READ_ROLES, "slot_add")
            case models.Intrinsic(op=op) if op.code in _HASHES:
                return True
            case models.Intrinsic(op=AVMOp.bzero, args=[models.UInt64Constant(value=n)]):
                return n == _WORD
            case models.Intrinsic(op=AVMOp.extract3, args=[_, _, models.UInt64Constant(value=n)]):
                return n == _WORD
            case models.Intrinsic(op=AVMOp.extract, immediates=[_, int(n)]):
                return n == _WORD
            case models.Intrinsic(
                op=(AVMOp.replace2 | AVMOp.replace3 | AVMOp.setbyte | AVMOp.setbit),
                args=[target, *_],
            ):
                return self.is_word(target, depth=depth - 1)
            case models.Intrinsic(
                op=(AVMOp.bitwise_or_bytes | AVMOp.bitwise_and_bytes | AVMOp.bitwise_xor_bytes),
                args=[a, b],
            ):
                return self.is_word(a, depth=depth - 1) and self.is_word(b, depth=depth - 1)
        return False


def _number(value: models.Value) -> int | None:
    if isinstance(value, models.BigUIntConstant):
        return value.value
    if isinstance(value, models.BytesConstant):
        return int.from_bytes(value.value)
    return None


_CallKey = tuple[str, tuple[models.Value, ...]]


@attrs.frozen
class _Page:
    """The contents a page box had when `reader` read a word of it. They still are the words of
    the page, except for the ones written since."""

    reader: models.Register
    written: frozenset[int]
    """byte positions of the words written since"""


@attrs.define
class _Known:
    words: dict[_Slot, _Word] = attrs.field(factory=dict)
    calls: dict[_CallKey, tuple[models.Register, ...]] = attrs.field(factory=dict)
    pages: dict[bytes, _Page] = attrs.field(factory=dict)
    covered: dict[bytes, int] = attrs.field(factory=dict)
    """by page: a number of bytes its box is at least long"""

    @classmethod
    def agreed(cls, states: Sequence["_Known"]) -> "_Known":
        if not states:
            return cls()
        first, *rest = states
        covered = {
            key: min([length, *(o.covered[key] for o in rest)])
            for key, length in first.covered.items()
            if all(key in o.covered for o in rest)
        }
        pages = dict[bytes, _Page]()
        for key, page in first.pages.items():
            others = [o.pages.get(key) for o in rest]
            if all(o is not None and o.reader == page.reader for o in others):
                written = page.written.union(*(o.written for o in others if o is not None))
                pages[key] = _Page(reader=page.reader, written=written)
        return cls(
            words={k: v for k, v in first.words.items() if all(o.words.get(k) == v for o in rest)},
            calls={k: v for k, v in first.calls.items() if all(o.calls.get(k) == v for o in rest)},
            pages=pages,
            covered=covered,
        )


def repeated_state_reads_elimination(
    context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    return _eliminate(context, subroutine, pages=False)


def page_reads_merge(context: IROptimizationContext, subroutine: models.Subroutine) -> bool:
    """With `SlotStorage.pages_are_boxes` declared, word reads of one page take their words from
    one read of the page box: the first of them reads the box whole, and the others take their 32
    bytes from what it read, as long as nothing has written them since.

    The first read fails where it failed before, if its box is not available; the others could
    not fail before and cannot fail now.

    With `SlotStorage.inline_page_writes`, a word write of a page that was read whole also checks
    the length of what was read: where the box already held the word, the write replaces its 32
    bytes in place, which is all the write subroutine does in that case; otherwise it calls the
    subroutine as before.
    """
    if not context.pages_are_boxes:
        return False
    return _eliminate(context, subroutine, pages=True)


def _eliminate(
    context: IROptimizationContext, subroutine: models.Subroutine, *, pages: bool
) -> bool:
    effects = context.state_effects
    if effects is None:
        return False
    storage = declared_storage(context.options.slot_storage)
    if not _has_reads(subroutine, effects):
        return False
    follower = _Follower(
        effects=effects,
        values=_Values.of(subroutine, context.storage_roles),
        hashed_apart=storage is not None and storage.hashed_slots_apart,
        page_boxes=context.pages_are_boxes,
        pages=pages,
        inline_reads=pages and storage is not None and storage.inline_page_reads,
        inline_writes=pages and storage is not None and storage.inline_page_writes,
    )
    order = _reverse_post_order(subroutine.body[0])
    at_end = _known_at_block_ends(follower, order)

    def walk(mode: str) -> bool:
        follower.mode = mode
        modified = False
        for block in order:
            modified |= follower.follow(block, _known_at_start(block, order[0], at_end, follower))
        return modified

    if pages:
        walk("count")
        follower.choose_page_reads()
    modified = walk("rewrite")
    for write in follower.writes_in_place:
        _write_in_place(subroutine, write, follower.values)
        modified = True
    return modified


def _has_reads(subroutine: models.Subroutine, effects: Mapping[str, StateEffects]) -> bool:
    for block in subroutine.body:
        for op in block.ops:
            node = op.source if isinstance(op, models.Assignment) else op
            if (
                isinstance(node, models.InvokeSubroutine)
                and effects.get(node.target.id, _UNKNOWN).repeatable
            ):
                return True
    return False


def _known_at_block_ends(
    follower: "_Follower", order: Sequence[models.BasicBlock]
) -> dict[models.BasicBlock, _Known]:
    """What is known where each block ends, with nothing rewritten yet.

    A first walk takes a predecessor that was not walked yet (a back edge) to agree with the
    others; further walks drop what that predecessor turns out not to know, until nothing
    changes.
    """
    follower.mode = "follow"
    at_end = dict[models.BasicBlock, _Known]()
    for _ in range(_MAX_WALKS):
        changed = False
        for block in order:
            known = _known_at_start(block, order[0], at_end, follower)
            follower.follow(block, known)
            if at_end.get(block) != known:
                at_end[block] = known
                changed = True
        if not changed:
            return at_end
    # no agreement reached: a block with a back edge into it starts with nothing
    at_end.clear()
    for block in order:
        if all(p in at_end for p in block.predecessors):
            known = _known_at_start(block, order[0], at_end, follower)
        else:
            known = _Known()
        follower.follow(block, known)
        at_end[block] = known
    return at_end


def _known_at_start(
    block: models.BasicBlock,
    entry: models.BasicBlock,
    at_end: Mapping[models.BasicBlock, _Known],
    follower: "_Follower",
) -> _Known:
    """What the predecessors walked so far agree on; nothing where the subroutine is entered,
    even if that block is also the top of a loop."""
    if block is entry:
        return _Known()
    states = list[_Known]()
    for predecessor in block.predecessors:
        if predecessor in at_end:
            state = at_end[predecessor]
            # what the branch taken says about a page
            on_edge = follower.covered_on_edge.get((predecessor, block))
            if on_edge:
                covered = dict(state.covered)
                for key, length in on_edge.items():
                    covered[key] = max(covered.get(key, 0), length)
                state = attrs.evolve(state, covered=covered)
            states.append(state)
    return _Known.agreed(states)


_MAX_WALKS = 12


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


@attrs.define
class _Follower:
    effects: Mapping[str, StateEffects]
    values: _Values
    hashed_apart: bool
    page_boxes: bool
    """whether the words of a page are the bytes of its box"""
    pages: bool
    """whether word reads of one page are to share a read of the page box"""
    inline_reads: bool
    """whether a word read that shares with no other is to read the page box itself too"""
    inline_writes: bool
    """whether a word write of a page read whole is to replace its bytes in place"""
    mode: str = "follow"
    """"follow" only updates what is known; "count" also notes which page reads would share;
    "rewrite" replaces ops"""
    page_reads: dict[models.Register, list[int]] = attrs.field(factory=dict)
    """by the target of the first word read of a page: the byte positions of the words that
    it and the reads sharing with it take"""
    page_writes: set[models.Register] = attrs.field(factory=set)
    """the first word reads of a page that a later write of the page can use"""
    covered_on_edge: dict[tuple[models.BasicBlock, models.BasicBlock], dict[bytes, int]] = (
        attrs.field(factory=dict)
    )
    """by branch taken: lengths the page boxes are known to have at least, when it is taken"""
    writes_in_place: list["_WriteInPlace"] = attrs.field(factory=list)

    def follow(self, block: models.BasicBlock, known: _Known) -> bool:
        """Takes `known` from the start of the block to its end; returns whether it rewrote."""
        modified = False
        ops = list[models.Op]()
        for op in block.ops:
            replacement = self._op(op, known)
            if replacement is None or self.mode != "rewrite":
                ops.append(op)
            else:
                ops.extend(replacement)
                modified = True
        if modified:
            block.ops = ops
        if self.page_boxes:
            self._branch(block, known)
        return modified

    def _branch(self, block: models.BasicBlock, known: _Known) -> None:
        """Notes what each way out of a conditional branch says about the page boxes."""
        match block.terminator:
            case models.ConditionalBranch(condition=condition, non_zero=non_zero, zero=zero) if (
                non_zero is not zero
            ):
                for target, holds in ((non_zero, True), (zero, False)):
                    covered = dict[bytes, int]()
                    self._cover(condition, known, covered, holds=holds)
                    self.covered_on_edge[block, target] = covered

    def _cover(
        self,
        condition: models.Value,
        known: _Known,
        covered: dict[bytes, int],
        *,
        holds: bool,
        depth: int = 4,
    ) -> None:
        """Given that a condition holds, or does not: a word that is then not zero is within
        its page box, for a word past the end of the box is zero."""
        if not depth:
            return
        values = self.values
        match values.definition(condition):
            case models.Intrinsic(op=AVMOp.not_, args=[inner]):
                self._cover(inner, known, covered, holds=not holds, depth=depth - 1)
            case models.Intrinsic(
                op=(AVMOp.eq_bytes | AVMOp.eq | AVMOp.neq_bytes | AVMOp.neq) as op, args=[a, b]
            ):
                equal = holds == (op in (AVMOp.eq_bytes, AVMOp.eq))
                for one, other in ((a, b), (b, a)):
                    # equal to something that is not zero, or not the number zero
                    if self._nonzero(other) if equal else self._zero(other, op):
                        self._cover_word(one, known, covered)
            case models.Intrinsic(op=(AVMOp.gt_bytes | AVMOp.gt), args=[greater, _]) if holds:
                self._cover_word(greater, known, covered)
            case models.Intrinsic(op=(AVMOp.lt_bytes | AVMOp.lt), args=[_, greater]) if holds:
                self._cover_word(greater, known, covered)
            case models.Intrinsic(op=(AVMOp.lte_bytes | AVMOp.lte), args=[greater, _]) if (
                not holds
            ):
                self._cover_word(greater, known, covered)
            case models.Intrinsic(op=(AVMOp.gte_bytes | AVMOp.gte), args=[_, greater]) if (
                not holds
            ):
                self._cover_word(greater, known, covered)
            case _ if holds:
                # a number that is not zero, taken from a word
                self._cover_word(condition, known, covered)

    def _zero(self, value: models.Value, comparison: AVMOp) -> bool:
        if isinstance(value, models.UInt64Constant):
            return value.value == 0
        # bytes that differ from zero bytes can still be the number zero, at another length
        numbers = comparison in (AVMOp.eq_bytes, AVMOp.neq_bytes)
        return numbers and value.atype == AVMType.bytes and self.values.is_zero(value)

    def _nonzero(self, value: models.Value) -> bool:
        if isinstance(value, models.UInt64Constant):
            return value.value != 0
        return value.atype == AVMType.bytes and self.values.is_nonzero(value)

    def _cover_word(
        self, value: models.Value, known: _Known, covered: dict[bytes, int], *, depth: int = 4
    ) -> None:
        """A value that is not zero: if it is a word of a page, or bytes or a number taken from
        one, that word is not zero."""
        value = self.values.copied(value)
        for slot, word in known.words.items():
            if word.form != "head" and self.values.copied(word.value) == value:
                position = self.values.page_position(slot)
                if position is not None:
                    page, at = position
                    covered[page] = max(covered.get(page, 0), at + _WORD)
        if not depth:
            return
        match self.values.definition(value):
            case models.Intrinsic(
                op=(
                    AVMOp.extract
                    | AVMOp.extract3
                    | AVMOp.substring
                    | AVMOp.substring3
                    | AVMOp.extract_uint16
                    | AVMOp.extract_uint32
                    | AVMOp.extract_uint64
                    | AVMOp.getbyte
                    | AVMOp.getbit
                    | AVMOp.btoi
                    | AVMOp.bitlen
                ),
                args=[whole, *_],
            ):
                self._cover_word(whole, known, covered, depth=depth - 1)

    def choose_page_reads(self) -> None:
        self.page_reads = {
            reader: positions
            for reader, positions in self.page_reads.items()
            if len(positions) > 1 or self.inline_reads or reader in self.page_writes
        }

    def _op(self, op: models.Op, known: _Known) -> Sequence[models.Op] | None:
        """Updates `known` for an op; returns what the op becomes if it is not to stay."""
        node: object = op.source if isinstance(op, models.Assignment) else op
        if isinstance(node, models.InvokeSubroutine):
            targets = tuple(op.targets) if isinstance(op, models.Assignment) else ()
            role = self.values.roles.get(node.target.id)
            if role in _READ_ROLES and len(targets) == 1:
                slot = self.values.accessed(role, node)
                if slot is not None:
                    return self._word_read(op, node, targets[0], slot, role, known)
            elif role in _WRITE_ROLES:
                return self._word_write(role, node, known, statement=op is node)
            return self._call(op, node, targets, known)
        if isinstance(node, models.Intrinsic):
            changed = _STATE_WRITES.get(node.op.code)
            if changed is not None and not self._page_word_replaced(node, known):
                self._changed(changed, known)
        elif isinstance(node, models.Assert) and self.page_boxes:
            self._cover(node.condition, known, known.covered, holds=True)
        elif isinstance(node, models.BoxWrite):
            self._changed(_STATE_WRITES[AVMOp.box_put.code], known)
        return None

    def _page_word_replaced(self, op: models.Intrinsic, known: _Known) -> bool:
        """A `box_replace` of 32 bytes at a word of a page box is a write of that word."""
        if not self.page_boxes:
            return False
        match op:
            case models.Intrinsic(
                op=AVMOp.box_replace,
                args=[
                    models.BytesConstant(value=page),
                    models.UInt64Constant(value=at),
                    value,
                ],
            ) if len(page) == _WORD and at % _WORD == 0 and self.values.is_word(value):
                slot = _Slot(base=None, offset=(int.from_bytes(page) + at // _WORD) % _SLOTS)
                self._slot_written(slot, 1, _Word(value=value, form="word"), known)
                return True
        return False

    def _changed(self, kinds: frozenset[str], known: _Known) -> None:
        """Something changed state of these kinds, and not through a declared write."""
        if _BOX in kinds:
            known.words.clear()
            known.pages.clear()
            known.covered.clear()
        self._forget_calls(kinds, known)

    def _forget_calls(self, kinds: frozenset[str], known: _Known) -> None:
        for key in [k for k in known.calls if self.effects.get(k[0], _UNKNOWN).reads & kinds]:
            del known.calls[key]

    def _call(
        self,
        op: models.Op,
        call: models.InvokeSubroutine,
        targets: tuple[models.Register, ...],
        known: _Known,
    ) -> Sequence[models.Op] | None:
        effects = self.effects.get(call.target.id, _UNKNOWN)
        if effects.writes:
            self._changed(effects.writes, known)
        if effects.slot_writes:
            for slot in [
                s for s in known.words if not all(self._apart(s, w) for w in effects.slot_writes)
            ]:
                del known.words[slot]
            for write in effects.slot_writes:
                self._pages_written(write, known)
            self._forget_calls(_STATE_WRITES[AVMOp.box_put.code], known)
        if not effects.repeatable or any(isinstance(a, models.Undefined) for a in call.args):
            return None
        key = (call.target.id, tuple(call.args))
        earlier = known.calls.get(key)
        if earlier is None or (targets and not earlier):
            known.calls[key] = targets
            return None
        if not targets:
            # a call for its failure alone: the earlier identical call did not fail
            self._log(f"removing repeated call {call}", op)
            return ()
        if len(earlier) != len(targets) or any(
            target.ir_type.maybe_avm_type != value.ir_type.maybe_avm_type
            for target, value in zip(targets, earlier, strict=True)
        ):
            return None
        self._log(f"replacing repeated call {call}", op)
        return [
            models.Assignment(targets=[target], source=value, source_location=op.source_location)
            for target, value in zip(targets, earlier, strict=True)
        ]

    def _log(self, message: str, op: models.Op) -> None:
        if self.mode == "rewrite":
            logger.debug(message, location=op.source_location)

    def _word_read(
        self,
        op: models.Op,
        call: models.InvokeSubroutine,
        target: models.Register,
        slot: _Slot,
        role: str,
        known: _Known,
    ) -> Sequence[models.Op] | None:
        word = known.words.get(slot)
        if word is not None and word.value.ir_type.maybe_avm_type != AVMType.bytes:
            word = None
        if word is None:
            known.words[slot] = _Word(value=target, form="word")
            if self.pages and role in _PAGE_ROLES:
                return self._page_read(op, call, target, known)
            return None
        location = op.source_location
        self._log(f"replacing {op} with the word known for its slot", op)
        if word.form == "word":
            return [
                models.Assignment(targets=[target], source=word.value, source_location=location)
            ]
        known.words[slot] = _Word(value=target, form="word")
        if word.form == "head":
            first = models.Intrinsic(
                op=AVMOp.extract,
                immediates=[0, _WORD],
                args=[word.value],
                source_location=location,
            )
            return [models.Assignment(targets=[target], source=first, source_location=location)]
        zeros = _derived(target, "zeros", PrimitiveIRType.bytes)
        return [
            _assign(zeros, AVMOp.bzero, _uint64(_WORD, location)),
            _assign(target, AVMOp.bitwise_or_bytes, zeros, word.value),
        ]

    def _page_read(
        self,
        op: models.Op,
        call: models.InvokeSubroutine,
        target: models.Register,
        known: _Known,
    ) -> Sequence[models.Op] | None:
        """A word read of a page nothing is known of reads the page box whole, if later reads
        are to share it; a read that can share takes its word from there."""
        key, at = _page_and_position(call)
        location = op.source_location
        page = known.pages.get(key)
        if page is None:
            known.pages[key] = _Page(reader=target, written=frozenset())
            if self.mode == "count":
                self.page_reads[target] = [at]
            positions = self.page_reads.get(target)
            if self.mode != "rewrite" or positions is None:
                return None
            self._log(f"reading the page box whole for {op}", op)
            box = _derived(target, "box", PrimitiveIRType.bytes)
            exists = _derived(target, "exists", PrimitiveIRType.bool)
            zeros = _derived(target, "zeros", PrimitiveIRType.bytes)
            words = _derived(target, "page", PrimitiveIRType.bytes)
            return [
                models.Assignment(
                    targets=[box, exists],
                    source=models.Intrinsic(
                        op=AVMOp.box_get, args=[call.args[0]], source_location=location
                    ),
                    source_location=location,
                ),
                # a word past the end of the box is zero
                _assign(zeros, AVMOp.bzero, _uint64(max(positions) + _WORD, location)),
                _assign(words, AVMOp.concat, box, zeros),
                _assign(target, AVMOp.extract3, words, call.args[1], _uint64(_WORD, location)),
            ]
        if at in page.written:
            return None
        if self.mode == "count":
            self.page_reads[page.reader].append(at)
        if self.mode != "rewrite" or page.reader not in self.page_reads:
            return None
        self._log(f"taking the word of {op} from the page read earlier", op)
        words = _derived(page.reader, "page", PrimitiveIRType.bytes)
        return [_assign(target, AVMOp.extract3, words, call.args[1], _uint64(_WORD, location))]

    def _word_write(
        self, role: str, call: models.InvokeSubroutine, known: _Known, *, statement: bool
    ) -> Sequence[models.Op] | None:
        written = self.values.written(role, call)
        if written is None:
            self._forget_calls(_STATE_WRITES[AVMOp.box_put.code], known)
            known.words.clear()
            known.pages.clear()
            return None
        slot, count = written
        value = call.args[-1]
        if count > 1:
            form = "head"
        elif self.values.is_word(value):
            form = "word"
        elif role == "write_words":
            form = "head"
        else:
            form = "pad"
        replacement = None
        if self.pages and statement and role == "write_page_word":
            page, at = _page_and_position(call)
            if known.covered.get(page, 0) >= at + _WORD:
                replacement = self._replace_word(call)
            elif self.inline_writes:
                self._page_write(call, known)
        self._slot_written(slot, count, _Word(value=value, form=form), known)
        return replacement

    def _replace_word(self, call: models.InvokeSubroutine) -> Sequence[models.Op] | None:
        """A word write of a page whose box is known to hold the word replaces its bytes, which
        is all the write subroutine does then."""
        if self.mode != "rewrite":
            return None
        self._log(f"replacing the bytes of the word in its box for {call}", call)
        location = call.source_location
        page, at, value = call.args
        ops = list[models.Op]()
        if not self.values.is_word(value):
            zeros = self.values.fresh("page_word%zeros", PrimitiveIRType.bytes, location)
            word = self.values.fresh("page_word%padded", PrimitiveIRType.bytes, location)
            ops += [
                _assign(zeros, AVMOp.bzero, _uint64(_WORD, location)),
                _assign(word, AVMOp.bitwise_or_bytes, zeros, value),
            ]
            value = word
        ops.append(
            models.Intrinsic(
                op=AVMOp.box_replace, args=[page, at, value], source_location=location
            )
        )
        return ops

    def _page_write(self, call: models.InvokeSubroutine, known: _Known) -> None:
        """A word write of a page that was read whole can use the length of what was read."""
        key, _ = _page_and_position(call)
        page = known.pages.get(key)
        if page is None:
            return
        if self.mode == "count":
            self.page_writes.add(page.reader)
        elif self.mode == "rewrite" and page.reader in self.page_reads:
            self.writes_in_place.append(
                _WriteInPlace(call=call, reader=page.reader, number=len(self.writes_in_place))
            )

    def _slot_written(self, slot: _Slot, count: int, word: _Word, known: _Known) -> None:
        self._forget_calls(_STATE_WRITES[AVMOp.box_put.code], known)
        kind = self.values.kind(slot, count)
        for other in [
            s
            for s in known.words
            if not (
                (s.offset - slot.offset) % _SLOTS >= count
                if s.base == slot.base
                else self._apart(s, kind)
            )
        ]:
            del known.words[other]
        self._pages_written(kind, known)
        known.words[slot] = word
        if self.page_boxes and count == 1 and word.form != "head":
            # a word that is not zero is stored, so its box reaches it from then on
            position = self.values.page_position(slot)
            if position is not None and self.values.is_nonzero(word.value):
                page, at = position
                known.covered[page] = max(known.covered.get(page, 0), at + _WORD)

    def _pages_written(self, write: SlotWrite, known: _Known) -> None:
        for key, page in list(known.pages.items()):
            first = int.from_bytes(key)
            if write.constant is not None:
                positions = {
                    distance * _WORD
                    for n in range(write.count)
                    if (distance := (write.constant + n - first) % _SLOTS) < _PAGE_WORDS
                }
                if positions:
                    known.pages[key] = _Page(reader=page.reader, written=page.written | positions)
            elif not self._apart(_Slot(base=None, offset=first), write):
                del known.pages[key]

    def _apart(self, slot: _Slot, written: SlotWrite) -> bool:
        """True if the word of a slot is certainly not one such a write changes."""
        if slot.base is None and written.constant is not None:
            return (slot.offset - written.constant) % _SLOTS >= written.count
        if not self.hashed_apart:
            return False
        if written.constant is not None:
            return (
                written.constant + written.count <= _SMALL
                and slot.offset < _SMALL
                and self.values.hashed(slot.base) is not None
            )
        if written.ending is None:
            return False
        if slot.base is None:
            # a larger constant can be a hash worked out ahead
            return slot.offset < _SMALL
        ending = self.values.hashed(slot.base)
        if ending is None or slot.offset >= _SMALL:
            return False
        shared = min(len(ending), len(written.ending))
        return shared > 0 and ending[-shared:] != written.ending[-shared:]


def _page_and_position(call: models.InvokeSubroutine) -> tuple[bytes, int]:
    match call.args:
        case [models.BytesConstant(value=page), models.UInt64Constant(value=at), *_]:
            return page, at
    raise InternalError("expected a word read of a constant page", call.source_location)


def _derived(register: models.Register, name: str, ir_type: IRType) -> models.Register:
    return models.Register(
        name=f"{register.name}%{name}",
        version=register.version,
        ir_type=ir_type,
        source_location=register.source_location,
    )


def _uint64(value: int, location: SourceLocation | None) -> models.UInt64Constant:
    return models.UInt64Constant(value=value, source_location=location)


def _assign(target: models.Register, op: AVMOp, *args: models.Value) -> models.Assignment:
    location = target.source_location
    return models.Assignment(
        targets=[target],
        source=models.Intrinsic(op=op, args=list(args), source_location=location),
        source_location=location,
    )


@attrs.frozen
class _WriteInPlace:
    call: models.InvokeSubroutine
    """a word write of a page"""
    reader: models.Register
    """the target of the word read that read the page box whole"""
    number: int


def _write_in_place(subroutine: models.Subroutine, write: _WriteInPlace, values: _Values) -> None:
    """Splits the block of a page word write in two around it: where the page box read earlier
    is long enough to hold the word, its bytes are replaced in place; otherwise the write is the
    call it was."""
    call = write.call
    location = call.source_location
    page, at, value = call.args
    assert isinstance(at, models.UInt64Constant)
    block, index = next(
        (b, i) for b in subroutine.body for i, op in enumerate(b.ops) if op is call
    )
    next_id = max(not_none(b.id) for b in subroutine.body) + 1
    terminator = block.terminator
    assert terminator is not None
    after = models.BasicBlock(
        id=next_id + 2,
        ops=block.ops[index + 1 :],
        terminator=terminator,
        comment="after_page_write",
        source_location=block.source_location,
    )
    for successor in terminator.unique_targets:
        successor.replace_predecessor(old=block, new=after)
    del block.ops[index:]

    def named(name: str, ir_type: IRType) -> models.Register:
        return _derived(write.reader, f"{name}{write.number}", ir_type)

    in_place = list[models.Op]()
    if values.is_word(value):
        word = value
    else:
        zeros = named("zeros", PrimitiveIRType.bytes)
        word = named("word", PrimitiveIRType.bytes)
        in_place += [
            _assign(zeros, AVMOp.bzero, _uint64(_WORD, location)),
            _assign(word, AVMOp.bitwise_or_bytes, zeros, value),
        ]
    in_place.append(
        models.Intrinsic(op=AVMOp.box_replace, args=[page, at, word], source_location=location)
    )
    replace = models.BasicBlock(
        id=next_id,
        ops=in_place,
        terminator=models.Goto(target=after, source_location=location),
        comment="page_write_in_place",
        source_location=block.source_location,
    )
    grow = models.BasicBlock(
        id=next_id + 1,
        ops=[call],
        terminator=models.Goto(target=after, source_location=location),
        comment="page_write_call",
        source_location=block.source_location,
    )
    size = named("size", PrimitiveIRType.uint64)
    holds = named("holds", PrimitiveIRType.bool)
    block.ops += [
        _assign(size, AVMOp.len_, _derived(write.reader, "box", PrimitiveIRType.bytes)),
        _assign(holds, AVMOp.gte, size, _uint64(at.value + _WORD, location)),
    ]
    block.terminator = models.ConditionalBranch(
        condition=holds, non_zero=replace, zero=grow, source_location=location
    )
    for branch in (replace, grow):
        branch.add_predecessor(block)
        after.add_predecessor(branch)
    position = subroutine.body.index(block) + 1
    subroutine.body[position:position] = [replace, grow, after]
