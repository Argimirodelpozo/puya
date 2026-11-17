"""Boxes and application state outlive a subroutine call, so the optimizer has to know who changes
them. What a subroutine may read and change, and whether it can be called again for the same
outcome, comes from the effect model (`puya.ir.optimize.effects`): its summaries say what a call
may do, and `changed_by` says which of those effects change what a subroutine reads.

`repeated_state_reads_elimination` follows what is known through a subroutine:

* a repeated call to a subroutine that can be repeated takes the earlier result, while nothing
  has changed the state the callee reads;
* with `SlotStorage` declared (see `puya.options`), a word read takes the word an earlier read or
  write of the same slot left, and survives writes to slots that are certainly other slots: a
  constant against another constant, one base at another distance, and whatever the declaration
  promises to be apart. Which words a call may write is in the summary of its target.

A read that is removed could not have failed: the same slot, or the same call, succeeded earlier
on every path, and nothing since could have taken its box away.

`replaced_word_writes_elimination` removes a word write that another write to the same slot
replaces before anything can tell that it happened, where the later write fails whenever the
removed one would have and leaves the same boxes behind. Together with the above, a chain that
reads a word, changes it and writes it back several times is left with one read and one write.

`page_reads_merge` runs once the other optimizations are done, for a declaration with
`pages_are_boxes`: word reads of one page share one read of the page box, and a word write of a
page whose box is known to hold the word replaces its bytes in place.

What is known at the start of a block is what every predecessor agrees on. A program that keeps
nothing in boxes, and declares nothing, is left as it is.
"""

import typing
from collections.abc import Callable, Iterator, Mapping, Sequence

import attrs

from puya import log
from puya.avm import AVMType
from puya.errors import InternalError
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import (
    BOX_CHANGES,
    OBSERVABLE,
    READS,
    Effect,
    EffectSummaries,
    changed_by,
    effects_of,
    is_repeatable,
    observers_of,
)
from puya.ir.optimize.slot_storage import (
    PAGE_ROLES,
    PAGE_WORDS,
    READ_ROLES,
    SLOTS,
    SMALL,
    WORD,
    WRITE_ROLES,
    Slot,
    SlotValues,
    SlotWrite,
)
from puya.ir.types_ import IRType, PrimitiveIRType
from puya.parse import SourceLocation
from puya.utils import not_none

logger = log.get_logger(__name__)

# the effects that can make a box shorter, or take it away
_CHANGES_BOX_SIZE: typing.Final = Effect.allocates_box | Effect.deletes_box
# the effects that change something a subroutine may read
_CHANGES_STATE: typing.Final = changed_by(READS & ~Effect.reads_budget)
# the effects of an operation that can tell whether a box has changed: it reads a box, or a
# balance, where the minimum balance of the application account follows the sizes of its boxes
_TELLS_BOX_CHANGES: typing.Final = observers_of(BOX_CHANGES)
# all that a subroutine may do if a call to it is to be removed as a write that is replaced
_ONLY_WRITES_BOXES: typing.Final = ~OBSERVABLE | Effect.fails | BOX_CHANGES


@attrs.frozen
class _Word:
    value: models.Value
    form: str
    """how the 32 bytes of the word follow from the value: it is them ("word"), it is them once
    left-padded with zero bytes ("pad"), or they are its first 32 bytes ("head")"""


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
    words: dict[Slot, _Word] = attrs.field(factory=dict)
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
    return _eliminate(context, subroutine, pages=True)


# a program in which no subroutine has one of these keeps nothing in boxes
_USES_BOXES: typing.Final = Effect.reads_box | Effect.reads_box_size | BOX_CHANGES


def _eliminate(
    context: IROptimizationContext, subroutine: models.Subroutine, *, pages: bool
) -> bool:
    summaries = context.effects
    if summaries is None:
        return False
    storage = summaries.storage
    if storage is None:
        if pages or not summaries.anywhere & _USES_BOXES:
            return False
    elif pages and not storage.pages_are_boxes:
        return False
    if not _has_reads(subroutine, summaries):
        return False
    follower = _Follower.of(summaries, subroutine, pages=pages)
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


def _has_reads(subroutine: models.Subroutine, summaries: EffectSummaries) -> bool:
    for block in subroutine.body:
        for op in block.ops:
            node = op.source if isinstance(op, models.Assignment) else op
            if isinstance(node, models.InvokeSubroutine) and is_repeatable(
                summaries.of(node.target)
            ):
                return True
    return False


def replaced_word_writes_elimination(
    context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    """With `SlotStorage` declared, removes a word write that is replaced before anything can
    tell that it happened: on every way on from it, either the program fails, which undoes all
    it did, or a write through the same subroutine to the same slot comes before any of

    * a read of that word, or of a word that may be the same one;
    * any other operation that reads a box, or a balance: the minimum balance of the
      application account follows the sizes of its boxes;
    * an inner transaction, in which another program may read either;
    * a call to a subroutine that may do one of these, or that may end the program;
    * the end of the subroutine, or of the program.

    Other operations cannot tell: computing, asserting, logging, scratch space and application
    state do not depend on a box, and a write to another word does not read this one.

    The later write must also fail whenever the removed one would have, and leave the boxes
    as both would have. That is proven for a word of a page, with `pages_are_boxes`, where what
    a write does is declared, in three cases:

    * the box is known to hold the word when the removed write runs, so that it only replaces
      32 bytes, which the later write then does;
    * the removed write is of a word that is zero, which changes nothing where the box does
      not hold the word;
    * the later write is of a word that is not zero, so that it makes the box as long as the
      removed write would have.

    In each of them the later write is the first to count the box against what the
    transaction may write where the removed one would have been, and it counts no less: no
    box gets shorter in between, since anything that can delete or resize a box reads a box,
    and a write to another kind of slot in between keeps the write.

    Where that is not proven, the write is removed only if the declaration promises that the
    program does not depend on it (`overwritten_writes_removable`). For a word of a page,
    the promise is not enough if a `box_replace` can be reached from the write: an earlier
    run of `page_reads_merge` may have put it in place of a word write because the box was
    known to hold the word, and that can rest on this write having made the box that long.

    Without that promise and without `pages_are_boxes`, every write stays.
    """
    summaries = context.effects
    if summaries is None or summaries.storage is None:
        return False
    storage = summaries.storage
    promised = storage.declaration.overwritten_writes_removable
    if not (promised or storage.pages_are_boxes):
        return False
    follower = _Follower.of(summaries, subroutine)
    if not _has_word_writes(subroutine, follower.values, promised=promised):
        return False
    if storage.pages_are_boxes:
        # which page word writes find their word in the box
        order = _reverse_post_order(subroutine.body[0])
        at_end = _known_at_block_ends(follower, order)
        follower.mode = "note"
        for block in order:
            follower.follow(block, _known_at_start(block, order[0], at_end, follower))
    removable = _ReplacedWrites(follower=follower, promised=promised).find(subroutine)
    if not removable:
        return False
    for block in subroutine.body:
        ops = [op for op in block.ops if op not in removable]
        if len(ops) != len(block.ops):
            for op in block.ops:
                if op in removable:
                    logger.debug(
                        f"removing {op}, which a later write replaces",
                        location=op.source_location,
                    )
            block.ops = ops
    return True


def _has_word_writes(subroutine: models.Subroutine, values: SlotValues, *, promised: bool) -> bool:
    """True if the subroutine has a write that could be removed as replaced."""
    for block in subroutine.body:
        for op in block.ops:
            if isinstance(op, models.InvokeSubroutine):
                role = values.roles.get(op.target.id)
                if role == "write_page_word" or (promised and role in WRITE_ROLES):
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
    summaries: EffectSummaries
    values: SlotValues
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
    """"follow" only updates what is known; "note" also notes which page word writes find
    their word in the box; "count" also notes which page reads would share; "rewrite"
    replaces ops"""
    held: set[models.InvokeSubroutine] = attrs.field(factory=set)
    """the word writes of a page whose box is known to hold the word when they run"""
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
    _changes_calls: dict[str, Effect] = attrs.field(factory=dict)
    """by subroutine id, for the calls that are remembered: the effects that change what the
    subroutine reads"""
    _effects: dict[models.IRVisitable, Effect] = attrs.field(factory=dict)
    """the effects of the operations seen so far that are not calls"""

    @classmethod
    def of(
        cls, summaries: EffectSummaries, subroutine: models.Subroutine, *, pages: bool = False
    ) -> "_Follower":
        storage = summaries.storage
        declaration = storage.declaration if storage is not None else None
        return cls(
            summaries=summaries,
            values=SlotValues.of(subroutine, storage.roles if storage is not None else {}),
            hashed_apart=declaration is not None and declaration.hashed_slots_apart,
            page_boxes=storage is not None and storage.pages_are_boxes,
            pages=pages,
            inline_reads=pages and declaration is not None and declaration.inline_page_reads,
            inline_writes=pages and declaration is not None and declaration.inline_page_writes,
        )

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
                    covered[page] = max(covered.get(page, 0), at + WORD)
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
        node: models.IRVisitable = op.source if isinstance(op, models.Assignment) else op
        if isinstance(node, models.InvokeSubroutine):
            targets = tuple(op.targets) if isinstance(op, models.Assignment) else ()
            role = self.values.roles.get(node.target.id)
            if role in READ_ROLES and len(targets) == 1:
                slot = self.values.accessed(role, node)
                if slot is not None:
                    return self._word_read(op, node, targets[0], slot, role, known)
            elif role in WRITE_ROLES:
                return self._word_write(role, node, known, statement=op is node)
            return self._call(op, node, targets, known)
        if isinstance(node, models.Assert):
            if self.page_boxes:
                self._cover(node.condition, known, known.covered, holds=True)
            return None
        if isinstance(node, models.Value):
            return None
        effects = self.effects_of(node)
        if effects & BOX_CHANGES and not self._page_word_replaced(node, known):
            self._boxes_changed(effects, known)
        self._forget_calls(effects, known)
        return None

    def effects_of(self, node: models.IRVisitable) -> Effect:
        """What an operation that is not a call may do, other than fail."""
        try:
            return self._effects[node]
        except KeyError:
            effects = self._effects[node] = effects_of(node)
            return effects

    def _page_word_replaced(self, op: models.IRVisitable, known: _Known) -> bool:
        """A `box_replace` of 32 bytes at a word of a page box is a write of that word."""
        replaced = self.values.page_word_replaced(op) if self.page_boxes else None
        if replaced is None:
            return False
        slot, value = replaced
        self._slot_written(slot, 1, _Word(value=value, form="word"), known)
        return True

    def _boxes_changed(self, effects: Effect, known: _Known) -> None:
        """An operation with the given effects may have changed any box, and not through a
        declared write."""
        known.words.clear()
        known.pages.clear()
        if effects & _CHANGES_BOX_SIZE:
            known.covered.clear()

    def _forget_calls(self, effects: Effect, known: _Known) -> None:
        """Forgets the results of the calls that an operation with the given effects may have
        changed the outcome of."""
        if not effects & _CHANGES_STATE:
            return
        for key in [k for k in known.calls if self._changes_calls[k[0]] & effects]:
            del known.calls[key]

    def _call(
        self,
        op: models.Op,
        call: models.InvokeSubroutine,
        targets: tuple[models.Register, ...],
        known: _Known,
    ) -> Sequence[models.Op] | None:
        effects = self.summaries.of(call.target)
        if effects & BOX_CHANGES:
            changes = self.summaries.box_changes(call.target)
            if changes.other:
                self._boxes_changed(effects, known)
            else:
                for slot in [
                    s for s in known.words if not all(self._apart(s, w) for w in changes.words)
                ]:
                    del known.words[slot]
                for write in changes.words:
                    self._pages_written(write, known)
        self._forget_calls(effects, known)
        if not is_repeatable(effects) or any(isinstance(a, models.Undefined) for a in call.args):
            return None
        key = (call.target.id, tuple(call.args))
        earlier = known.calls.get(key)
        if earlier is None or (targets and not earlier):
            known.calls[key] = targets
            self._changes_calls[call.target.id] = changed_by(effects & READS)
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
        slot: Slot,
        role: str,
        known: _Known,
    ) -> Sequence[models.Op] | None:
        word = known.words.get(slot)
        if word is not None and word.value.ir_type.maybe_avm_type != AVMType.bytes:
            word = None
        if word is None:
            known.words[slot] = _Word(value=target, form="word")
            if self.pages and role in PAGE_ROLES:
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
                immediates=[0, WORD],
                args=[word.value],
                source_location=location,
            )
            return [models.Assignment(targets=[target], source=first, source_location=location)]
        zeros = _derived(target, "zeros", PrimitiveIRType.bytes)
        return [
            _assign(zeros, AVMOp.bzero, _uint64(WORD, location)),
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
                _assign(zeros, AVMOp.bzero, _uint64(max(positions) + WORD, location)),
                _assign(words, AVMOp.concat, box, zeros),
                _assign(target, AVMOp.extract3, words, call.args[1], _uint64(WORD, location)),
            ]
        if at in page.written:
            return None
        if self.mode == "count":
            self.page_reads[page.reader].append(at)
        if self.mode != "rewrite" or page.reader not in self.page_reads:
            return None
        self._log(f"taking the word of {op} from the page read earlier", op)
        words = _derived(page.reader, "page", PrimitiveIRType.bytes)
        return [_assign(target, AVMOp.extract3, words, call.args[1], _uint64(WORD, location))]

    def _word_write(
        self, role: str, call: models.InvokeSubroutine, known: _Known, *, statement: bool
    ) -> Sequence[models.Op] | None:
        self._forget_calls(self.summaries.of(call.target), known)
        written = self.values.written(role, call)
        if written is None:
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
        if self.page_boxes and statement and role == "write_page_word":
            page, at = _page_and_position(call)
            if known.covered.get(page, 0) >= at + WORD:
                if self.mode == "note":
                    self.held.add(call)
                elif self.pages:
                    replacement = self._replace_word(call)
            elif self.pages and self.inline_writes:
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
                _assign(zeros, AVMOp.bzero, _uint64(WORD, location)),
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

    def _slot_written(self, slot: Slot, count: int, word: _Word, known: _Known) -> None:
        kind = self.values.kind(slot, count)
        for other in [
            s
            for s in known.words
            if not (
                (s.offset - slot.offset) % SLOTS >= count
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
                known.covered[page] = max(known.covered.get(page, 0), at + WORD)

    def _pages_written(self, write: SlotWrite, known: _Known) -> None:
        for key, page in list(known.pages.items()):
            first = int.from_bytes(key)
            if write.constant is not None:
                positions = {
                    distance * WORD
                    for n in range(write.count)
                    if (distance := (write.constant + n - first) % SLOTS) < PAGE_WORDS
                }
                if positions:
                    known.pages[key] = _Page(reader=page.reader, written=page.written | positions)
            elif not self._apart(Slot(base=None, offset=first), write):
                del known.pages[key]

    def slots_apart(self, slot: Slot, other: Slot) -> bool:
        """True if two slots are certainly not the same."""
        if slot.base == other.base:
            return slot.offset != other.offset
        return self._apart(slot, self.values.kind(other, 1))

    def _apart(self, slot: Slot, written: SlotWrite) -> bool:
        """True if the word of a slot is certainly not one such a write changes."""
        if slot.base is None and written.constant is not None:
            return (slot.offset - written.constant) % SLOTS >= written.count
        if not self.hashed_apart:
            return False
        if written.constant is not None:
            return (
                written.constant + written.count <= SMALL
                and slot.offset < SMALL
                and self.values.hashed(slot.base) is not None
            )
        if written.ending is None:
            return False
        if slot.base is None:
            # a larger constant can be a hash worked out ahead
            return slot.offset < SMALL
        ending = self.values.hashed(slot.base)
        if ending is None or slot.offset >= SMALL:
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


def _write_in_place(
    subroutine: models.Subroutine, write: _WriteInPlace, values: SlotValues
) -> None:
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
            _assign(zeros, AVMOp.bzero, _uint64(WORD, location)),
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
        _assign(holds, AVMOp.gte, size, _uint64(at.value + WORD, location)),
    ]
    block.terminator = models.ConditionalBranch(
        condition=holds, non_zero=replace, zero=grow, source_location=location
    )
    for branch in (replace, grow):
        branch.add_predecessor(block)
        after.add_predecessor(branch)
    position = subroutine.body.index(block) + 1
    subroutine.body[position:position] = [replace, grow, after]


@attrs.frozen
class _Replacing:
    """What is known of the word of a slot at a point of a subroutine: on every way on from
    there, either the program fails, or a write through `writer` to that slot comes before
    anything that can tell which word the slot holds."""

    writer: str
    """the id of the subroutine that the write goes through"""
    nonzero: bool
    """the word it writes is certainly not zero"""
    shrinks: bool
    """a box may get shorter, or be deleted, before it runs"""


_Replaced = dict[Slot, _Replacing] | None
"""by slot; None where the program fails on every way on, so that nothing can ever tell"""


@attrs.define
class _ReplacedWrites:
    follower: _Follower
    """what is known going forward, with the slots and values of the subroutine"""
    promised: bool
    """whether the declaration promises that a replaced write can be removed"""
    removable: set[models.Op] = attrs.field(factory=set)
    _reach_in_place: set[models.BasicBlock] = attrs.field(factory=set)
    """the blocks from whose start a `box_replace` can be reached"""
    _in_place_follows: bool = False
    """whether a `box_replace` can be reached from the op that is being looked at"""

    def find(self, subroutine: models.Subroutine) -> set[models.Op]:
        """The word writes of a subroutine that can be removed as replaced.

        What is known where a block starts follows from what is known where its successors
        start, so the blocks are walked from the last one back, until nothing changes. A
        successor that was not walked yet (a back edge) is taken to agree with the others.
        """
        order = list(reversed(_reverse_post_order(subroutine.body[0])))
        self._reach_in_place = _blocks_reaching(order, _is_box_replace)
        at_start = dict[models.BasicBlock, _Replaced]()
        for _ in range(_MAX_WALKS):
            changed = False
            self.removable.clear()
            for block in order:
                state = self._at_start(block, at_start)
                if block not in at_start or at_start[block] != state:
                    at_start[block] = state
                    changed = True
            if not changed:
                return self.removable
        # no agreement reached: every write stays
        return set()

    def _at_start(
        self, block: models.BasicBlock, at_start: Mapping[models.BasicBlock, _Replaced]
    ) -> _Replaced:
        state = self._at_end(block, at_start)
        self._in_place_follows = any(s in self._reach_in_place for s in block.successors)
        for op in reversed(block.ops):
            state = self._before(op, state)
            self._in_place_follows = self._in_place_follows or _is_box_replace(op)
        if state and block.phis:
            # a phi gives its register another value each time the block is entered
            state = _without_slots_of([phi.register for phi in block.phis], state)
        return state

    def _at_end(
        self, block: models.BasicBlock, at_start: Mapping[models.BasicBlock, _Replaced]
    ) -> _Replaced:
        """What the successors of a block agree on."""
        match block.terminator:
            case models.Fail():
                return None
            case models.SubroutineReturn() | models.ProgramExit() | None:
                # whoever called may read anything, and what is written stays as the program
                # ends
                return {}
        states = [
            state
            for successor in block.successors
            if (state := at_start.get(successor)) is not None
        ]
        if not states:
            return None
        first, *rest = states
        agreed = dict[Slot, _Replacing]()
        for slot, replacing in first.items():
            others = [state.get(slot) for state in rest]
            if all(other is not None and other.writer == replacing.writer for other in others):
                agreed[slot] = _Replacing(
                    writer=replacing.writer,
                    nonzero=replacing.nonzero and all(o.nonzero for o in others if o),
                    shrinks=replacing.shrinks or any(o.shrinks for o in others if o),
                )
        return agreed

    def _before(self, op: models.Op, state: _Replaced) -> _Replaced:
        """Takes what is known after an op to before it."""
        node: models.IRVisitable = op.source if isinstance(op, models.Assignment) else op
        if isinstance(node, models.InvokeSubroutine):
            state = self._before_call(op, node, state)
        elif (
            state
            and not isinstance(node, models.Value | models.Assert)
            and self.follower.effects_of(node) & _TELLS_BOX_CHANGES
        ):
            state = {}
        if state and isinstance(op, models.Assignment):
            # a register has its value from here on: what is known of a slot that it names
            # is not known of the slot that an earlier value of it named
            state = _without_slots_of(op.targets, state)
        return state

    def _before_call(
        self, op: models.Op, call: models.InvokeSubroutine, state: _Replaced
    ) -> _Replaced:
        follower = self.follower
        effects = follower.summaries.of(call.target)
        if Effect.exits in effects:
            # the program may end in the call, with every write so far in place
            state = {}
        role = follower.values.roles.get(call.target.id)
        if role in WRITE_ROLES:
            return self._before_write(op, call, role, effects, state)
        if not state:
            return state
        if role in READ_ROLES:
            read = follower.values.accessed(role, call)
            if read is not None:
                return {
                    slot: replacing
                    for slot, replacing in state.items()
                    if follower.slots_apart(slot, read)
                }
        if effects & _TELLS_BOX_CHANGES:
            return {}
        return state

    def _before_write(
        self,
        op: models.Op,
        call: models.InvokeSubroutine,
        role: str,
        effects: Effect,
        state: _Replaced,
    ) -> _Replaced:
        values = self.follower.values
        written = values.written(role, call)
        if state and role != "write_page_word":
            # it may delete a box, or make one shorter; a word write of a page does neither
            state = {slot: attrs.evolve(known, shrinks=True) for slot, known in state.items()}
        if written is None or written[1] != 1 or op is not call:
            # it replaces no word for certain, and it reads none
            return state
        slot, _ = written
        value = call.args[-1]
        later = state.get(slot) if state else None
        if (
            later is not None
            and later.writer == call.target.id
            and not effects & ~_ONLY_WRITES_BOXES
            and self._unneeded(call, role, slot, value, later)
        ):
            self.removable.add(op)
            # what replaces it is what replaces the words before it
            return state
        if state is None:
            return None
        state = dict(state)
        state[slot] = _Replacing(
            writer=call.target.id, nonzero=values.is_nonzero(value), shrinks=False
        )
        return state

    def _unneeded(
        self,
        call: models.InvokeSubroutine,
        role: str,
        slot: Slot,
        value: models.Value,
        later: _Replacing,
    ) -> bool:
        """True if a write that a later write to the same slot replaces can go: the later
        write fails whenever this one would, and leaves the same boxes behind."""
        follower = self.follower
        if not follower.page_boxes or follower.values.page_position(slot) is None:
            # nothing is known of what the write does to its box, other than what the
            # declaration promises
            return self.promised
        # a word of a page, where what a write does to the box is declared
        if (
            role == "write_page_word"
            and not later.shrinks
            and (call in follower.held or follower.values.is_zero(value) or later.nonzero)
        ):
            return True
        # Otherwise the box could end up shorter without this write. A `box_replace` that
        # is there because a page box was known to hold a word can rest on this write
        # having made the box that long, so that the promise is not enough before one.
        return self.promised and not self._in_place_follows


def _is_box_replace(op: models.Op) -> bool:
    return isinstance(op, models.Intrinsic) and op.op is AVMOp.box_replace


def _blocks_reaching(
    post_order: Sequence[models.BasicBlock], wanted: Callable[[models.Op], bool]
) -> set[models.BasicBlock]:
    """The blocks from whose start an op of the wanted kind can be reached."""
    reaching = {block for block in post_order if any(map(wanted, block.ops))}
    pending = list(reaching)
    while pending:
        for predecessor in pending.pop().predecessors:
            if predecessor not in reaching:
                reaching.add(predecessor)
                pending.append(predecessor)
    return reaching


def _without_slots_of(
    registers: Sequence[models.Register], state: dict[Slot, _Replacing]
) -> dict[Slot, _Replacing]:
    if any(slot.base in registers for slot in state):
        return {slot: known for slot, known in state.items() if slot.base not in registers}
    return state
