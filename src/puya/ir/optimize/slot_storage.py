"""Declared slot storage (see `puya.options.SlotStorage`) as it applies to one program: which
of its subroutines read and write words, and which slot a call to one of them names.

A slot is followed through the definitions of a subroutine as a base and a distance from it:
a constant number, or a register plus a constant. Two slots with one base at different
distances are different slots, and so are two different constants. What else is known to be
apart is what the declaration promises.
"""

import functools
import json
import os
import typing
from collections.abc import Mapping

import attrs

from puya.avm import AVMType
from puya.ir import models
from puya.ir._utils import get_bytes_constant
from puya.ir.avm_ops import AVMOp
from puya.ir.types_ import IRType
from puya.options import SlotStorage
from puya.parse import SourceLocation

WORD: typing.Final = 32
"""bytes in a word"""
SLOTS: typing.Final = 2**256
SMALL: typing.Final = 2**64
PAGE_WORDS: typing.Final = 64

READ_ROLES: typing.Final = frozenset(("read_page_word", "read_word", "read_slot"))
WRITE_ROLES: typing.Final = frozenset(("write_page_word", "write_words", "write_slot"))
PAGE_ROLES: typing.Final = frozenset(("read_page_word", "write_page_word"))
SLOT_ADD: typing.Final = "slot_add"

_HASHES: typing.Final = frozenset(
    (AVMOp.keccak256.code, AVMOp.sha256.code, AVMOp.sha512_256.code, AVMOp.sha3_256.code)
)
# how many ops deep a word is followed back to the word that it was made from
_MAX_CHANGES: typing.Final = 64
# ops whose result is a number without leading zero bytes
_BYTE_MATH: typing.Final = frozenset(
    (
        AVMOp.add_bytes,
        AVMOp.sub_bytes,
        AVMOp.mul_bytes,
        AVMOp.div_floor_bytes,
        AVMOp.mod_bytes,
        AVMOp.bsqrt,
    )
)


@attrs.frozen(kw_only=True)
class SlotWrite:
    """A write to declared slot storage, as far as it can be described without the values of
    the subroutine it is in. With neither `constant` nor `ending`, it can be to any slot."""

    constant: int | None = None
    """the slot, if it is a constant number"""
    count: int = 1
    """how many slots from a constant one on it can reach"""
    ending: bytes | None = None
    """for a slot that is the result of a hash op plus a number below 2**64: the constant bytes
    the hashed bytes end in"""


ANY_SLOT: typing.Final = SlotWrite()


@attrs.frozen(kw_only=True)
class DeclaredStorage:
    """A declaration of slot storage, for the subroutines of one program."""

    declaration: SlotStorage
    roles: Mapping[str, str]
    """subroutine id -> the declared role of that subroutine"""
    pages_are_boxes: bool
    """whether the pages of the declaration are boxes of the program itself"""


def declared_storage(option: SlotStorage | None) -> SlotStorage | None:
    # PUYA_SLOT_STORAGE (the option as JSON) stands in for the option while a frontend cannot
    # write it yet
    return option or _storage_from_environment(os.environ.get("PUYA_SLOT_STORAGE"))


@functools.cache
def _storage_from_environment(declaration: str | None) -> SlotStorage | None:
    return SlotStorage(**json.loads(declaration)) if declaration else None


def storage_roles(storage: SlotStorage, program: models.Program) -> dict[str, str]:
    """subroutine id -> the declared role of that subroutine"""
    roles = dict[str, str]()
    for role in (*sorted(READ_ROLES), *sorted(WRITE_ROLES), SLOT_ADD):
        name = getattr(storage, role)
        if name is not None:
            for sub in program.subroutines:
                # a program that holds several copies of a subroutine qualifies their ids
                if sub.id == name or sub.id.startswith(f"{name}."):
                    roles[sub.id] = role
    return roles


@attrs.frozen
class Slot:
    base: models.Value | None
    """None for a slot that is a constant number"""
    offset: int
    """modulo 2**256"""


@attrs.frozen
class SlotValues:
    """The declared roles, and what the definitions in a subroutine say about a value."""

    roles: Mapping[str, str]
    definitions: Mapping[models.Register, models.ValueProvider]
    pages: frozenset[bytes]
    """the constant pages the subroutine reads or writes words of"""
    versions: dict[str, int]
    """the highest version assigned of each register name"""
    checked: frozenset[models.Register] = frozenset()
    """results of byte math that an assert finds to fit a word, see `_checked_to_fit_a_word`"""
    _words: dict[models.Register, bool] = attrs.field(factory=dict, init=False, eq=False)
    _fitting: dict[models.Register, bool] = attrs.field(factory=dict, init=False, eq=False)

    @classmethod
    def of(cls, subroutine: models.Subroutine, roles: Mapping[str, str]) -> "SlotValues":
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
                    ) if roles.get(target.id) in PAGE_ROLES and len(page) == WORD:
                        pages.add(page)
        return cls(
            roles=roles,
            definitions=definitions,
            pages=frozenset(pages),
            versions=versions,
            checked=_checked_to_fit_a_word(subroutine, definitions),
        )

    def fresh(
        self, name: str, ir_type: IRType, location: SourceLocation | None
    ) -> models.Register:
        """A register no assignment of the subroutine has as its target."""
        version = self.versions[name] = self.versions.get(name, -1) + 1
        return models.Register(
            name=name, version=version, ir_type=ir_type, source_location=location
        )

    def page_position(self, slot: Slot) -> tuple[bytes, int] | None:
        """The page a constant slot is in, of those the subroutine names, and its byte
        position there."""
        if slot.base is None:
            for page in self.pages:
                distance = (slot.offset - int.from_bytes(page)) % SLOTS
                if distance < PAGE_WORDS:
                    return page, distance * WORD
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

    def slot(self, value: models.Value) -> Slot | None:
        """The slot a value names, as a base and a distance from it; None if the value is not
        one that is the same wherever it is used."""
        offset = 0
        while True:
            value = self.copied(value)
            number = _number(value)
            if number is not None:
                return Slot(base=None, offset=(number + offset) % SLOTS)
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
                ) if self.roles.get(target.id) == SLOT_ADD:
                    offset += distance
                    value = base
                case _:
                    break
        if not isinstance(value, models.Register):
            return None
        return Slot(base=value, offset=offset % SLOTS)

    def accessed(self, role: str, call: models.InvokeSubroutine) -> Slot | None:
        """The slot a declared read or write is of, None if it cannot be told."""
        if role not in PAGE_ROLES:
            return self.slot(call.args[0])
        match call.args:
            case [models.BytesConstant(value=page), models.UInt64Constant(value=at), *_] if (
                len(page) == WORD and at % WORD == 0
            ):
                return Slot(base=None, offset=(int.from_bytes(page) + at // WORD) % SLOTS)
        return None

    def written(self, role: str, call: models.InvokeSubroutine) -> tuple[Slot, int] | None:
        """The first slot a declared write is to, and how many slots it may reach."""
        slot = self.accessed(role, call)
        if slot is None:
            return None
        if role == "write_words" and not self.is_one_word(call.args[1]):
            return slot, PAGE_WORDS
        return slot, 1

    def is_one_word(self, words: models.Value) -> bool:
        """True if what is passed to a write of words is certainly one word. It is declared
        to be a whole number of words, and at least one, so it is one word if it is less than
        64 bytes long: the sum of two values of at most 32 bytes takes 33 bytes at most, and
        a bitwise op gives as many bytes as its longer operand has."""
        return self.is_word(words) or self._under_two_words(words, depth=8)

    def _under_two_words(self, value: models.Value, *, depth: int) -> bool:
        if self.fits_word(value):
            return True
        if not depth:
            return False
        match self.definition(value):
            case models.Intrinsic(op=AVMOp.add_bytes, args=[a, b]):
                return self.fits_word(a) and self.fits_word(b)
            case models.Intrinsic(
                op=(AVMOp.bitwise_or_bytes | AVMOp.bitwise_and_bytes | AVMOp.bitwise_xor_bytes),
                args=[a, b],
            ):
                return self._under_two_words(a, depth=depth - 1) and self._under_two_words(
                    b, depth=depth - 1
                )
        return False

    def page_word_replaced(self, op: models.IRVisitable) -> tuple[Slot, models.Value] | None:
        """For a `box_replace` of 32 bytes at a word of a box that can be a page: the slot of
        that word, and the bytes. It is a write of that word where pages are boxes."""
        match op:
            case models.Intrinsic(
                op=AVMOp.box_replace,
                args=[
                    models.BytesConstant(value=page),
                    models.UInt64Constant(value=at),
                    value,
                ],
            ) if len(page) == WORD and at % WORD == 0 and self.is_word(value):
                return Slot(base=None, offset=(int.from_bytes(page) + at // WORD) % SLOTS), value
        return None

    def kind(self, slot: Slot, count: int) -> SlotWrite:
        """A write as other subroutines see it: without the values of this one."""
        if slot.base is None:
            return SlotWrite(constant=slot.offset, count=count)
        ending = self.hashed(slot.base)
        if ending is None or slot.offset + count > SMALL:
            return ANY_SLOT
        return SlotWrite(ending=ending)

    def hashed(self, base: models.Value | None) -> bytes | None:
        """For the result of a hash op, the constant ending of the bytes hashed."""
        if base is None:
            return None
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
            return len(value.value) == WORD
        if not isinstance(value, models.Register):
            return False
        try:
            return self._words[value]
        except KeyError:
            pass
        result = self._words[value] = bool(depth) and self._is_word(value, depth - 1)
        return result

    def _is_word(self, value: models.Register, depth: int) -> bool:
        match self.definitions.get(value):
            case models.InvokeSubroutine(target=target):
                return self.roles.get(target.id) in (*READ_ROLES, SLOT_ADD)
            case models.Intrinsic(op=op) if op.code in _HASHES:
                return True
            case models.Intrinsic(op=AVMOp.bzero, args=[models.UInt64Constant(value=n)]):
                return n == WORD
            case models.Intrinsic(op=AVMOp.extract3, args=[_, _, models.UInt64Constant(value=n)]):
                return n == WORD
            case models.Intrinsic(op=AVMOp.extract, immediates=[_, int(n)]):
                return n == WORD
            case models.Intrinsic(
                op=(AVMOp.replace2 | AVMOp.replace3 | AVMOp.setbyte | AVMOp.setbit),
                args=[target, *_],
            ):
                return self.is_word(target, depth=depth)
            case models.Intrinsic(
                op=(AVMOp.bitwise_or_bytes | AVMOp.bitwise_and_bytes | AVMOp.bitwise_xor_bytes),
                args=[a, b],
            ):
                # as long as the longer of the two
                return (self.is_word(a, depth=depth) and self.fits_word(b, depth=depth)) or (
                    self.is_word(b, depth=depth) and self.fits_word(a, depth=depth)
                )
        return False

    def changed_from(self, value: models.Value, base: models.Value) -> tuple[int, int] | None:
        """For a word that is the word `base` with bytes or bits replaced, or with the bits of
        a constant set: the bytes in which a byte of `base` that is not zero may have become
        zero, and the bytes in which the word may differ from `base` at all. Each is a number
        with bit n set for byte n of the word. None if `value` is not known to be made from
        `base` in one of these ways.
        """
        base = self.copied(base)
        zeroed = changed = 0
        for _ in range(_MAX_CHANGES):
            value = self.copied(value)
            if value == base:
                return zeroed, changed
            keeps_set_bits = False
            match self.definition(value):
                case models.Intrinsic(
                    op=AVMOp.replace2, immediates=[int(start)], args=[target, part]
                ):
                    mask = _byte_mask(start, self.exact_length(part))
                case models.Intrinsic(
                    op=AVMOp.replace3,
                    args=[target, models.UInt64Constant(value=start), part],
                ):
                    mask = _byte_mask(start, self.exact_length(part))
                case models.Intrinsic(
                    op=AVMOp.setbyte, args=[target, models.UInt64Constant(value=start), _]
                ):
                    mask = _byte_mask(start, 1)
                case models.Intrinsic(
                    op=AVMOp.setbit, args=[target, models.UInt64Constant(value=bit), _]
                ) if target.atype == AVMType.bytes:
                    mask = _byte_mask(bit // 8, 1)
                case models.Intrinsic(op=AVMOp.bitwise_or_bytes, args=[a, b]):
                    # a bit that is set stays set, and the constant is right-aligned in the word
                    a_constant = self._constant(a)
                    constant, target = (
                        (self._constant(b), a) if a_constant is None else (a_constant, b)
                    )
                    if constant is None or len(constant) > WORD:
                        return None
                    mask = sum(
                        1 << (WORD - len(constant) + at)
                        for at, byte in enumerate(constant)
                        if byte
                    )
                    keeps_set_bits = True
                case _:
                    return None
            if mask is None:
                return None
            changed |= mask
            if not keeps_set_bits:
                zeroed |= mask
            value = target
        return None

    def exact_length(self, value: models.Value) -> int | None:
        """The number of bytes that a value certainly has, if it is known."""
        constant = self._constant(value)
        if constant is not None:
            return len(constant)
        match self.definition(value):
            case models.Intrinsic(op=AVMOp.itob):
                return 8
            case models.Intrinsic(op=AVMOp.bzero, args=[models.UInt64Constant(value=n)]):
                return n
            case models.Intrinsic(op=AVMOp.extract3, args=[_, _, models.UInt64Constant(value=n)]):
                return n
            case models.Intrinsic(op=AVMOp.extract, immediates=[_, int(n)]) if n:
                return n
        return WORD if self.is_word(value) else None

    def _constant(self, value: models.Value) -> bytes | None:
        value = self.copied(value)
        return get_bytes_constant(value) if isinstance(value, models.Constant) else None

    def fits_word(self, value: models.Value, *, depth: int = 8) -> bool:
        """True if the value is certainly at most 32 bytes long."""
        value = self.copied(value)
        if isinstance(value, models.Constant):
            constant = get_bytes_constant(value)
            return constant is not None and len(constant) <= WORD
        if not isinstance(value, models.Register):
            return False
        try:
            return self._fitting[value]
        except KeyError:
            pass
        result = self._fitting[value] = (
            value in self.checked
            or self.is_word(value, depth=depth)
            or (bool(depth) and self._fits_word(value, depth - 1))
        )
        return result

    def _fits_word(self, value: models.Register, depth: int) -> bool:
        match self.definitions.get(value):
            case models.Intrinsic(op=(AVMOp.itob | AVMOp.bsqrt)):
                # 8 bytes, and the root of a number of at most 64 bytes
                return True
            case models.Intrinsic(op=(AVMOp.sub_bytes | AVMOp.div_floor_bytes), args=[a, _]):
                # no more than the first operand, and without leading zero bytes
                return self.fits_word(a, depth=depth)
            case models.Intrinsic(op=AVMOp.mod_bytes, args=[a, b]):
                # less than the second operand and no more than the first
                modulus = _number(self.copied(b))
                if modulus is not None and 0 < modulus <= SLOTS:
                    return True
                return self.fits_word(a, depth=depth) or self.fits_word(b, depth=depth)
            case models.Intrinsic(
                op=(AVMOp.replace2 | AVMOp.replace3 | AVMOp.setbyte | AVMOp.bitwise_not_bytes),
                args=[target, *_],
            ):
                return self.fits_word(target, depth=depth)
            case models.Intrinsic(
                op=(AVMOp.bitwise_or_bytes | AVMOp.bitwise_and_bytes | AVMOp.bitwise_xor_bytes),
                args=[a, b],
            ):
                return self.fits_word(a, depth=depth) and self.fits_word(b, depth=depth)
            case models.Intrinsic(op=AVMOp.bzero, args=[models.UInt64Constant(value=n)]):
                return n <= WORD
            case models.Intrinsic(op=AVMOp.extract3, args=[_, _, models.UInt64Constant(value=n)]):
                return n <= WORD
            case models.Intrinsic(op=AVMOp.extract, immediates=[_, int(n)]):
                return 0 < n <= WORD
        return False


def _byte_mask(start: int, length: int | None) -> int | None:
    """A number with a bit set for each of `length` bytes of a word from byte `start` on."""
    if length is None or start + length > WORD:
        return None
    return ((1 << length) - 1) << start


def _checked_to_fit_a_word(
    subroutine: models.Subroutine, definitions: Mapping[models.Register, models.ValueProvider]
) -> frozenset[models.Register]:
    """The results of byte math that an assert, later in the block that computes them, finds
    to be no more than a number below 2**256, with no call and no `box_replace` in between.

    Byte math gives a number without leading zero bytes, so such a result is at most 32 bytes
    long once the assert has passed. Whatever uses the result runs after the op that computes
    it, and so after the assert, unless it is one of the ops between the two: only a call or
    a `box_replace` is asked whether what it is given is a word.
    """
    checked = set[models.Register]()
    for block in subroutine.body:
        computed = dict[models.Register, int]()
        last_use = -1
        for index, op in enumerate(block.ops):
            source = op.source if isinstance(op, models.Assignment) else op
            match op:
                case models.Assignment(targets=[target], source=models.Intrinsic(op=math)) if (
                    math in _BYTE_MATH
                ):
                    computed[target] = index
                case models.Assert(condition=models.Register() as condition):
                    limited = _limited(definitions.get(condition))
                    if limited is not None:
                        value, limit = limited
                        if (
                            isinstance(value, models.Register)
                            and limit < SLOTS
                            and computed.get(value, -1) > last_use
                        ):
                            checked.add(value)
            if isinstance(source, models.InvokeSubroutine) or (
                isinstance(source, models.Intrinsic) and source.op is AVMOp.box_replace
            ):
                last_use = index
    return frozenset(checked)


def _limited(condition: models.ValueProvider | None) -> tuple[models.Value, int] | None:
    """For a comparison of byte arrays as numbers against a constant: the value that is not
    above a number if the comparison holds, and that number."""
    match condition:
        case models.Intrinsic(op=AVMOp.lte_bytes, args=[value, limit]):
            strict = False
        case models.Intrinsic(op=AVMOp.lt_bytes, args=[value, limit]):
            strict = True
        case models.Intrinsic(op=AVMOp.gte_bytes, args=[limit, value]):
            strict = False
        case models.Intrinsic(op=AVMOp.gt_bytes, args=[limit, value]):
            strict = True
        case _:
            return None
    number = _number(limit)
    if number is None:
        return None
    return value, number - strict


def _number(value: models.Value) -> int | None:
    if isinstance(value, models.BigUIntConstant):
        return value.value
    if isinstance(value, models.BytesConstant):
        return int.from_bytes(value.value)
    return None
