import enum
from collections.abc import Mapping
from functools import cached_property

import attrs
from immutabledict import immutabledict

from puya.algo_constants import MAINNET_AVM_VERSION


class LocalsCoalescingStrategy(enum.StrEnum):
    root_operand = enum.auto()
    root_operand_excluding_args = enum.auto()
    aggressive = enum.auto()


@attrs.frozen(kw_only=True)
class SlotStorage:
    """Subroutines with which a program keeps 32-byte words in boxes, by subroutine id (a name
    also stands for the subroutines whose id is that name, a `.` and a qualifier). Each word has
    a slot, a number below 2**256; a slot that was never written holds 32 zero bytes.

    * `read_page_word(page, at) -> bytes`: the word of slot `page + at / 32`, where `page` is 32
      bytes read as a number and `at` a multiple of 32
    * `write_page_word(page, at, value)`: sets the word of that slot to `value`
    * `read_word(slot) -> bytes`: the word of `slot`, 32 bytes read as a number
    * `write_words(slot, words)`: sets the words of `slot` and of the slots after it to the
      32-byte pieces of `words`, which holds between 1 and 64 of them
    * `read_slot(number) -> bytes`, `write_slot(number, value)`: the same for the slot
      `number` modulo 2**256
    * `slot_add(slot, n) -> bytes`: the 32 bytes of `slot + n` modulo 2**256, with no effect

    Declaring them promises that:

    * a `value` that is passed to a write is at most 32 bytes long, and a read returns the 32
      bytes last written to its slot by any of the writers, where a `value` is stored
      left-padded with zero bytes;
    * a write changes the word of no other slot;
    * a read has no effect, and does not fail once a read or a write of the same slot has
      succeeded in the same transaction;
    * nothing else in the program changes these words, except through box operations and
      inner transactions that the optimizer can see.

    `pages_are_boxes` promises that the words the page subroutines reach are the bytes of the
    box named `page` of the application itself, 32 for each word from byte 0, that a word past
    the end of the box or of a box that does not exist is zero, that such a box is at most 2048
    bytes long, and that no write through any of the declared subroutines makes it shorter. It
    also promises what a word write of a page does. Where the box holds the word, it replaces
    its 32 bytes. Where it does not, a word that is zero changes nothing, and any other word
    first makes the box just long enough to hold it, creating the box if there is none. It
    fails only where one of these box operations does. Word reads of one page can then share
    one read of its box, a word write can replace the bytes in the box itself where the box is
    known to hold the word, and a word write that a later write of the same word replaces can
    be left out where the later one alone does the same to the box. It is ignored for a
    program whose page subroutines use the boxes of another application, read the ledger or
    submit inner transactions.

    `overwritten_writes_removable` promises that the program does not depend on a write that
    is replaced before anything can tell. A write through one of these subroutines, followed
    by a write through the same subroutine to the same slot, may be left out if no operation
    between the two reads a box or the ledger (a balance or a minimum balance, which follows
    the sizes of the boxes), submits an inner transaction or can end the program. This is a
    promise about more than the words: either the later write fails wherever the earlier one
    would have and the two leave the boxes as the later one alone does, whatever the values
    are, or the program does not depend on the difference. For a word of a page with
    `pages_are_boxes`, where what a write does is known, it is used only where the later write
    is not known to do the same to the box.

    Two requests trade program size for fewer operations run. `inline_page_reads` asks for a
    word read of a page to read the box itself even where no other read shares it.
    `inline_page_writes` asks for a word write of a page whose box was read to compare the
    length read with the position of the word, and to replace the bytes in place where the box
    holds it.

    `hashed_slots_apart` promises that the program does not depend on two slots being the same
    when they are derived differently: the result of a hash op, plus a number below 2**64, is
    no slot below 2**64, and is not the result of hashing bytes with another constant ending,
    plus a number below 2**64.
    """

    read_page_word: str | None = None
    write_page_word: str | None = None
    read_word: str | None = None
    write_words: str | None = None
    read_slot: str | None = None
    write_slot: str | None = None
    slot_add: str | None = None
    pages_are_boxes: bool = False
    inline_page_reads: bool = False
    inline_page_writes: bool = False
    hashed_slots_apart: bool = False
    overwritten_writes_removable: bool = False


@attrs.frozen(kw_only=True)
class PuyaOptions:
    output_teal: bool = False
    output_source_map: bool = False
    output_assembly_report: bool = False
    output_arc32: bool = False
    output_arc56: bool = False
    output_ssa_ir: bool = False
    output_optimization_ir: bool = False
    output_destructured_ir: bool = False
    output_memory_ir: bool = False
    output_bytecode: bool = False
    output_teal_intermediates: bool = False
    output_op_statistics: bool = False
    debug_level: int = 1
    optimization_level: int = 1
    treat_warnings_as_errors: bool = False
    target_avm_version: int = MAINNET_AVM_VERSION
    cli_template_definitions: Mapping[str, int | bytes] = attrs.field(factory=dict)
    template_vars_prefix: str = "TMPL_"
    # TODO: the below is probably not scalable as a set of optimisation on/off flags,
    #       but it'll do for now
    locals_coalescing_strategy: LocalsCoalescingStrategy = LocalsCoalescingStrategy.root_operand
    optimizations_override: Mapping[str, bool] = attrs.field(default=immutabledict())
    expand_all_bytes: bool = False
    validate_abi_args: bool = True
    validate_abi_return: bool = True
    slot_storage: SlotStorage | None = None

    @cached_property
    def template_variables(self) -> Mapping[str, int | bytes]:
        return {self.template_vars_prefix + k: v for k, v in self.cli_template_definitions.items()}
