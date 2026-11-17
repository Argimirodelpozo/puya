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
class LinearMemory:
    """Subroutines with which a program reads and writes 32-byte words of a byte-addressed
    memory that it keeps in scratch space, by subroutine id. Declaring them lets the optimizer
    take a word read from an earlier word write, and join neighbouring word writes.

    * `read_word(address) -> bytes`: the 32 bytes at `address`
    * `read_word_uint64(address) -> uint64`: the same word as a number, failing if it does not fit
    * `write_word(address, value)`: puts `value`, left-padded with zero bytes to 32, at `address`
    * `write_range(address, value)`: puts the bytes of `value` at `address`
    * `alloc(size) -> uint64`: moves the free-memory pointer and returns its earlier value

    `fresh_allocations` promises that `alloc` returns memory that only its result leads to: the
    free-memory pointer is not a word the program reaches through a computed address, and two
    allocations do not overlap. An allocation whose pointer never leaves a subroutine is then known
    to change only through word writes to that pointer.

    Both promises are the frontend's to give, program by program. They are false for a program
    whose own code can compute a memory address or look at the free-memory pointer (inline
    assembly): the optimizer cannot tell such code from the frontend's own accesses.

    `removable_allocations` promises that nothing observes where an object is, so an allocation
    whose pointer is not used can be dropped.
    """

    read_word: str | None = None
    read_word_uint64: str | None = None
    write_word: str | None = None
    write_range: str | None = None
    alloc: str | None = None
    fresh_allocations: bool = False
    removable_allocations: bool = False


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
    linear_memory: LinearMemory | None = None

    @cached_property
    def template_variables(self) -> Mapping[str, int | bytes]:
        return {self.template_vars_prefix + k: v for k, v in self.cli_template_definitions.items()}
