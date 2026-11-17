"""With a linear memory declared and the frontend's promise that allocations are fresh (see
`puya.options.LinearMemory`), a word that a subroutine reads through a pointer parameter, before
anything it does can have changed memory, can be read by its callers instead and handed over as an
argument.

This is done for a parameter that is used for nothing else, so that afterwards the parameter is
unused and dropped. The caller often knows the word, because it wrote it: then nothing is read at
all, and an object nobody reads any more does not have to be built (see
`scratch_writes.remove_unread_word_writes`).
"""

from collections.abc import Iterable, Iterator, Mapping

import attrs

from puya import log
from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.context import IROptimizationContext, ScratchEffects
from puya.ir.optimize.scratch_reads import _Memory, _reverse_post_order
from puya.ir.optimize.scratch_writes import _definitions
from puya.ir.register_read_collector import RegisterReadCollector

logger = log.get_logger(__name__)

_NAME_PREFIX = f"word{models.TMP_VAR_INDICATOR}"
_WORD_OF = f"{models.TMP_VAR_INDICATOR}word"


@attrs.frozen
class _CallSite:
    caller: models.Subroutine
    block: models.BasicBlock
    op: models.Op
    call: models.InvokeSubroutine


def _call_sites(program: models.Program) -> dict[str, list[_CallSite]]:
    sites = dict[str, list[_CallSite]]()
    for caller in program.all_subroutines:
        for block in caller.body:
            for op in block.ops:
                node = op.source if isinstance(op, models.Assignment) else op
                if isinstance(node, models.InvokeSubroutine):
                    sites.setdefault(node.target.id, []).append(
                        _CallSite(caller=caller, block=block, op=op, call=node)
                    )
    return sites


def _in_a_cycle(sites: Mapping[str, list[_CallSite]]) -> set[str]:
    """Subroutines that can reach themselves through calls."""
    callers = {callee: {site.caller.id for site in found} for callee, found in sites.items()}
    cyclic = set[str]()
    for start, direct in callers.items():
        seen = set[str]()
        pending = list(direct)
        while pending:
            current = pending.pop()
            if current == start:
                cyclic.add(start)
                break
            if current not in seen:
                seen.add(current)
                pending.extend(callers.get(current, ()))
    return cyclic


def _before_memory_changes(
    sub: models.Subroutine, effects: Mapping[str, ScratchEffects]
) -> Iterator[models.Op]:
    """The operations of a subroutine that run while memory is still as the caller left it."""
    unchanged_at_end = dict[models.BasicBlock, bool]()
    for block in _reverse_post_order(sub.entry):
        unchanged = block is sub.entry or (
            bool(block.predecessors)
            and all(unchanged_at_end.get(p, False) for p in block.predecessors)
        )
        if unchanged:
            for op in block.ops:
                if _may_write_scratch(op, effects):
                    unchanged = False
                    break
                yield op
        unchanged_at_end[block] = unchanged


def _may_write_scratch(op: models.Op, effects: Mapping[str, ScratchEffects]) -> bool:
    node = op.source if isinstance(op, models.Assignment) else op
    if isinstance(node, models.InvokeSubroutine):
        return effects[node.target.id].writes
    if isinstance(node, models.Intrinsic):
        return node.op in (AVMOp.store, AVMOp.stores)
    return isinstance(node, models.WriteSlot | models.NewSlot)


@attrs.define
class _Uses:
    """How a subroutine uses one of its parameters as a pointer."""

    only_reads: bool = True
    """every use is a word read while memory is still as the caller left it, or hands the pointer
    to a parameter of another subroutine of which the same holds"""
    reads: list[models.Assignment] = attrs.field(factory=list)
    passed_to: list[tuple[str, int]] = attrs.field(factory=list)


def _parameter_uses(
    sub: models.Subroutine, memory: _Memory, effects: Mapping[str, ScratchEffects]
) -> dict[int, _Uses]:
    index_of = {(p.name, p.version): index for index, p in enumerate(sub.parameters)}
    uses = {index: _Uses() for index in index_of.values()}
    early = {id(op) for op in _before_memory_changes(sub, effects)}

    def parameter(value: models.Value) -> int | None:
        if not isinstance(value, models.Register):
            return None
        base = memory.address(value).base
        if not isinstance(base, models.Register):
            return None
        return index_of.get((base.name, base.version))

    def other(values: Iterable[models.Value]) -> None:
        for value in values:
            index = parameter(value)
            if index is not None:
                uses[index].only_reads = False

    def reads(node: models.IRVisitable) -> Iterable[models.Value]:
        collector = RegisterReadCollector()
        node.accept(collector)
        return collector.used_registers

    for block in sub.body:
        for phi in block.phis:
            other(argument.value for argument in phi.args)
        for op in block.ops:
            node = op.source if isinstance(op, models.Assignment) else op
            match node:
                case models.Register():
                    pass  # a copy: the uses of the copy decide
                case models.Intrinsic(op=AVMOp.add, args=[models.UInt64Constant(), _]) | (
                    models.Intrinsic(op=(AVMOp.add | AVMOp.sub), args=[_, models.UInt64Constant()])
                ):
                    pass  # an address at a constant distance: its uses decide
                case models.InvokeSubroutine(target=target, args=arguments):
                    role = memory.role(target)
                    for position, argument in enumerate(arguments):
                        index = parameter(argument)
                        if index is None:
                            continue
                        if id(op) not in early:
                            uses[index].only_reads = False
                        elif role in ("read", "read64"):
                            if isinstance(op, models.Assignment) and len(op.targets) == 1:
                                uses[index].reads.append(op)
                            else:
                                uses[index].only_reads = False
                        elif role is None:
                            uses[index].passed_to.append((target.id, position))
                        else:
                            uses[index].only_reads = False
                case _:
                    other(reads(node))
        if block.terminator is not None:
            other(reads(block.terminator))
    return uses


def promote_word_reads(context: IROptimizationContext, program: models.Program) -> bool:
    declared = context.linear_memory
    effects = context.scratch_effects
    if declared is None or effects is None or not declared.fresh_allocations:
        return False
    sites = _call_sites(program)
    cyclic = _in_a_cycle(sites)
    accessors = {declared.read_word, declared.read_word_uint64}
    memories = dict[str, _Memory]()
    uses = dict[str, dict[int, _Uses]]()
    for sub in program.subroutines:
        if sub.id in cyclic or sub.id in accessors or sub.id not in sites:
            continue
        memories[sub.id] = _Memory(declared, _definitions(sub))
        uses[sub.id] = _parameter_uses(sub, memories[sub.id], effects)
    # a pointer handed on is only read if the receiving parameter is only read
    changed = True
    while changed:
        changed = False
        for found in uses.values():
            for use in found.values():
                if use.only_reads and not all(
                    callee in uses and uses[callee][position].only_reads
                    for callee, position in use.passed_to
                ):
                    use.only_reads = False
                    changed = True
    versions = dict[str, dict[str, int]]()
    modified = False
    for sub in program.subroutines:
        if sub.id not in uses:
            continue
        memory = memories[sub.id]
        promoted = dict[tuple[int, int, str], models.Parameter]()
        for index, use in uses[sub.id].items():
            if not use.only_reads:
                continue
            for op in use.reads:
                read = op.source
                assert isinstance(read, models.InvokeSubroutine)
                address = memory.address(read.args[0])
                assert isinstance(address.base, models.Register)
                key = (index, address.offset, read.target.id)
                parameter = promoted.get(key)
                if parameter is None:
                    (target,) = op.targets
                    parameter = promoted[key] = models.Parameter(
                        name=_unused_name(sub, f"{address.base.name}{_WORD_OF}{len(promoted)}"),
                        version=0,
                        ir_type=target.ir_type,
                        implicit_return=False,
                        source_location=op.source_location,
                    )
                    for site in sites[sub.id]:
                        _read_at(site, index, address.offset, read.target, parameter, versions)
                logger.debug(
                    f"{sub.id}: callers now read the word at"
                    f" {address.base.name} + {address.offset}",
                    location=op.source_location,
                )
                op.source = parameter
        if promoted:
            sub.parameters = [*sub.parameters, *promoted.values()]
            modified = True
    return modified


def _unused_name(sub: models.Subroutine, wanted: str) -> str:
    taken = {register.name for register in sub.get_assigned_registers()}
    name = wanted
    while name in taken:
        name += "_"
    return name


def _read_at(
    site: _CallSite,
    index: int,
    offset: int,
    accessor: models.Subroutine,
    parameter: models.Parameter,
    versions: dict[str, dict[str, int]],
) -> None:
    """Reads the word before the call and passes it."""
    location = site.op.source_location
    taken = versions.get(site.caller.id)
    if taken is None:
        taken = versions[site.caller.id] = {}
        for register in site.caller.get_assigned_registers():
            if register.name.startswith(_NAME_PREFIX):
                taken[register.name] = max(taken.get(register.name, 0), register.version + 1)

    def fresh(name: str, ir_type: models.IRType) -> models.Register:
        name = _NAME_PREFIX + name
        version = taken.get(name, 0)
        taken[name] = version + 1
        return models.Register(
            name=name, version=version, ir_type=ir_type, source_location=location
        )

    new_ops = list[models.Op]()
    address: models.Value = site.call.args[index]
    if offset:
        constant = models.UInt64Constant(value=abs(offset), source_location=location)
        moved = models.Intrinsic(
            op=AVMOp.add if offset > 0 else AVMOp.sub,
            args=[address, constant],
            source_location=location,
        )
        (moved_type,) = moved.types
        address = fresh("at", moved_type)
        new_ops.append(
            models.Assignment(targets=[address], source=moved, source_location=location)
        )
    word = fresh("read", parameter.ir_type)
    new_ops.append(
        models.Assignment(
            targets=[word],
            source=models.InvokeSubroutine(
                target=accessor, args=[address], source_location=location
            ),
            source_location=location,
        )
    )
    at = site.block.ops.index(site.op)
    site.block.ops[at:at] = new_ops
    site.call.args.append(word)


def remove_unused_parameters(context: IROptimizationContext, program: models.Program) -> bool:
    """Parameters that a subroutine does not read are not passed. Limited to programs that gave
    the promise this module works under, so that every other program compiles as before."""
    declared = context.linear_memory
    if declared is None or not declared.fresh_allocations:
        return False
    sites = _call_sites(program)
    modified = False
    for sub in program.subroutines:
        collector = RegisterReadCollector()
        collector.visit_all_blocks(sub.body)
        used = {(register.name, register.version) for register in collector.used_registers}
        unused = [
            index
            for index, parameter in enumerate(sub.parameters)
            if (parameter.name, parameter.version) not in used and not parameter.implicit_return
        ]
        if not unused:
            continue
        logger.debug(f"{sub.id}: {len(unused)} unused parameter(s) are no longer passed")
        for site in sites.get(sub.id, ()):
            for index in reversed(unused):
                del site.call.args[index]
        sub.parameters = [p for i, p in enumerate(sub.parameters) if i not in unused]
        modified = True
    return modified
