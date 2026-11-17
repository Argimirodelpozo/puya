import enum
import functools
import typing
from collections import defaultdict
from collections.abc import Callable, Sequence

import attrs

from puya import log
from puya.avm import AVMType
from puya.ir import models
from puya.ir._utils import get_bytes_constant
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize._utils import HasHighLevelOps
from puya.ir.optimize.context import IROptimizationContext
from puya.ir.optimize.effects import Effect, EffectSummaries, effects_of, observers_of
from puya.ir.optimize.value_bounds import ValueBounds
from puya.ir.types_ import PrimitiveIRType
from puya.ir.visitor import NoOpIRVisitor

logger = log.get_logger(__name__)


def constant_reads_and_unobserved_writes_elimination(
    context: IROptimizationContext, subroutine: models.Subroutine
) -> bool:
    if HasHighLevelOps.check(subroutine.body):
        return False

    @functools.cache
    def bounds() -> ValueBounds:
        return ValueBounds.of(subroutine)

    any_modified = False
    for block in subroutine.body:
        while _StateTrackingVisitor.optimise(block, context.effects, bounds):
            any_modified = True
    return any_modified


@enum.unique
class _StateType(enum.Enum):
    box = enum.auto()
    app_global = enum.auto()
    app_local = enum.auto()
    slot = enum.auto()


# the effects that change each type of state
_WRITES: typing.Final = {
    _StateType.box: Effect.writes_box | Effect.allocates_box | Effect.deletes_box,
    _StateType.app_global: Effect.writes_global,
    _StateType.app_local: Effect.writes_local,
    _StateType.slot: Effect.writes_scratch,
}
# the effects of operations that can tell whether a write to each type of state has happened
_OBSERVES: typing.Final = {typ: observers_of(writes) for typ, writes in _WRITES.items()}

_ReadType = typing.Literal["rw", "len", "extract"]

_StateKey = tuple[models.Value | bytes | _ReadType, ...]


@attrs.define(kw_only=True)
class _StateTrackingVisitor(NoOpIRVisitor[None]):
    modified: bool = attrs.field(default=False, init=False)
    _block: models.BasicBlock
    _summaries: EffectSummaries | None
    _bounds: Callable[[], ValueBounds]
    _read_results: defaultdict[_StateType, dict[_StateKey, Sequence[models.Value]]] = attrs.field(
        factory=lambda: defaultdict(dict)
    )
    _last_write: dict[_StateType, tuple[_StateKey, models.Op]] = attrs.field(factory=dict)

    @classmethod
    def optimise(
        cls,
        block: models.BasicBlock,
        summaries: EffectSummaries | None,
        bounds: Callable[[], ValueBounds],
    ) -> bool:
        visitor = cls(block=block, summaries=summaries, bounds=bounds)
        # iterate over a copy b.c. visiting may
        # remove ops. (see _handle_write() below)
        for op in block.ops.copy():
            op.accept(visitor)
        return visitor.modified

    def _cached_read(
        self,
        assignment: models.Assignment,
        typ: _StateType,
        key: tuple[models.Value | _ReadType, ...],
        *,
        allow_truncated_reads: bool = False,
    ) -> None:
        # doing a read resets the unobserved-writes cache
        self._last_write.pop(typ, None)

        normalized_key = _normalize_key(key)
        typ_read_results = self._read_results[typ]
        try:
            last_read = typ_read_results[normalized_key]
        except KeyError:
            typ_read_results[normalized_key] = assignment.targets
            return

        if len(last_read) == len(assignment.targets) or (
            allow_truncated_reads and len(last_read) > len(assignment.targets)
        ):
            logger.debug(
                f"replacing {typ.name} read with cached value for key: {_key_str(key)}",
                location=assignment.source.source_location,
            )
            new_source: models.ValueProvider
            try:
                (new_source,) = last_read
            except ValueError:
                new_source = models.ValueTuple(
                    values=last_read[: len(assignment.targets)],
                    source_location=None,
                )
            assignment.source = new_source
            self.modified = True

    def _handle_write(
        self,
        op: models.Op,
        typ: _StateType,
        key: tuple[models.Value | _ReadType, ...],
        values: tuple[models.Value, ...],
    ) -> None:
        # update cache with new value.
        # this is conservative to naively avoid the problem when multiple
        # values point to the same slot
        normalized_key = _normalize_key(key)
        self._read_results[typ] = {normalized_key: values}
        try:
            last_write_key, last_write_op = self._last_write[typ]
        except KeyError:
            pass
        else:
            if last_write_key == normalized_key and self._is_redundant(last_write_op, op):
                # remove last_write if it is overwritten with another write, there
                # are no intervening reads, and the program ends the same way without it
                logger.debug(
                    f"removing unobserved {typ.name} write to key: {_key_str(key)}",
                    location=last_write_op.source_location,
                )
                self._block.ops.remove(last_write_op)
                self.modified = True
        self._last_write[typ] = (normalized_key, op)

    def _is_redundant(self, first: models.Op, second: models.Op) -> bool:
        """Given two writes to the same key with nothing in between that can tell if the first
        has happened: True if the second alone succeeds exactly when both do. Then both ways
        leave the same state, and the first write can go."""
        match first, second:
            case models.WriteSlot(), models.WriteSlot():
                return True
            case (
                (
                    models.Intrinsic(op=AVMOp.app_global_put, args=[_, a]),
                    models.Intrinsic(op=AVMOp.app_global_put, args=[_, b]),
                )
                | (
                    models.Intrinsic(op=AVMOp.app_local_put, args=[_, _, a]),
                    models.Intrinsic(op=AVMOp.app_local_put, args=[_, _, b]),
                )
            ):
                return self._fits_where_fits(a, b)
            case (
                models.Intrinsic(op=AVMOp.box_put, args=[_, a]),
                models.Intrinsic(op=AVMOp.box_put, args=[_, b]),
            ):
                return self._same_length(a, b)
            case (
                models.BoxWrite(value=a, delete_first=a_delete_first),
                models.BoxWrite(value=b, delete_first=b_delete_first),
            ):
                # Writing over a box fails if the size differs, writing after a delete does not.
                # A larger value in between can also exceed what the transaction may write.
                return a_delete_first == b_delete_first and self._same_length(a, b)
            case _:
                return False

    def _same_length(self, a: models.Value, b: models.Value) -> bool:
        # A box takes a value of the size it has, or any value if it is created by the write.
        # With equal sizes, the first write fails exactly when the second alone does, and the
        # size of the box, which the minimum balance follows, is the same after each of them.
        a_min, a_max = self._bounds().length(a)
        return a_min == a_max and (a_min, a_max) == self._bounds().length(b)

    def _fits_where_fits(self, a: models.Value, b: models.Value) -> bool:
        # A state value has a type, counted against the schema, and a size limit shared with
        # its key. The first write must not hit a limit that the second stays within.
        match a.ir_type.maybe_avm_type, b.ir_type.maybe_avm_type:
            case AVMType.uint64, AVMType.uint64:
                return True
            case AVMType.bytes, AVMType.bytes:
                return self._bounds().length(a)[1] <= self._bounds().length(b)[0]
            case _:
                return False

    def _invalidate(self, typ: _StateType) -> None:
        self._read_results[typ].clear()
        self._last_write.pop(typ, None)

    def _apply(self, effects: Effect) -> None:
        """Forget what an operation with the given effects may have changed, and keep the
        writes it may have seen."""
        for typ in _StateType:
            if effects & _WRITES[typ]:
                self._invalidate(typ)
            elif effects & _OBSERVES[typ]:
                self._last_write.pop(typ, None)
        if Effect.exits in effects:
            # the program may end here, with every write so far in place
            self._last_write.clear()

    @typing.override
    def visit_assignment(self, op: models.Assignment) -> None:
        match op.source:
            # SLOTS
            case models.ReadSlot(slot=slot):
                self._cached_read(op, _StateType.slot, key=(slot,))
            # GLOBAL STATE
            case models.Intrinsic(
                op=AVMOp.app_global_get_ex, args=[models.UInt64Constant(value=0), key]
            ):
                self._cached_read(op, _StateType.app_global, key=(key,))
            case models.Intrinsic(op=AVMOp.app_global_get, args=[key]):
                self._cached_read(
                    op, _StateType.app_global, key=(key,), allow_truncated_reads=True
                )
            # LOCAL STATE
            case models.Intrinsic(
                op=AVMOp.app_local_get_ex, args=[account, models.UInt64Constant(value=0), key]
            ):
                self._cached_read(op, _StateType.app_local, key=(account, key))
            case models.Intrinsic(op=AVMOp.app_local_get, args=[account, key]):
                self._cached_read(
                    op, _StateType.app_local, key=(account, key), allow_truncated_reads=True
                )
            # BOX
            case models.Intrinsic(op=AVMOp.box_get, args=[key]) | models.BoxRead(key=key):
                self._cached_read(op, _StateType.box, key=("rw", key))
            case models.Intrinsic(op=AVMOp.box_len, args=[key]):
                self._cached_read(op, _StateType.box, key=("len", key))
            case models.Intrinsic(op=AVMOp.box_extract, args=[key, offset, length]):
                self._cached_read(op, _StateType.box, key=("extract", key, offset, length))
            # OTHER
            case other_source:
                # visit in case invalidations need to occur
                other_source.accept(self)

    @typing.override
    def visit_box_write(self, write: models.BoxWrite) -> None:
        self._handle_box_put_or_write(write, key=write.key, value=write.value)

    @typing.override
    def visit_write_slot(self, write: models.WriteSlot) -> None:
        self._handle_write(op=write, typ=_StateType.slot, key=(write.slot,), values=(write.value,))

    @typing.override
    def visit_intrinsic_op(self, op: models.Intrinsic) -> None:
        match op.op:
            # GLOBAL STATE
            case AVMOp.app_global_put:
                key, value = op.args
                self._handle_write(
                    op, _StateType.app_global, key=(key,), values=(value, _const_true())
                )
            # LOCAL STATE
            case AVMOp.app_local_put:
                (account, key, value) = op.args
                self._handle_write(
                    op, _StateType.app_local, key=(account, key), values=(value, _const_true())
                )
            # BOX
            case AVMOp.box_put:
                key, value = op.args
                self._handle_box_put_or_write(op, key=key, value=value)
            case AVMOp.box_create:
                self._invalidate(_StateType.box)
                key, len_ = op.args
                self._cache_box_len(key=key, length=len_)
            # SLOTS
            case AVMOp.load | AVMOp.store:
                # a scratch slot that an op names is reserved, so it is never a slot
                # that the reads and writes followed here refer to
                pass
            # OTHER
            case _:
                # Among others, this forgets:
                # - global and local state at a delete
                # - boxes at any other box op, including one on an application given by id,
                #   which can be this one
                # - boxes at an inner transaction: another application may share box access
                # and keeps the last write to state that the op may have seen, as an inner
                # transaction may see all of it, and a minimum balance follows the boxes
                self._apply(effects_of(op))

    @typing.override
    def visit_invoke_subroutine(self, callsub: models.InvokeSubroutine) -> None:
        self._apply(effects_of(callsub, summaries=self._summaries))

    def _handle_box_put_or_write(
        self, op: models.Intrinsic | models.BoxWrite, *, key: models.Value, value: models.Value
    ) -> None:
        self._handle_write(op, _StateType.box, key=("rw", key), values=(value, _const_true()))
        if isinstance(value, models.Constant):
            bytes_const = get_bytes_constant(value)
            if bytes_const is not None:
                bytes_const_len = models.UInt64Constant(
                    value=len(bytes_const), source_location=op.source_location
                )
                self._cache_box_len(key=key, length=bytes_const_len)

    def _cache_box_len(self, *, key: models.Value, length: models.Value) -> None:
        len_key = _normalize_key(("len", key))
        len_result = (length, _const_true())
        self._read_results[_StateType.box][len_key] = len_result


def _normalize_key(keys: tuple[models.Value | _ReadType, ...]) -> _StateKey:
    state_key = list[models.Value | bytes | _ReadType]()
    for key in keys:
        if (
            isinstance(key, models.Constant)
            and (bytes_const := get_bytes_constant(key)) is not None
        ):
            state_key.append(bytes_const)
        else:
            state_key.append(key)
    return tuple(state_key)


def _key_str(key: tuple[models.Value | _ReadType, ...]) -> str:
    try:
        (single_key,) = key
    except ValueError:
        return f"({','.join(map(str, key))})"
    else:
        return str(single_key)


def _const_true() -> models.Value:
    return models.UInt64Constant(
        value=1,
        ir_type=PrimitiveIRType.bool,
        source_location=None,
    )
