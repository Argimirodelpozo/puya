"""What an IR operation or a subroutine can do besides producing its results.

Every kind of effect is a separate fact (`Effect`), so a pass can ask the question it actually
depends on: an operation that writes no state can still fail, and an operation that cannot fail
can still read state that another operation changes.

A flag that is set means "may", a flag that is clear is a guarantee. Anything that is not
classified here may do everything.

Failure is part of the same model. For many AVM ops it depends on the operands, so
`effects_of` takes the bounds known for the values of the subroutine (`ValueBounds`) and
leaves `Effect.fails` out when the operands rule a failure out. An op that uses an account, an
asset, an application or a box can fail because that is not available to the transaction.
Running out of opcode budget is not a failure of the operation that happens to use up the
budget, since optimizations change what a program costs. Code that may never finish is
different: a subroutine with a loop or recursion may fail.

`analyse_subroutine_effects` summarises every subroutine of a program, so a call can be
treated as the union of what its target may do.
"""

import enum
import typing
from collections.abc import Callable, Iterable, Mapping

import attrs
import networkx as nx  # type: ignore[import-untyped]

from puya.avm import AVMType
from puya.ir import models
from puya.ir._utils import get_bytes_constant
from puya.ir.avm_ops import AVMOp
from puya.ir.optimize.value_bounds import (
    MAX_BIGUINT_BYTES,
    MAX_BYTES_LENGTH,
    MAX_UINT64,
    ValueBounds,
)


class Effect(enum.Flag):
    none = 0
    fails = enum.auto()
    """may stop the program with an error"""
    exits = enum.auto()
    """may end the program without an error"""
    reads_scratch = enum.auto()
    writes_scratch = enum.auto()
    reads_global = enum.auto()
    """reads global state of the application that is running"""
    writes_global = enum.auto()
    reads_local = enum.auto()
    """reads local state that the running application keeps in an account"""
    writes_local = enum.auto()
    reads_box = enum.auto()
    """reads the contents of a box"""
    writes_box = enum.auto()
    """changes the contents of a box without changing its size"""
    reads_box_size = enum.auto()
    """depends on whether a box exists, or on its size"""
    allocates_box = enum.auto()
    """may create a box or change the size of one, which also moves the minimum balance of the
    application account"""
    deletes_box = enum.auto()
    """may delete a box, which also moves the minimum balance of the application account"""
    reads_txn = enum.auto()
    """reads what is fixed while the program runs: the transaction, its group, program
    arguments, block headers and the global fields other than the opcode budget"""
    reads_ledger = enum.auto()
    """reads accounts, assets or other applications, including their state and boxes"""
    writes_ledger = enum.auto()
    """changes accounts, assets or other applications, or the parameters of this one"""
    reads_budget = enum.auto()
    """reads the remaining opcode budget, which every operation changes"""
    reads_itxn = enum.auto()
    """reads the results of the last inner transaction group that was submitted"""
    writes_itxn = enum.auto()
    """changes the inner transaction group that is being built"""
    submits_itxn = enum.auto()
    """submits inner transactions, so other programs may run"""
    logs = enum.auto()


EVERYTHING: typing.Final = ~Effect.none
READS: typing.Final = (
    Effect.reads_scratch
    | Effect.reads_global
    | Effect.reads_local
    | Effect.reads_box
    | Effect.reads_box_size
    | Effect.reads_txn
    | Effect.reads_ledger
    | Effect.reads_budget
    | Effect.reads_itxn
)
OBSERVABLE: typing.Final = EVERYTHING & ~READS
"""what is lost if an operation is removed: an operation with none of these, whose results are
unused, can be removed"""
DETERMINISTIC: typing.Final = Effect.fails | Effect.reads_txn
"""an operation with no effect outside these gives the same outcome each time it runs with the
same operands"""

_BOX_ALLOCATION: typing.Final = Effect.allocates_box | Effect.deletes_box
_CHANGED_BY: typing.Final[Mapping[Effect, Effect]] = {
    Effect.reads_scratch: Effect.writes_scratch,
    Effect.reads_global: Effect.writes_global,
    Effect.reads_local: Effect.writes_local,
    Effect.reads_box: Effect.writes_box | _BOX_ALLOCATION,
    Effect.reads_box_size: _BOX_ALLOCATION,
    Effect.reads_txn: Effect.none,
    Effect.reads_ledger: Effect.writes_ledger | _BOX_ALLOCATION,
    Effect.reads_budget: EVERYTHING,
    Effect.reads_itxn: Effect.submits_itxn,
}


def changed_by(reads: Effect) -> Effect:
    """The effects that can change what an operation with the given read effects sees."""
    result = Effect.none
    for read, writes in _CHANGED_BY.items():
        if read & reads:
            result |= writes
    return result


def observers_of(writes: Effect) -> Effect:
    """The read effects of operations that can see a change made by the given effects.

    The opcode budget is left out: every operation changes it, and no optimization keeps it.
    """
    result = Effect.none
    for read, changes in _CHANGED_BY.items():
        if read is not Effect.reads_budget and changes & writes:
            result |= read
    return result


# other programs can run: they can read the state of this application and change the ledger,
# and with shared box access they can change the boxes of this application.
# They cannot call this application, so its global and local state stay as they are.
_INNER_TRANSACTIONS: typing.Final = (
    Effect.submits_itxn
    | Effect.writes_itxn
    | Effect.reads_global
    | Effect.reads_local
    | Effect.reads_box
    | Effect.reads_box_size
    | Effect.reads_ledger
    | Effect.writes_ledger
    | Effect.writes_box
    | _BOX_ALLOCATION
)
_BOX_READ: typing.Final = Effect.reads_box | Effect.reads_box_size
# the application named by an operand can be the one that is running
_OTHER_APP_BOX_WRITE: typing.Final = (
    Effect.reads_box_size | Effect.reads_ledger | Effect.writes_ledger
)

_INTRINSIC_EFFECTS: typing.Final[Mapping[AVMOp, Effect]] = {
    # effects of an op other than failure, which is decided by _MAY_FAIL
    **dict.fromkeys(
        (
            AVMOp.add,
            AVMOp.sub,
            AVMOp.mul,
            AVMOp.div_floor,
            AVMOp.mod,
            AVMOp.exp,
            AVMOp.expw,
            AVMOp.addw,
            AVMOp.mulw,
            AVMOp.divmodw,
            AVMOp.divw,
            AVMOp.sqrt,
            AVMOp.shl,
            AVMOp.shr,
            AVMOp.bitlen,
            AVMOp.not_,
            AVMOp.eq,
            AVMOp.neq,
            AVMOp.lt,
            AVMOp.lte,
            AVMOp.gt,
            AVMOp.gte,
            AVMOp.and_,
            AVMOp.or_,
            AVMOp.bitwise_and,
            AVMOp.bitwise_or,
            AVMOp.bitwise_xor,
            AVMOp.bitwise_not,
            AVMOp.select,
            AVMOp.itob,
            AVMOp.btoi,
            AVMOp.len_,
            AVMOp.concat,
            AVMOp.bzero,
            AVMOp.substring,
            AVMOp.substring3,
            AVMOp.extract,
            AVMOp.extract3,
            AVMOp.extract_uint16,
            AVMOp.extract_uint32,
            AVMOp.extract_uint64,
            AVMOp.replace2,
            AVMOp.replace3,
            AVMOp.getbit,
            AVMOp.setbit,
            AVMOp.getbyte,
            AVMOp.setbyte,
            AVMOp.add_bytes,
            AVMOp.sub_bytes,
            AVMOp.mul_bytes,
            AVMOp.div_floor_bytes,
            AVMOp.mod_bytes,
            AVMOp.bsqrt,
            AVMOp.eq_bytes,
            AVMOp.neq_bytes,
            AVMOp.lt_bytes,
            AVMOp.lte_bytes,
            AVMOp.gt_bytes,
            AVMOp.gte_bytes,
            AVMOp.bitwise_and_bytes,
            AVMOp.bitwise_or_bytes,
            AVMOp.bitwise_xor_bytes,
            AVMOp.bitwise_not_bytes,
            AVMOp.sha256,
            AVMOp.sha3_256,
            AVMOp.sha512,
            AVMOp.sha512_256,
            AVMOp.keccak256,
            AVMOp.sumhash512,
            AVMOp.mimc,
            AVMOp.poseidon2,
            AVMOp.base64_decode,
            AVMOp.json_ref,
            AVMOp.ed25519verify,
            AVMOp.ed25519verify_bare,
            AVMOp.ecdsa_verify,
            AVMOp.ecdsa_pk_decompress,
            AVMOp.ecdsa_pk_recover,
            AVMOp.ec_add,
            AVMOp.ec_map_to,
            AVMOp.ec_multi_scalar_mul,
            AVMOp.ec_pairing_check,
            AVMOp.ec_scalar_mul,
            AVMOp.ec_subgroup_check,
            AVMOp.falcon_verify,
            AVMOp.vrf_verify,
        ),
        Effect.none,
    ),
    **dict.fromkeys(
        (
            AVMOp.txn,
            AVMOp.txna,
            AVMOp.txnas,
            AVMOp.gtxn,
            AVMOp.gtxna,
            AVMOp.gtxnas,
            AVMOp.gtxns,
            AVMOp.gtxnsa,
            AVMOp.gtxnsas,
            # `global OpcodeBudget` is told apart in _intrinsic_effects
            AVMOp.global_,
            AVMOp.arg,
            AVMOp.arg_0,
            AVMOp.arg_1,
            AVMOp.arg_2,
            AVMOp.arg_3,
            AVMOp.args,
            AVMOp.block,
            # what earlier transactions of the group created and left in scratch space
            AVMOp.gaid,
            AVMOp.gaids,
            AVMOp.gload,
            AVMOp.gloads,
            AVMOp.gloadss,
        ),
        Effect.reads_txn,
    ),
    **dict.fromkeys(
        (
            AVMOp.balance,
            AVMOp.min_balance,
            AVMOp.acct_params_get,
            AVMOp.app_params_get,
            AVMOp.asset_params_get,
            AVMOp.asset_holding_get,
            AVMOp.app_opted_in,
            AVMOp.online_stake,
            AVMOp.voter_params_get,
        ),
        Effect.reads_ledger,
    ),
    AVMOp.app_params_set: Effect.writes_ledger,
    # scratch space
    AVMOp.load: Effect.reads_scratch,
    AVMOp.loads: Effect.reads_scratch,
    AVMOp.store: Effect.writes_scratch,
    AVMOp.stores: Effect.writes_scratch,
    # application state; reading the state of application 0, the one that is running, is told
    # apart in _intrinsic_effects
    AVMOp.app_global_get: Effect.reads_global,
    AVMOp.app_global_get_ex: Effect.reads_global | Effect.reads_ledger,
    AVMOp.app_global_put: Effect.writes_global,
    AVMOp.app_global_del: Effect.writes_global,
    AVMOp.app_local_get: Effect.reads_local,
    AVMOp.app_local_get_ex: Effect.reads_local | Effect.reads_ledger,
    AVMOp.app_local_put: Effect.writes_local,
    AVMOp.app_local_del: Effect.writes_local,
    # boxes
    AVMOp.box_get: _BOX_READ,
    AVMOp.box_extract: _BOX_READ,
    AVMOp.box_len: Effect.reads_box_size,
    AVMOp.box_create: Effect.reads_box_size | Effect.allocates_box,
    AVMOp.box_put: Effect.reads_box_size | Effect.writes_box | Effect.allocates_box,
    AVMOp.box_replace: Effect.reads_box_size | Effect.writes_box,
    AVMOp.box_splice: _BOX_READ | Effect.writes_box,
    AVMOp.box_resize: Effect.reads_box_size | Effect.allocates_box,
    AVMOp.box_del: Effect.reads_box_size | Effect.deletes_box,
    AVMOp.app_box_get: _BOX_READ | Effect.reads_ledger,
    AVMOp.app_box_extract: _BOX_READ | Effect.reads_ledger,
    AVMOp.app_box_len: Effect.reads_box_size | Effect.reads_ledger,
    AVMOp.app_box_create: _OTHER_APP_BOX_WRITE | Effect.allocates_box,
    AVMOp.app_box_put: _OTHER_APP_BOX_WRITE | Effect.writes_box | Effect.allocates_box,
    AVMOp.app_box_replace: _OTHER_APP_BOX_WRITE | Effect.writes_box,
    AVMOp.app_box_splice: _OTHER_APP_BOX_WRITE | Effect.reads_box | Effect.writes_box,
    AVMOp.app_box_resize: _OTHER_APP_BOX_WRITE | Effect.allocates_box,
    AVMOp.app_box_del: _OTHER_APP_BOX_WRITE | Effect.deletes_box,
    # logs and inner transactions
    AVMOp.log: Effect.logs,
    AVMOp.itxn_begin: Effect.writes_itxn,
    AVMOp.itxn_next: Effect.writes_itxn,
    AVMOp.itxn_field: Effect.writes_itxn,
    AVMOp.itxn_submit: _INNER_TRANSACTIONS,
    AVMOp.itxn: Effect.reads_itxn,
    AVMOp.itxna: Effect.reads_itxn,
    AVMOp.itxnas: Effect.reads_itxn,
    AVMOp.gitxn: Effect.reads_itxn,
    AVMOp.gitxna: Effect.reads_itxn,
    AVMOp.gitxnas: Effect.reads_itxn,
}

# Reading these fields of the current transaction cannot fail.
# Not listed:
# - FirstValidTime: fails if negative
# - NumLogs, LastLog, CreatedAssetID, CreatedApplicationID: reading these from the current
#   transaction always fails
# - ApprovalProgram, ClearStateProgram: a program can be longer than a stack value may be
NEVER_FAILING_TXN_FIELDS: typing.Final = frozenset(
    {
        "Sender",
        "Fee",
        "FirstValid",
        "LastValid",
        "Note",
        "Lease",
        "Receiver",
        "Amount",
        "CloseRemainderTo",
        "VotePK",
        "SelectionPK",
        "VoteFirst",
        "VoteLast",
        "VoteKeyDilution",
        "Type",
        "TypeEnum",
        "XferAsset",
        "AssetAmount",
        "AssetSender",
        "AssetReceiver",
        "AssetCloseTo",
        "GroupIndex",
        "TxID",
        "ApplicationID",
        "OnCompletion",
        "NumAppArgs",
        "NumAccounts",
        "RekeyTo",
        "ConfigAsset",
        "ConfigAssetTotal",
        "ConfigAssetDecimals",
        "ConfigAssetDefaultFrozen",
        "ConfigAssetUnitName",
        "ConfigAssetName",
        "ConfigAssetURL",
        "ConfigAssetMetadataHash",
        "ConfigAssetManager",
        "ConfigAssetReserve",
        "ConfigAssetFreeze",
        "ConfigAssetClawback",
        "FreezeAsset",
        "FreezeAssetAccount",
        "FreezeAssetFrozen",
        "NumAssets",
        "NumApplications",
        "GlobalNumUint",
        "GlobalNumByteSlice",
        "LocalNumUint",
        "LocalNumByteSlice",
        "ExtraProgramPages",
        "Nonparticipation",
        "StateProofPK",
        "NumApprovalProgramPages",
        "NumClearStateProgramPages",
        "RejectVersion",
    }
)
# Reading these global fields cannot fail.
# Not listed:
# - LatestTimestamp: fails if negative
NEVER_FAILING_GLOBAL_FIELDS: typing.Final = frozenset(
    {
        "MinTxnFee",
        "MinBalance",
        "MaxTxnLife",
        "ZeroAddress",
        "GroupSize",
        "LogicSigVersion",
        "Round",
        "CurrentApplicationID",
        "CreatorAddress",
        "CurrentApplicationAddress",
        "GroupID",
        "OpcodeBudget",
        "CallerApplicationID",
        "CallerApplicationAddress",
        "AssetCreateMinBalance",
        "AssetOptInMinBalance",
        "GenesisHash",
        "PayoutsEnabled",
        "PayoutsGoOnlineFee",
        "PayoutsPercent",
        "PayoutsMinBalance",
        "PayoutsMaxBalance",
    }
)

_MAX_STATE_KEY_LENGTH: typing.Final = 64
_MAX_SCRATCH_SLOT: typing.Final = 255
_MAX_SHIFT: typing.Final = 63

_MayFail = Callable[[models.Intrinsic, ValueBounds], bool]


def _never(_op: models.Intrinsic, _bounds: ValueBounds) -> bool:
    return False


def _add(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    return bounds.uint64(a)[1] + bounds.uint64(b)[1] > MAX_UINT64


def _sub(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    return not bounds.not_above(b, a)


def _mul(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    return bounds.uint64(a)[1] * bounds.uint64(b)[1] > MAX_UINT64


def _zero_divisor(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    _, b = op.args
    return bounds.uint64(b)[0] == 0


def _power(limit: int) -> _MayFail:
    def may_fail(op: models.Intrinsic, bounds: ValueBounds) -> bool:
        a, b = op.args
        a_low, a_high = bounds.uint64(a)
        b_low, b_high = bounds.uint64(b)
        if a_low == 0 and b_low == 0:
            return True
        # 0 and 1 stay as they are, anything to the power of 0 is 1
        if a_high <= 1 or b_high == 0:
            return False
        if b_high > limit.bit_length():
            return True
        return bool(a_high**b_high > limit)

    return may_fail


def _divmodw(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    _, _, c, d = op.args
    return bounds.uint64(c)[0] == 0 and bounds.uint64(d)[0] == 0


def _divw(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, _, c = op.args
    c_low = bounds.uint64(c)[0]
    # the quotient fits in 64 bits only if the high word is below the divisor
    return c_low == 0 or bounds.uint64(a)[1] >= c_low


def _shift(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    _, b = op.args
    return bounds.uint64(b)[1] > _MAX_SHIFT


def _btoi(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    (a,) = op.args
    return bounds.length(a)[1] > 8


def _bzero(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    (a,) = op.args
    return bounds.uint64(a)[1] > MAX_BYTES_LENGTH


def _concat(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    return bounds.length(a)[1] + bounds.length(b)[1] > MAX_BYTES_LENGTH


def _extract(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    (a,) = op.args
    start, length = op.immediates
    if not (isinstance(start, int) and isinstance(length, int)):
        return True
    # a length of 0 takes the rest, so only the start has to be inside
    return start + length > bounds.length(a)[0]


def _extract3(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b, c = op.args
    return bounds.uint64(b)[1] + bounds.uint64(c)[1] > bounds.length(a)[0]


def _substring(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    (a,) = op.args
    start, end = op.immediates
    if not (isinstance(start, int) and isinstance(end, int)):
        return True
    return end < start or end > bounds.length(a)[0]


def _substring3(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b, c = op.args
    c_low, c_high = bounds.uint64(c)
    return bounds.uint64(b)[1] > c_low or c_high > bounds.length(a)[0]


def _extract_uint(num_bytes: int) -> _MayFail:
    def may_fail(op: models.Intrinsic, bounds: ValueBounds) -> bool:
        a, b = op.args
        return bounds.uint64(b)[1] + num_bytes > bounds.length(a)[0]

    return may_fail


def _getbyte(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    return bounds.uint64(b)[1] >= bounds.length(a)[0]


def _setbyte(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b, c = op.args
    return bounds.uint64(b)[1] >= bounds.length(a)[0] or bounds.uint64(c)[1] > 255


def _bit_index_outside(a: models.Value, b: models.Value, bounds: ValueBounds) -> bool:
    match a.ir_type.maybe_avm_type:
        case AVMType.uint64:
            num_bits = 64
        case AVMType.bytes:
            num_bits = 8 * bounds.length(a)[0]
        case _:
            return True
    return bounds.uint64(b)[1] >= num_bits


def _getbit(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    return _bit_index_outside(a, b, bounds)


def _setbit(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b, c = op.args
    return _bit_index_outside(a, b, bounds) or bounds.uint64(c)[1] > 1


def _replace2(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b = op.args
    (start,) = op.immediates
    if not isinstance(start, int):
        return True
    return start + bounds.length(b)[1] > bounds.length(a)[0]


def _replace3(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    a, b, c = op.args
    return bounds.uint64(b)[1] + bounds.length(c)[1] > bounds.length(a)[0]


def _byte_math(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    return any(bounds.length(arg)[1] > MAX_BIGUINT_BYTES for arg in op.args)


def _byte_math_of_constants(holds: Callable[[int, int], bool]) -> _MayFail:
    # the outcome depends on the values and not only on their lengths
    def may_fail(op: models.Intrinsic, _bounds: ValueBounds) -> bool:
        operands = list[int]()
        for arg in op.args:
            constant = get_bytes_constant(arg) if isinstance(arg, models.Constant) else None
            if constant is None or len(constant) > MAX_BIGUINT_BYTES:
                return True
            operands.append(int.from_bytes(constant, "big"))
        a, b = operands
        return not holds(a, b)

    return may_fail


def _same_type_operands(op: models.Intrinsic, _bounds: ValueBounds) -> bool:
    a, b = op.args
    a_type = a.ir_type.maybe_avm_type
    return a_type not in (AVMType.uint64, AVMType.bytes) or a_type != b.ir_type.maybe_avm_type


def _txn_field(op: models.Intrinsic, _bounds: ValueBounds) -> bool:
    (field,) = op.immediates
    return field not in NEVER_FAILING_TXN_FIELDS


def _global_field(op: models.Intrinsic, _bounds: ValueBounds) -> bool:
    (field,) = op.immediates
    return field not in NEVER_FAILING_GLOBAL_FIELDS


def _scratch_slot(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    return bounds.uint64(op.args[0])[1] > _MAX_SCRATCH_SLOT


def _app_global_get(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    (key,) = op.args
    return bounds.length(key)[1] > _MAX_STATE_KEY_LENGTH


def _app_global_get_ex(op: models.Intrinsic, bounds: ValueBounds) -> bool:
    app, key = op.args
    # another application has to be available to the transaction
    return not _is_running_app(app) or bounds.length(key)[1] > _MAX_STATE_KEY_LENGTH


_MAY_FAIL: typing.Final[Mapping[AVMOp, _MayFail]] = {
    # An op that is not listed may fail. The reasons the AVM gives for an op are in the
    # langspec; an op is listed only if every one of them is decided here.
    **dict.fromkeys(
        (
            AVMOp.addw,
            AVMOp.mulw,
            AVMOp.sqrt,
            AVMOp.bitlen,
            AVMOp.not_,
            AVMOp.lt,
            AVMOp.lte,
            AVMOp.gt,
            AVMOp.gte,
            AVMOp.and_,
            AVMOp.or_,
            AVMOp.bitwise_and,
            AVMOp.bitwise_or,
            AVMOp.bitwise_xor,
            AVMOp.bitwise_not,
            AVMOp.select,
            AVMOp.itob,
            AVMOp.len_,
            # unlike byte math, these take inputs of any length
            AVMOp.bitwise_and_bytes,
            AVMOp.bitwise_or_bytes,
            AVMOp.bitwise_xor_bytes,
            AVMOp.bitwise_not_bytes,
            AVMOp.sha256,
            AVMOp.sha3_256,
            AVMOp.sha512,
            AVMOp.sha512_256,
            AVMOp.keccak256,
            AVMOp.sumhash512,
            AVMOp.load,
            AVMOp.store,
        ),
        _never,
    ),
    AVMOp.eq: _same_type_operands,
    AVMOp.neq: _same_type_operands,
    AVMOp.add: _add,
    AVMOp.sub: _sub,
    AVMOp.mul: _mul,
    AVMOp.div_floor: _zero_divisor,
    AVMOp.mod: _zero_divisor,
    AVMOp.exp: _power(MAX_UINT64),
    AVMOp.expw: _power(2**128 - 1),
    AVMOp.divmodw: _divmodw,
    AVMOp.divw: _divw,
    AVMOp.shl: _shift,
    AVMOp.shr: _shift,
    AVMOp.btoi: _btoi,
    AVMOp.bzero: _bzero,
    AVMOp.concat: _concat,
    AVMOp.extract: _extract,
    AVMOp.extract3: _extract3,
    AVMOp.substring: _substring,
    AVMOp.substring3: _substring3,
    AVMOp.extract_uint16: _extract_uint(2),
    AVMOp.extract_uint32: _extract_uint(4),
    AVMOp.extract_uint64: _extract_uint(8),
    AVMOp.getbyte: _getbyte,
    AVMOp.setbyte: _setbyte,
    AVMOp.getbit: _getbit,
    AVMOp.setbit: _setbit,
    AVMOp.replace2: _replace2,
    AVMOp.replace3: _replace3,
    **dict.fromkeys(
        (
            AVMOp.add_bytes,
            AVMOp.mul_bytes,
            AVMOp.bsqrt,
            AVMOp.eq_bytes,
            AVMOp.neq_bytes,
            AVMOp.lt_bytes,
            AVMOp.lte_bytes,
            AVMOp.gt_bytes,
            AVMOp.gte_bytes,
        ),
        _byte_math,
    ),
    AVMOp.sub_bytes: _byte_math_of_constants(lambda a, b: a >= b),
    AVMOp.div_floor_bytes: _byte_math_of_constants(lambda _, b: b != 0),
    AVMOp.mod_bytes: _byte_math_of_constants(lambda _, b: b != 0),
    AVMOp.txn: _txn_field,
    AVMOp.global_: _global_field,
    AVMOp.loads: _scratch_slot,
    AVMOp.stores: _scratch_slot,
    AVMOp.app_global_get: _app_global_get,
    AVMOp.app_global_get_ex: _app_global_get_ex,
}

_CONSTANTS_ONLY: typing.Final = ValueBounds()


@attrs.frozen
class EffectSummaries:
    """What a call to each subroutine of a program may do."""

    _by_id: Mapping[str, Effect]

    def of(self, subroutine: models.Subroutine) -> Effect:
        # a subroutine that was not analysed may do everything
        return self._by_id.get(subroutine.id, EVERYTHING)


def effects_of(
    node: models.IRVisitable,
    *,
    summaries: EffectSummaries | None = None,
    bounds: ValueBounds | None = None,
) -> Effect:
    """What an operation, the source of an assignment or a block terminator may do.

    A call may do everything unless `summaries` is given. Without `bounds`, only constant
    operands rule a failure out.
    """
    match node:
        case models.Assignment(source=source):
            return effects_of(source, summaries=summaries, bounds=bounds)
        case models.Value() | models.ValueTuple() | models.Phi():
            return Effect.none
        case models.Intrinsic():
            return _intrinsic_effects(node, bounds or _CONSTANTS_ONLY)
        case models.InvokeSubroutine(target=target):
            return EVERYTHING if summaries is None else summaries.of(target)
        case models.Assert(condition=models.UInt64Constant(value=value)) if value:
            return Effect.none
        case models.Assert():
            return Effect.fails
        case models.ReadSlot():
            return Effect.reads_scratch
        case models.WriteSlot():
            return Effect.writes_scratch
        case models.NewSlot():
            # allocation keeps its bookkeeping in scratch space, and fails when it runs out
            return Effect.reads_scratch | Effect.writes_scratch | Effect.fails
        case models.BoxRead():
            # fails if the box does not exist
            return _BOX_READ | Effect.fails
        case models.BoxWrite(delete_first=delete_first):
            effects = _INTRINSIC_EFFECTS[AVMOp.box_put] | Effect.fails
            if delete_first:
                effects |= _INTRINSIC_EFFECTS[AVMOp.box_del]
            return effects
        case models.InnerTransactionField(array_index=None):
            # only built for a transaction of the group that was just submitted
            return Effect.reads_itxn
        case models.InnerTransactionField():
            return Effect.reads_itxn | Effect.fails
        case (
            models.ExtractValue()
            | models.ReplaceValue()
            | models.DecodeBytes()
            | models.BytesEncode()
            | models.ArrayLength()
            | models.ArrayPop()
            | models.ArrayConcat()
        ):
            # these fail on an index out of bounds, on a value that does not fit its encoding
            # and on a value that is not encoded as its type says
            return Effect.fails
        case models.Fail():
            return Effect.fails
        case models.ProgramExit():
            return Effect.exits
        case (
            models.Goto()
            | models.ConditionalBranch()
            | models.GotoNth()
            | models.Switch()
            | models.SubroutineReturn()
        ):
            return Effect.none
        case _:
            return EVERYTHING


def lost_if_removed(
    node: models.IRVisitable,
    *,
    summaries: EffectSummaries | None,
    bounds: Callable[[], ValueBounds],
) -> Effect:
    """What a program would lose if an operation whose results are unused were removed.

    `bounds` is only called when it is the operands that decide.
    """
    lost = effects_of(node, summaries=summaries) & OBSERVABLE
    if lost == Effect.fails:
        lost = effects_of(node, summaries=summaries, bounds=bounds()) & OBSERVABLE
    return lost


def _intrinsic_effects(op: models.Intrinsic, bounds: ValueBounds) -> Effect:
    try:
        effects = _INTRINSIC_EFFECTS[op.op]
    except KeyError:
        return EVERYTHING
    match op:
        case models.Intrinsic(op=AVMOp.global_, immediates=["OpcodeBudget"]):
            effects = Effect.reads_budget
        case models.Intrinsic(op=AVMOp.app_global_get_ex, args=[app, _]) if _is_running_app(app):
            effects = Effect.reads_global
        case models.Intrinsic(op=AVMOp.app_local_get_ex, args=[_, app, _]) if _is_running_app(app):
            effects = Effect.reads_local
    may_fail = _MAY_FAIL.get(op.op)
    if may_fail is None or may_fail(op, bounds):
        effects |= Effect.fails
    return effects


def _is_running_app(app: models.Value) -> bool:
    # an application operand of 0 names the application that is running
    return isinstance(app, models.UInt64Constant) and app.value == 0


def analyse_subroutine_effects(program: models.Program) -> EffectSummaries:
    """Summarise what a call to each subroutine may do: what its own operations may do, and
    what the subroutines it calls may do.

    Subroutines that call each other, directly or not, share one summary: the union over all
    of them, which is the least summary that is closed under their calls.
    """
    local = dict[str, Effect]()
    graph = nx.DiGraph()
    for subroutine in program.all_subroutines:
        local[subroutine.id], callees = _local_effects(subroutine)
        graph.add_node(subroutine.id)
        for callee in callees:
            graph.add_edge(subroutine.id, callee)
    condensed = nx.condensation(graph)
    summaries = dict[str, Effect]()
    component_effects = dict[int, Effect]()
    # every call leads to a later component, so going backwards, callees come first
    for component in reversed(list(nx.topological_sort(condensed))):
        members: Iterable[str] = condensed.nodes[component]["members"]
        effects = Effect.none
        for callee_component in condensed.successors(component):
            effects |= component_effects[callee_component]
        recursive = False
        for member in members:
            # a callee that is not part of the program may do everything
            effects |= local.get(member, EVERYTHING)
            recursive = recursive or graph.has_edge(member, member)
        if recursive or len(condensed.nodes[component]["members"]) > 1:
            # recursion that does not end runs out of opcode budget
            effects |= Effect.fails
        component_effects[component] = effects
        for member in members:
            summaries[member] = effects
    return EffectSummaries(summaries)


def _local_effects(subroutine: models.Subroutine) -> tuple[Effect, set[str]]:
    bounds = ValueBounds.of(subroutine)
    effects = Effect.none
    callees = set[str]()
    for block in subroutine.body:
        block_effects = Effect.none
        block_callees = set[str]()
        for op in block.ops:
            node = op.source if isinstance(op, models.Assignment) else op
            if isinstance(node, models.InvokeSubroutine):
                block_callees.add(node.target.id)
            else:
                block_effects |= effects_of(node, bounds=bounds)
        if isinstance(block.terminator, models.Fail) and not block_callees:
            # everything a failing program did is undone, so its failure is all that is left.
            # A call could end the program before the failure is reached.
            block_effects = Effect.fails
        elif block.terminator is not None:
            block_effects |= effects_of(block.terminator)
        effects |= block_effects
        callees |= block_callees
    if _has_loop(subroutine):
        # a loop that does not end runs out of opcode budget
        effects |= Effect.fails
    return effects, callees


def _has_loop(subroutine: models.Subroutine) -> bool:
    # depth first, a block reached again while it is still being explored closes a loop
    unvisited, exploring, done = 0, 1, 2
    state = dict.fromkeys(subroutine.body, unvisited)
    for start in subroutine.body:
        if state[start] != unvisited:
            continue
        state[start] = exploring
        stack = [(start, iter(start.successors))]
        while stack:
            block, successors = stack[-1]
            for successor in successors:
                successor_state = state.get(successor, done)
                if successor_state == exploring:
                    return True
                if successor_state == unvisited:
                    state[successor] = exploring
                    stack.append((successor, iter(successor.successors)))
                    break
            else:
                state[block] = done
                stack.pop()
    return False
