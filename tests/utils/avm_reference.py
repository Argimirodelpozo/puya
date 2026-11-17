"""A reference implementation of the AVM ops that compute values, to check what optimizations
assume about them: what an op gives for its operands, and when it fails."""

import math
import random
from collections.abc import Callable, Sequence

import attrs

from puya.ir.avm_ops import AVMOp
from puya.ir.types_ import IRType, PrimitiveIRType

MAX_UINT64 = 2**64 - 1
MAX_BYTES = 4096
_MAX_MATH_BYTES = 64

Operand = int | bytes


class OpFailsError(Exception):
    pass


def _check(condition: bool) -> None:  # noqa: FBT001
    if not condition:
        raise OpFailsError


def _fit(value: int, limit: int = MAX_UINT64) -> int:
    _check(0 <= value <= limit)
    return value


def _number(value: bytes) -> int:
    _check(len(value) <= _MAX_MATH_BYTES)
    return int.from_bytes(value, "big")


def _digits(value: int) -> bytes:
    _check(value >= 0)
    return value.to_bytes((value.bit_length() + 7) // 8, "big")


def _data(value: bytes) -> bytes:
    _check(len(value) <= MAX_BYTES)
    return value


def _exp(a: int, b: int, limit: int) -> int:
    _check(not (a == 0 and b == 0))
    if a <= 1 or b == 0:
        return a**b if b else 1
    _check(b <= limit.bit_length())
    return _fit(a**b, limit)


def _extract(a: bytes, start: int, length: int) -> bytes:
    _check(start + length <= len(a))
    return a[start : start + length]


def _bit(a: Operand, index: int) -> int:
    if isinstance(a, int):
        _check(index < 64)
        return (a >> index) & 1
    _check(index < 8 * len(a))
    return (a[index // 8] >> (7 - index % 8)) & 1


def _set_bit(a: Operand, index: int, bit: int) -> Operand:
    _check(bit <= 1)
    if isinstance(a, int):
        _check(index < 64)
        return (a & ~(1 << index)) | (bit << index)
    _check(index < 8 * len(a))
    result = bytearray(a)
    mask = 1 << (7 - index % 8)
    result[index // 8] = (result[index // 8] & ~mask) | (mask if bit else 0)
    return bytes(result)


def _replace(a: bytes, start: int, b: bytes) -> bytes:
    _check(start + len(b) <= len(a))
    return a[:start] + b + a[start + len(b) :]


def _bitwise(a: bytes, b: bytes, operation: Callable[[int, int], int]) -> bytes:
    size = max(len(a), len(b))
    return bytes(
        operation(x, y)
        for x, y in zip(a.rjust(size, b"\x00"), b.rjust(size, b"\x00"), strict=True)
    )


@attrs.frozen
class ReferenceOp:
    op: AVMOp
    operands: str
    """a letter per operand: u for a uint64, b for a byte array"""
    reference: Callable[..., Operand]
    """takes the immediates and then the operands, raises OpFailsError where the op fails"""
    immediates: Callable[[random.Random], Sequence[int]] = lambda _: ()
    result: IRType = PrimitiveIRType.uint64
    decided_by_bounds: bool = True
    """False if bounds on the operands are never enough to tell that the op does not fail"""

    def __str__(self) -> str:
        return f"{self.op.code}({self.operands})"

    def run(self, immediates: Sequence[int], operands: Sequence[Operand]) -> Operand | None:
        """The result of the op, or None if the op fails."""
        try:
            return self.reference(*immediates, *operands)
        except OpFailsError:
            return None


def _small(rng: random.Random) -> int:
    return rng.choice([0, 1, 2, 7, 8, 9, 31, 32, 33, 64, 65, 255])


_bytes = PrimitiveIRType.bytes
REFERENCE_OPS = [
    ReferenceOp(AVMOp.add, "uu", lambda a, b: _fit(a + b)),
    ReferenceOp(AVMOp.sub, "uu", lambda a, b: _fit(a - b)),
    ReferenceOp(AVMOp.mul, "uu", lambda a, b: _fit(a * b)),
    ReferenceOp(AVMOp.div_floor, "uu", lambda a, b: (_check(b != 0), a // max(b, 1))[1]),
    ReferenceOp(AVMOp.mod, "uu", lambda a, b: (_check(b != 0), a % max(b, 1))[1]),
    ReferenceOp(AVMOp.exp, "uu", lambda a, b: _exp(a, b, MAX_UINT64)),
    ReferenceOp(AVMOp.shl, "uu", lambda a, b: (_check(b <= 63), (a << min(b, 63)) % 2**64)[1]),
    ReferenceOp(AVMOp.shr, "uu", lambda a, b: (_check(b <= 63), a >> min(b, 63))[1]),
    ReferenceOp(AVMOp.sqrt, "u", math.isqrt),
    ReferenceOp(AVMOp.bitlen, "u", lambda a: a.bit_length()),
    ReferenceOp(AVMOp.bitlen, "b", lambda a: int.from_bytes(a, "big").bit_length()),
    ReferenceOp(AVMOp.not_, "u", lambda a: int(a == 0)),
    ReferenceOp(AVMOp.lt, "uu", lambda a, b: int(a < b)),
    ReferenceOp(AVMOp.gte, "uu", lambda a, b: int(a >= b)),
    ReferenceOp(AVMOp.eq, "uu", lambda a, b: int(a == b)),
    ReferenceOp(AVMOp.and_, "uu", lambda a, b: int(bool(a and b))),
    ReferenceOp(AVMOp.bitwise_and, "uu", lambda a, b: a & b),
    ReferenceOp(AVMOp.bitwise_or, "uu", lambda a, b: a | b),
    ReferenceOp(AVMOp.bitwise_xor, "uu", lambda a, b: a ^ b),
    ReferenceOp(AVMOp.bitwise_not, "u", lambda a: a ^ MAX_UINT64),
    ReferenceOp(
        AVMOp.divw,
        "uuu",
        lambda a, b, c: (_check(c != 0 and a < c), ((a << 64) | b) // max(c, 1))[1],
    ),
    ReferenceOp(AVMOp.select, "uuu", lambda a, b, c: b if c else a),
    ReferenceOp(AVMOp.select, "bbu", lambda a, b, c: b if c else a, result=_bytes),
    ReferenceOp(AVMOp.itob, "u", lambda a: a.to_bytes(8, "big"), result=_bytes),
    ReferenceOp(AVMOp.btoi, "b", lambda a: (_check(len(a) <= 8), int.from_bytes(a[:8], "big"))[1]),
    ReferenceOp(AVMOp.len_, "b", len),
    ReferenceOp(AVMOp.concat, "bb", lambda a, b: _data(a + b), result=_bytes),
    ReferenceOp(
        AVMOp.bzero,
        "u",
        lambda a: (_check(a <= MAX_BYTES), bytes(min(a, MAX_BYTES)))[1],
        result=_bytes,
    ),
    ReferenceOp(
        AVMOp.extract,
        "b",
        lambda start, length, a: (
            _extract(a, start, length) if length else _extract(a, start, max(len(a) - start, 0))
        ),
        immediates=lambda rng: (_small(rng), _small(rng)),
        result=_bytes,
    ),
    ReferenceOp(AVMOp.extract3, "buu", _extract, result=_bytes),
    ReferenceOp(
        AVMOp.substring,
        "b",
        lambda start, end, a: (_check(start <= end), _extract(a, start, max(end - start, 0)))[1],
        immediates=lambda rng: (_small(rng), _small(rng)),
        result=_bytes,
    ),
    ReferenceOp(
        AVMOp.substring3,
        "buu",
        lambda a, b, c: (_check(b <= c), _extract(a, b, max(c - b, 0)))[1],
        result=_bytes,
    ),
    ReferenceOp(AVMOp.extract_uint16, "bu", lambda a, b: int.from_bytes(_extract(a, b, 2), "big")),
    ReferenceOp(AVMOp.extract_uint32, "bu", lambda a, b: int.from_bytes(_extract(a, b, 4), "big")),
    ReferenceOp(AVMOp.extract_uint64, "bu", lambda a, b: int.from_bytes(_extract(a, b, 8), "big")),
    ReferenceOp(AVMOp.getbyte, "bu", lambda a, b: _extract(a, b, 1)[0]),
    ReferenceOp(
        AVMOp.setbyte,
        "buu",
        lambda a, b, c: (_check(c <= 255), _replace(a, b, bytes([min(c, 255)])))[1],
        result=_bytes,
    ),
    ReferenceOp(AVMOp.getbit, "uu", _bit),
    ReferenceOp(AVMOp.getbit, "bu", _bit),
    ReferenceOp(AVMOp.setbit, "uuu", _set_bit),
    ReferenceOp(AVMOp.setbit, "buu", _set_bit, result=_bytes),
    ReferenceOp(
        AVMOp.replace2,
        "bb",
        lambda start, a, b: _replace(a, start, b),
        immediates=lambda rng: (_small(rng),),
        result=_bytes,
    ),
    ReferenceOp(AVMOp.replace3, "bub", _replace, result=_bytes),
    ReferenceOp(
        AVMOp.add_bytes, "bb", lambda a, b: _digits(_number(a) + _number(b)), result=_bytes
    ),
    ReferenceOp(
        AVMOp.sub_bytes,
        "bb",
        lambda a, b: _digits(_number(a) - _number(b)),
        result=_bytes,
        decided_by_bounds=False,
    ),
    ReferenceOp(
        AVMOp.mul_bytes, "bb", lambda a, b: _digits(_number(a) * _number(b)), result=_bytes
    ),
    ReferenceOp(
        AVMOp.div_floor_bytes,
        "bb",
        lambda a, b: (_check(_number(b) != 0), _digits(_number(a) // max(_number(b), 1)))[1],
        result=_bytes,
        decided_by_bounds=False,
    ),
    ReferenceOp(
        AVMOp.mod_bytes,
        "bb",
        lambda a, b: (_check(_number(b) != 0), _digits(_number(a) % max(_number(b), 1)))[1],
        result=_bytes,
        decided_by_bounds=False,
    ),
    ReferenceOp(AVMOp.bsqrt, "b", lambda a: _digits(math.isqrt(_number(a))), result=_bytes),
    ReferenceOp(AVMOp.lt_bytes, "bb", lambda a, b: int(_number(a) < _number(b))),
    ReferenceOp(AVMOp.eq_bytes, "bb", lambda a, b: int(_number(a) == _number(b))),
    ReferenceOp(
        AVMOp.bitwise_or_bytes, "bb", lambda a, b: _bitwise(a, b, int.__or__), result=_bytes
    ),
    ReferenceOp(
        AVMOp.bitwise_and_bytes, "bb", lambda a, b: _bitwise(a, b, int.__and__), result=_bytes
    ),
    ReferenceOp(
        AVMOp.bitwise_xor_bytes, "bb", lambda a, b: _bitwise(a, b, int.__xor__), result=_bytes
    ),
    ReferenceOp(AVMOp.bitwise_not_bytes, "b", lambda a: bytes(x ^ 255 for x in a), result=_bytes),
    ReferenceOp(AVMOp.sha256, "b", lambda _: bytes(32), result=_bytes),
    ReferenceOp(AVMOp.sha512, "b", lambda _: bytes(64), result=_bytes),
]

_UINT64_VALUES = [
    *range(10),
    *(31, 32, 33, 62, 63, 64, 65, 127, 128, 255, 256, 257, 4095, 4096, 4097),
    *(2**16 - 1, 2**16, 2**32 - 1, 2**32, 2**32 + 1, 2**63 - 1, 2**63),
    *(MAX_UINT64 - 64, MAX_UINT64 - 1, MAX_UINT64),
]
_LENGTHS = [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 65, 100, 2048, 4000, 4095, 4096]


def random_uint64(rng: random.Random, earlier: Sequence[Operand] = ()) -> int:
    """A uint64 operand: one of the values where ops start to behave differently, a value
    that is related to an earlier operand in such a way, or any other value."""
    choice = rng.random()
    if earlier and choice < 0.5:
        # around where an op on this and an earlier operand starts to fail
        operand = rng.choice(earlier)
        if isinstance(operand, bytes):
            value = len(operand) + rng.choice([-9, -8, -7, -5, -4, -3, -2, -1, 0, 0, 1, 2])
        else:
            value = rng.choice(
                [
                    operand,
                    operand + 1,
                    operand - 1,
                    MAX_UINT64 - operand,
                    MAX_UINT64 - operand + 1,
                    MAX_UINT64 // max(operand, 1),
                    MAX_UINT64 // max(operand, 1) + 1,
                ]
            )
        return min(max(value, 0), MAX_UINT64)
    if choice < 0.9:
        return rng.choice(_UINT64_VALUES)
    return rng.getrandbits(rng.choice([4, 8, 16, 32, 64]))


def random_bytes(rng: random.Random, earlier: Sequence[Operand] = ()) -> bytes:
    """A byte array operand, of a length chosen like the values of `random_uint64`."""
    choice = rng.random()
    if earlier and choice < 0.3:
        # what is left of an earlier byte array after an earlier position, give or take
        length = max((len(o) for o in earlier if isinstance(o, bytes)), default=0)
        length -= min((o for o in earlier if isinstance(o, int)), default=0)
        length = min(max(length + rng.choice([-1, 0, 0, 1]), 0), MAX_BYTES)
    elif choice < 0.85:
        length = rng.choice(_LENGTHS)
    else:
        length = rng.randrange(MAX_BYTES + 1)
    fill = rng.choice([0, 1, 255, None])
    if fill is None:
        return rng.randbytes(length)
    return bytes([fill]) * length
