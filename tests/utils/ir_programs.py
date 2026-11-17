"""Made-up subroutines in SSA form and a way to run them, for tests of what optimizations
assume about a whole subroutine: values that depend on asserts, branches and loops.

A program is a sequence of statements over a few uint64 variables and a few byte array
variables: assignments of AVM ops, asserts, if/else and while loops, and the sequences that a
frontend emits around arithmetic on integers that are wider than 64 bits (a check of the
number of bits after an addition, a modulo to wrap around, a pad to a full word).
"""

import random
import typing
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import attrs

from puya.ir import models
from puya.ir.avm_ops import AVMOp
from puya.ir.types_ import IRType, PrimitiveIRType
from puya.parse import SourceLocation
from tests.utils.avm_reference import REFERENCE_OPS, Operand, OpFailsError
from tests.utils.ir import assert_, assign, biguint, branch, bytes_, goto, intrinsic, phi, uint64

_UINT64 = PrimitiveIRType.uint64
_BYTES = PrimitiveIRType.bytes
_BOOL = PrimitiveIRType.bool
_LOCATION = SourceLocation(file=Path("/test_ir_programs.py"), line=1)
_WORD = 2**256
_MAX_MATH_BYTES = 64

_UINT64_CONSTANTS = [0, 1, 2, 3, 7, 8, 10, 31, 32, 63, 64, 100, 255, 256, 2**32, 2**63, 2**64 - 1]
_NUMBERS = [0, 1, 2, 255, 256, 2**64 - 1, 2**64, 2**128, _WORD - 1, _WORD]
_BYTES_CONSTANTS = [
    b"",
    bytes(1),
    bytes(8),
    bytes(32),
    b"\x01",
    b"\x00\x01",
    b"\xff" * 8,
    b"\xff" * 32,
    bytes(31) + b"\x05",
]
_UINT64_COMPARISONS = [AVMOp.lt, AVMOp.lte, AVMOp.gt, AVMOp.gte, AVMOp.eq, AVMOp.neq]
_NUMBER_COMPARISONS = [
    AVMOp.lt_bytes,
    AVMOp.lte_bytes,
    AVMOp.gt_bytes,
    AVMOp.gte_bytes,
    AVMOp.eq_bytes,
    AVMOp.neq_bytes,
]
_BYTE_MATH = [
    AVMOp.add_bytes,
    AVMOp.sub_bytes,
    AVMOp.mul_bytes,
    AVMOp.div_floor_bytes,
    AVMOp.mod_bytes,
]


def _number(value: bytes) -> int:
    if len(value) > _MAX_MATH_BYTES:
        raise OpFailsError
    return int.from_bytes(value, "big")


_REFERENCE: typing.Final[Mapping[tuple[AVMOp, str], Callable[..., Operand]]] = {
    **{(spec.op, spec.operands): spec.reference for spec in REFERENCE_OPS},
    # the comparisons that the reference ops have no entry for
    (AVMOp.lte, "uu"): lambda a, b: int(a <= b),
    (AVMOp.gt, "uu"): lambda a, b: int(a > b),
    (AVMOp.neq, "uu"): lambda a, b: int(a != b),
    (AVMOp.or_, "uu"): lambda a, b: int(bool(a or b)),
    (AVMOp.eq, "bb"): lambda a, b: int(a == b),
    (AVMOp.neq, "bb"): lambda a, b: int(a != b),
    (AVMOp.lte_bytes, "bb"): lambda a, b: int(_number(a) <= _number(b)),
    (AVMOp.gt_bytes, "bb"): lambda a, b: int(_number(a) > _number(b)),
    (AVMOp.gte_bytes, "bb"): lambda a, b: int(_number(a) >= _number(b)),
    (AVMOp.neq_bytes, "bb"): lambda a, b: int(_number(a) != _number(b)),
}


def run_op(op: models.Intrinsic, operands: Sequence[Operand]) -> Operand | None:
    """The result of an AVM op for the given operands, or None if the op fails."""
    kinds = "".join("u" if isinstance(operand, int) else "b" for operand in operands)
    try:
        return _REFERENCE[op.op, kinds](*op.immediates, *operands)
    except OpFailsError:
        return None


@attrs.define
class Program:
    """A made-up subroutine. It returns the values that its variables have at the end."""

    rng: random.Random
    num_variables: int = 3
    num_statements: int = 12
    parameters: list[models.Parameter] = attrs.field(factory=list, init=False)
    subroutine: models.Subroutine = attrs.field(init=False)
    _blocks: list[models.BasicBlock] = attrs.field(factory=list, init=False)
    _versions: Counter[str] = attrs.field(factory=Counter, init=False)
    # the register that holds each variable where statements are being added
    _variables: dict[str, models.Register] = attrs.field(factory=dict, init=False)
    _current: models.BasicBlock = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        self._current = self._block()
        for index in range(self.num_variables):
            for name, ir_type in ((f"u{index}", _UINT64), (f"n{index}", _BYTES)):
                if self.rng.random() < 0.6:
                    param = models.Parameter(
                        name=name,
                        version=0,
                        ir_type=ir_type,
                        source_location=None,
                        implicit_return=False,
                    )
                    self.parameters.append(param)
                    self._variables[name] = param
                elif ir_type == _UINT64:
                    self._set(name, uint64(self.rng.choice(_UINT64_CONSTANTS)))
                else:
                    self._set(name, self._bytes_constant())
        self._statements(self.num_statements, depth=0)
        results: list[models.Value] = [self._variables[name] for name in sorted(self._variables)]
        self._current.terminator = models.SubroutineReturn(result=results, source_location=None)
        self.subroutine = models.Subroutine(
            id="made_up",
            short_name="made_up",
            source_location=None,
            parameters=self.parameters,
            returns=[result.ir_type for result in results],
            body=self._blocks,
            inline=None,
        )
        self.subroutine.validate_with_ssa()

    def random_inputs(self) -> list[Operand]:
        """A value for each parameter: mostly small numbers and whole words."""
        rng = self.rng
        inputs = list[Operand]()
        for param in self.parameters:
            choice = rng.random()
            if param.ir_type == _UINT64:
                if choice < 0.6:
                    inputs.append(rng.choice([0, 1, 2, 3, 5, 8, 10, 31, 32, 64, 255, 256, 1000]))
                else:
                    inputs.append(rng.choice([*_UINT64_CONSTANTS, 2**32 - 1, 2**64 - 2]))
            elif choice < 0.45:
                number = rng.choice([0, 1, 2, 3, 5, 10, 200, 255, 256, 1000, *_NUMBERS[5:9]])
                size = 32 if number >= 2**64 or rng.random() < 0.7 else 8
                inputs.append(number.to_bytes(size, "big"))
            elif choice < 0.8:
                number = rng.choice([*_NUMBERS, 3, 5, 10, 2**128 - 1, 2**255, 2**511])
                data = number.to_bytes((number.bit_length() + 7) // 8, "big")
                if rng.random() < 0.5:
                    data = data.rjust(rng.choice([8, 32, 32, 64, 65]), b"\x00")
                inputs.append(data)
            else:
                inputs.append(rng.randbytes(rng.choice([0, 1, 7, 8, 9, 16, 31, 32, 33, 64, 65])))
        return inputs

    # blocks and registers

    def _block(self) -> models.BasicBlock:
        block = models.BasicBlock(
            id=len(self._blocks), source_location=_LOCATION, ops=[], terminator=None
        )
        self._blocks.append(block)
        return block

    def _register(self, name: str, ir_type: IRType) -> models.Register:
        self._versions[name] += 1
        return models.Register(
            name=name, version=self._versions[name], ir_type=ir_type, source_location=None
        )

    def _set(self, name: str, source: models.ValueProvider) -> models.Register:
        """Assigns to a variable."""
        target = self._register(name, _UINT64 if name.startswith("u") else _BYTES)
        self._current.ops.append(assign(target, source))
        self._variables[name] = target
        return target

    def _temporary(self, source: models.ValueProvider, ir_type: IRType) -> models.Register:
        target = self._register("tmp", ir_type)
        self._current.ops.append(assign(target, source))
        return target

    def _assert(self, condition: models.ValueProvider) -> None:
        self._current.ops.append(assert_(self._temporary(condition, _BOOL)))

    # statements

    def _statements(self, count: int, *, depth: int) -> None:
        for _ in range(count):
            choice = self.rng.random()
            if choice < 0.35:
                self._assignment()
            elif choice < 0.6:
                self._arithmetic()
            elif choice < 0.64:
                self._assert(self._condition())
            elif choice < 0.85 and depth < 3:
                self._if_else(depth)
            elif depth < 2:
                self._while(depth)
            else:
                self._assignment()

    def _if_else(self, depth: int) -> None:
        condition = self._temporary(self._condition(), _BOOL)
        then, otherwise, merge = self._block(), self._block(), self._block()
        branch(self._current, condition, non_zero=then, zero=otherwise)
        before = dict(self._variables)
        ends = list[tuple[models.BasicBlock, dict[str, models.Register]]]()
        for block, statements in (
            (then, self.rng.randrange(1, 4)),
            (otherwise, self.rng.randrange(3)),
        ):
            self._variables = dict(before)
            self._current = block
            self._statements(statements, depth=depth + 1)
            goto(self._current, merge)
            ends.append((self._current, self._variables))
        (then_end, then_variables), (otherwise_end, otherwise_variables) = ends
        self._variables = {}
        for name, value in before.items():
            if then_variables[name] == otherwise_variables[name]:
                self._variables[name] = then_variables[name]
            else:
                merged = phi(
                    self._register(name, value.ir_type),
                    (then_variables[name], then_end),
                    (otherwise_variables[name], otherwise_end),
                )
                merge.phis.append(merged)
                self._variables[name] = merged.register
        self._current = merge

    def _while(self, depth: int) -> None:
        header, body, after = self._block(), self._block(), self._block()
        goto(self._current, header)
        # a phi for every variable, whether the loop changes it or not
        phis = {
            name: phi(self._register(name, value.ir_type), (value, self._current))
            for name, value in self._variables.items()
        }
        header.phis.extend(phis.values())
        self._variables = {name: merged.register for name, merged in phis.items()}
        self._current = header
        # most of the time a counter that is compared with a bound, and stepped in the body
        counter = self.rng.choice(sorted(self._variables)) if self.rng.random() < 0.8 else None
        if counter is None:
            condition = self._condition()
        elif counter.startswith("u"):
            comparison = self.rng.choice([AVMOp.lt, AVMOp.lt, AVMOp.lte, AVMOp.neq, AVMOp.gt])
            condition = intrinsic(comparison, self._variables[counter], self._uint64())
        else:
            comparison = self.rng.choice(
                [AVMOp.lt_bytes, AVMOp.lt_bytes, AVMOp.lte_bytes, AVMOp.neq_bytes, AVMOp.gt_bytes]
            )
            condition = intrinsic(comparison, self._variables[counter], self._bytes())
        branch(header, self._temporary(condition, _BOOL), non_zero=body, zero=after)
        at_header = dict(self._variables)
        self._current = body
        self._statements(self.rng.randrange(4), depth=depth + 1)
        if counter is not None:
            value = self._variables[counter]
            if counter.startswith("u"):
                step = self.rng.choice([AVMOp.add, AVMOp.add, AVMOp.add, AVMOp.sub, AVMOp.mul])
                self._set(
                    counter, intrinsic(step, value, uint64(self.rng.choice([1, 1, 1, 2, 3])))
                )
            else:
                step = self.rng.choice(
                    [AVMOp.add_bytes, AVMOp.add_bytes, AVMOp.add_bytes, AVMOp.sub_bytes]
                )
                self._set(counter, intrinsic(step, value, biguint(self.rng.choice([1, 1, 2]))))
        if self.rng.random() < 0.3:
            # and sometimes changed again after that
            self._statements(1, depth=depth + 1)
        for name, merged in phis.items():
            merged.args.append(
                models.PhiArgument(value=self._variables[name], through=self._current)
            )
        goto(self._current, header)
        self._variables = at_header
        self._current = after

    # values

    def _variable(self, kind: str) -> str:
        return self.rng.choice([name for name in sorted(self._variables) if name.startswith(kind)])

    def _uint64(self) -> models.Value:
        if self.rng.random() < 0.65:
            return self._variables[self._variable("u")]
        return uint64(self.rng.choice(_UINT64_CONSTANTS))

    def _bytes_constant(self) -> models.Constant:
        choice = self.rng.random()
        if choice < 0.4:
            return biguint(self.rng.choice(_NUMBERS))
        if choice < 0.7:
            return bytes_(self.rng.choice(_BYTES_CONSTANTS))
        return bytes_(self.rng.randbytes(self.rng.choice([1, 2, 8, 16, 32, 33, 64, 65])))

    def _bytes(self) -> models.Value:
        if self.rng.random() < 0.65:
            return self._variables[self._variable("n")]
        return self._bytes_constant()

    def _condition(self) -> models.Intrinsic:
        rng = self.rng
        choice = rng.random()
        if choice < 0.4:
            return intrinsic(rng.choice(_UINT64_COMPARISONS), self._uint64(), self._uint64())
        if choice < 0.75:
            comparison = rng.choice([*_NUMBER_COMPARISONS, AVMOp.eq, AVMOp.neq])
            return intrinsic(comparison, self._bytes(), self._bytes())
        if choice < 0.85:
            bits = self._temporary(intrinsic(AVMOp.bitlen, self._bytes()), _UINT64)
            return intrinsic(AVMOp.lte, bits, uint64(rng.choice([8, 16, 64, 128, 256, 512])))
        if choice < 0.9:
            length = self._temporary(intrinsic(AVMOp.len_, self._bytes()), _UINT64)
            comparison = rng.choice([AVMOp.lte, AVMOp.eq, AVMOp.gte])
            return intrinsic(comparison, length, uint64(rng.choice([0, 8, 32, 64])))
        if choice < 0.94:
            return intrinsic(AVMOp.not_, self._temporary(self._condition(), _BOOL))
        return intrinsic(
            rng.choice([AVMOp.and_, AVMOp.or_]),
            self._temporary(self._condition(), _BOOL),
            self._temporary(self._condition(), _BOOL),
        )

    def _assignment(self) -> None:
        rng = self.rng
        choice = rng.random()
        if rng.random() < 0.45:
            name = self._variable("u")
            if choice < 0.5:
                op = rng.choice(
                    [
                        AVMOp.add,
                        AVMOp.sub,
                        AVMOp.mul,
                        AVMOp.div_floor,
                        AVMOp.mod,
                        AVMOp.bitwise_and,
                        AVMOp.bitwise_or,
                        AVMOp.bitwise_xor,
                        AVMOp.shr,
                    ]
                )
                source = intrinsic(op, self._uint64(), self._uint64())
            elif choice < 0.6:
                source = intrinsic(AVMOp.len_, self._bytes())
            elif choice < 0.7:
                source = intrinsic(AVMOp.btoi, self._bytes())
            elif choice < 0.8:
                source = intrinsic(
                    AVMOp.bitlen, self._bytes() if rng.random() < 0.7 else self._uint64()
                )
            elif choice < 0.87:
                op = rng.choice(
                    [
                        AVMOp.extract_uint64,
                        AVMOp.extract_uint32,
                        AVMOp.extract_uint16,
                        AVMOp.getbyte,
                    ]
                )
                source = intrinsic(op, self._bytes(), uint64(rng.choice([0, 0, 1, 8, 24, 31])))
            elif choice < 0.93:
                source = intrinsic(
                    AVMOp.select,
                    self._uint64(),
                    self._uint64(),
                    self._temporary(self._condition(), _BOOL),
                    types=[_UINT64],
                )
            else:
                source = self._condition()
            self._set(name, source)
            return
        name = self._variable("n")
        if choice < 0.45:
            source = intrinsic(rng.choice(_BYTE_MATH), self._bytes(), self._bytes())
        elif choice < 0.6:
            op = rng.choice(
                [AVMOp.bitwise_and_bytes, AVMOp.bitwise_or_bytes, AVMOp.bitwise_xor_bytes]
            )
            source = intrinsic(op, self._bytes(), self._bytes())
        elif choice < 0.7:
            source = intrinsic(AVMOp.itob, self._uint64())
        elif choice < 0.75:
            source = intrinsic(AVMOp.bzero, uint64(rng.choice([0, 1, 8, 32, 64])))
        elif choice < 0.82:
            source = intrinsic(AVMOp.concat, self._bytes(), self._bytes())
        elif choice < 0.9:
            part = rng.choice(
                [(0, 0), (0, 8), (24, 8), (0, 32), (16, 16), (1, 0), (8, 0), (31, 1)]
            )
            source = intrinsic(AVMOp.extract, self._bytes(), immediates=part)
        elif choice < 0.93:
            source = intrinsic(AVMOp.extract3, self._bytes(), self._uint64(), self._uint64())
        elif choice < 0.95:
            source = intrinsic(AVMOp.substring3, self._bytes(), self._uint64(), self._uint64())
        elif choice < 0.97:
            source = intrinsic(AVMOp.bsqrt, self._bytes())
        elif choice < 0.985:
            source = intrinsic(AVMOp.bitwise_not_bytes, self._bytes())
        else:
            source = intrinsic(
                AVMOp.select,
                self._bytes(),
                self._bytes(),
                self._temporary(self._condition(), _BOOL),
                types=[_BYTES],
            )
        self._set(name, source)

    def _arithmetic(self) -> None:
        """What a frontend emits around arithmetic on integers of 256 bits."""
        rng = self.rng
        word = biguint(_WORD)
        x = self._variables[self._variable("n")]
        y = self._variables[self._variable("n")]
        choice = rng.random()
        if choice < 0.1:
            # an argument of a narrow type
            bits = self._temporary(intrinsic(AVMOp.bitlen, x), _UINT64)
            self._assert(intrinsic(AVMOp.lte, bits, uint64(rng.choice([8, 16, 64, 128, 256]))))
        elif choice < 0.2:
            # checked addition or multiplication
            op = rng.choice([AVMOp.add_bytes, AVMOp.add_bytes, AVMOp.mul_bytes])
            result = self._set(self._variable("n"), intrinsic(op, x, y))
            bits = self._temporary(intrinsic(AVMOp.bitlen, result), _UINT64)
            self._assert(intrinsic(AVMOp.lte, bits, uint64(rng.choice([256, 256, 128, 64]))))
        elif choice < 0.32:
            # addition or multiplication that wraps around
            op = rng.choice([AVMOp.add_bytes, AVMOp.add_bytes, AVMOp.mul_bytes])
            wide = self._temporary(intrinsic(op, x, y), _BYTES)
            modulus = biguint(rng.choice([_WORD, _WORD, 2**128, 2**64, 2**160, 2**8]))
            self._set(self._variable("n"), intrinsic(AVMOp.mod_bytes, wide, modulus))
        elif choice < 0.47:
            # subtraction that wraps around, sometimes where it is known not to
            known = rng.random()
            if known < 0.3:
                self._assert(intrinsic(rng.choice([AVMOp.lte_bytes, AVMOp.lt_bytes]), y, x))
            elif known < 0.5:
                self._assert(intrinsic(rng.choice([AVMOp.gte_bytes, AVMOp.gt_bytes]), x, y))
            if rng.random() < 0.5:
                bits = self._temporary(intrinsic(AVMOp.bitlen, x), _UINT64)
                self._assert(intrinsic(AVMOp.lte, bits, uint64(256)))
            operands = rng.choice([(x, word), (word, x)])
            raised = self._temporary(intrinsic(AVMOp.add_bytes, *operands), _BYTES)
            lowered = self._temporary(intrinsic(AVMOp.sub_bytes, raised, y), _BYTES)
            self._set(self._variable("n"), intrinsic(AVMOp.mod_bytes, lowered, word))
        elif choice < 0.55:
            # checked subtraction
            self._assert(intrinsic(AVMOp.gte_bytes, x, y))
            self._set(self._variable("n"), intrinsic(AVMOp.sub_bytes, x, y))
        elif choice < 0.65:
            # a pad to a full word, and a part of the word
            zeros = self._temporary(
                intrinsic(AVMOp.bzero, uint64(rng.choice([32, 32, 8, 16]))), _BYTES
            )
            operands = rng.choice([(zeros, x), (x, zeros)])
            padded = self._set(self._variable("n"), intrinsic(AVMOp.bitwise_or_bytes, *operands))
            if rng.random() < 0.5:
                part = rng.choice([(0, 32), (24, 8), (16, 16), (0, 0), (0, 8)])
                self._set(self._variable("n"), intrinsic(AVMOp.extract, padded, immediates=part))
        elif choice < 0.75:
            # the last bytes as a uint64, and back
            modulus = biguint(rng.choice([2**64, 2**64, 2**32, 2**16, 2**8]))
            low = self._temporary(intrinsic(AVMOp.mod_bytes, x, modulus), _BYTES)
            number = self._set(self._variable("u"), intrinsic(AVMOp.btoi, low))
            if rng.random() < 0.5:
                self._set(self._variable("n"), intrinsic(AVMOp.itob, number))
        elif choice < 0.9:
            # byte math on uint64 values
            a: models.Value = self._temporary(intrinsic(AVMOp.itob, self._uint64()), _BYTES)
            b: models.Value
            if rng.random() < 0.6:
                b = self._temporary(intrinsic(AVMOp.itob, self._uint64()), _BYTES)
            else:
                b = biguint(rng.choice([0, 1, 2, 10, 2**32, 2**64 - 1, 2**64]))
            if rng.random() < 0.5:
                a, b = b, a
            if rng.random() < 0.4:
                self._set(self._variable("u"), intrinsic(rng.choice(_NUMBER_COMPARISONS), a, b))
                return
            result = self._temporary(intrinsic(rng.choice(_BYTE_MATH), a, b), _BYTES)
            use = rng.random()
            if use < 0.3:
                self._set(self._variable("u"), intrinsic(AVMOp.btoi, result))
            elif use < 0.6:
                self._set(self._variable("n"), intrinsic(AVMOp.add_bytes, result, y))
            elif use < 0.8:
                self._set(self._variable("u"), intrinsic(AVMOp.lt_bytes, result, y))
            else:
                self._set(self._variable("n"), result)
        else:
            # a conversion and back
            number = self._temporary(intrinsic(AVMOp.btoi, x), _UINT64)
            self._set(self._variable("n"), intrinsic(AVMOp.itob, number))


class Observer(typing.Protocol):
    """Sees what a run does: every op before it runs, with the values of the registers that
    are defined at that point, and every register once it has been given a value."""

    def before_op(
        self, block: models.BasicBlock, index: int, env: Mapping[models.Register, Operand]
    ) -> None: ...

    def defined(
        self,
        block: models.BasicBlock,
        index: int | None,
        register: models.Register,
        value: Operand,
    ) -> None: ...

    def before_branch(
        self, block: models.BasicBlock, env: Mapping[models.Register, Operand]
    ) -> None: ...


def value_of(env: Mapping[models.Register, Operand], value: models.Value) -> Operand:
    match value:
        case models.UInt64Constant(value=number):
            return number
        case models.BigUIntConstant(value=number):
            return number.to_bytes((number.bit_length() + 7) // 8, "big")
        case models.BytesConstant(value=data):
            return data
        case models.Register():
            return env[value]
        case _:
            raise AssertionError(f"unexpected value: {value}")


def run(
    subroutine: models.Subroutine,
    inputs: Sequence[Operand],
    observer: Observer | None = None,
    *,
    max_ops: int = 400,
) -> tuple[Operand, ...] | str:
    """Runs a subroutine of assignments of AVM ops and values, asserts, gotos and conditional
    branches. Gives the values it returns, "fails" if it fails, or "runs on" if it has not
    come to an end after `max_ops` ops."""
    env: dict[models.Register, Operand] = dict(zip(subroutine.parameters, inputs, strict=True))
    block = subroutine.body[0]
    previous: models.BasicBlock | None = None
    ops_run = 0
    while True:
        # the phis of a block take their values at the same time
        arriving = {
            merged.register: env[arg.value]
            for merged in block.phis
            for arg in merged.args
            if arg.through is previous
        }
        env.update(arriving)
        if observer:
            for register, value in arriving.items():
                observer.defined(block, None, register, value)
        for index, op in enumerate(block.ops):
            ops_run += 1
            if ops_run > max_ops:
                return "runs on"
            if observer:
                observer.before_op(block, index, env)
            match op:
                case models.Assert(condition=condition):
                    if not value_of(env, condition):
                        return "fails"
                case models.Assignment(targets=[target], source=models.Intrinsic() as source):
                    result = run_op(source, [value_of(env, arg) for arg in source.args])
                    if result is None:
                        return "fails"
                    env[target] = result
                    if observer:
                        observer.defined(block, index, target, result)
                case models.Assignment(targets=[target], source=models.Value() as source):
                    env[target] = value_of(env, source)
                    if observer:
                        observer.defined(block, index, target, env[target])
                case _:
                    raise AssertionError(f"unexpected op: {op}")
        previous = block
        match block.terminator:
            case models.Goto(target=block):
                pass
            case models.ConditionalBranch(condition=condition, non_zero=non_zero, zero=zero):
                if observer:
                    observer.before_branch(previous, env)
                block = non_zero if value_of(env, condition) else zero
            case models.SubroutineReturn(result=results):
                return tuple(value_of(env, result) for result in results)
            case terminator:
                raise AssertionError(f"unexpected terminator: {terminator}")


def describe(subroutine: models.Subroutine) -> str:
    lines = list[str]()
    for block in subroutine.body:
        lines.append(f"{block}:")
        lines.extend(
            f"    {merged.register.local_id} = phi("
            + ", ".join(f"{arg.value.local_id} <- {arg.through}" for arg in merged.args)
            + ")"
            for merged in block.phis
        )
        lines.extend(f"    {op}" for op in block.ops)
        lines.append(f"    {block.terminator}")
    return "\n".join(lines)
