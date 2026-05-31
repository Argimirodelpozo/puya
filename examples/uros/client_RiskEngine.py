# This file is auto-generated, do not modify
# flake8: noqa
# fmt: off
import typing

import algopy


class RiskEngine(algopy.arc4.ARC4Client, typing.Protocol):
    @algopy.arc4.abimethod
    def mark_price(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def bid_price(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def ask_price(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def accrue_funding(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def settle_funding(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def funding_owed(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def health_factor(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def liquidation_price(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod
    def max_borrow(
        self,
        _uros_prep: algopy.gtxn.ApplicationCallTransaction,
        market: algopy.arc4.UIntN[typing.Literal[64]],
        qty: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> algopy.arc4.UIntN[typing.Literal[64]]: ...

    @algopy.arc4.abimethod(allow_actions=['UpdateApplication'])
    def uros_guard(
        self,
    ) -> None: ...

    @algopy.arc4.abimethod
    def uros_set_setup(
        self,
        setup_app: algopy.arc4.UIntN[typing.Literal[64]],
    ) -> None: ...
