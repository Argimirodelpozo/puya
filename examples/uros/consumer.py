from algopy import Application, Global, Txn, UInt64, arc4, itxn
from algopy.arc4 import abimethod

# `client_RiskEngine.py` is puyapy's generated typed client (`--output-client`), checked in here.
# Regenerate with `puyapy contract.py --output-client` if RiskEngine's ABI changes.
from client_RiskEngine import RiskEngine

# A *contract* client for the uros-split RiskEngine: another on-chain app that calls a split
# method via an inner transaction, using that generated typed client. This is the composability
# payoff — the chunk code lives on-chain in `setup`'s boxes, so this consumer ships no program
# bytes; it just calls RiskEngine.mark_price, and `arc4.abi_call` composes the required
# `[setup.prepare, main.mark_price(prepare, ...)]` inner group for it. RiskEngine then sees THIS
# app as the caller (real Txn.sender == this app). Note `mark_price`'s leading `_uros_prep` arg in
# the generated client: the grafted guard's "preceded by setup.prepare" requirement, surfaced as
# a typed transaction parameter.


class Consumer(arc4.ARC4Contract):
    def __init__(self) -> None:
        self.setup_app = UInt64(0)
        self.main_app = UInt64(0)

    @abimethod
    def configure(self, setup_app: UInt64, main_app: UInt64) -> None:
        assert Txn.sender == Global.creator_address, "only creator"
        self.setup_app = setup_app
        self.main_app = main_app

    @abimethod
    def best_quote(self, market: UInt64, qty: UInt64) -> UInt64:
        # the prepare app-call that must precede mark_price; passing it as the `_uros_prep` arg
        # makes abi_call put it first in the inner group -> [setup.prepare, main.mark_price].
        prepare = itxn.ApplicationCall(
            app_id=self.setup_app,
            app_args=(arc4.arc4_signature("prepare()void"),),
            apps=(Application(self.main_app),),
        )
        result, _txn = arc4.abi_call(
            RiskEngine.mark_price, prepare, market, qty, app_id=self.main_app
        )
        return result.native
