from algopy import Global, Txn, UInt64, arc4, gtxn

# uros-relay infrastructure that the splitter splices into the user's `main` contract at compile
# time (so user contracts don't hand-write it): the swap guard, the setter for the trusted
# `setup` app id, and the per-method "preceded by setup.prepare" guard template. Compiled to
# embedded AWST; methods/state are grafted onto the target by rewriting the owning contract
# reference, and the guard template's `appl` arg + asserts are prepended to each split method.
# See puya.ir.uros.inject_setup.


class UrosInfra(arc4.ARC4Contract):
    def __init__(self) -> None:
        self.uros_setup = UInt64(0)

    @arc4.abimethod(allow_actions=["UpdateApplication"])
    def uros_guard(self) -> None:
        # only the `setup` app may swap this contract's program (via inner UpdateApplication)
        assert Global.caller_application_id == self.uros_setup, "uros: only setup may swap"

    @arc4.abimethod
    def uros_set_setup(self, setup_app: UInt64) -> None:
        assert Txn.sender == Global.creator_address, "uros: only creator"
        self.uros_setup = setup_app

    @arc4.abimethod
    def uros_prepare_guard(self, _uros_prep: gtxn.ApplicationCallTransaction) -> None:
        # TEMPLATE: not grafted as a method. The splitter lifts this `appl` arg + the asserts
        # below and prepends them to every split method, so each heavy method is preceded by
        # a real setup.prepare (and ABI clients auto-place that app call before it).
        assert _uros_prep.app_id.id == self.uros_setup, "uros: must be preceded by setup.prepare"
        assert _uros_prep.app_args(0) == arc4.arc4_signature(
            "prepare()void"
        ), "uros: preceding txn is not setup.prepare"
