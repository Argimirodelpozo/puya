from algopy import Account, Bytes, Txn, UInt64, arc4, op, subroutine, urange

# A (toy but plausibly-shaped) on-chain risk engine for a perps/lending protocol. It has three
# feature areas — pricing, funding, margin — each with several ABI methods that share one
# feature helper, and each helper embeds that area's calibrated model table (~4 KB of
# coefficients a protocol might ship on-chain). Together the model tables push the contract well
# past the 8 KB AVM program-size cap, so it can't be deployed as a single program.
#
# `splitter="uros"` turns on the splitter; each method is tagged with its feature area via
# `chunk="..."`. The grouping is *motivated*: the methods of an area all call that area's helper,
# so they genuinely belong in the same chunk (the auto-packer would force them together anyway —
# only one chunk's code is live at a time, so a callee can't be a stub). The numbers come from a
# deterministic hash-fold over the model table so the example can verify them off-chain; a real
# engine would do real fixed-point math over the coefficients.

_PRICE_MODEL = "a17f3c08" * 1000  # pricing curve coefficients    (4000 B)
_FUNDING_CURVE = "5e2db941" * 1000  # funding-rate curve          (4000 B)
_RISK_WEIGHTS = "c4093b7e" * 1000  # collateral / risk weights    (4000 B)


class RiskEngine(arc4.ARC4Contract, splitter="uros"):
    def __init__(self) -> None:
        self.calls = UInt64(0)
        self.last_result = UInt64(0)
        self.last_caller = Account()

    # ----- pricing -----

    @arc4.abimethod(chunk="pricing")
    def mark_price(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._price(market, qty, UInt64(0)))

    @arc4.abimethod(chunk="pricing")
    def bid_price(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._price(market, qty, UInt64(1)))

    @arc4.abimethod(chunk="pricing")
    def ask_price(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._price(market, qty, UInt64(2)))

    # ----- funding -----

    @arc4.abimethod(chunk="funding")
    def accrue_funding(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._funding(market, qty, UInt64(0)))

    @arc4.abimethod(chunk="funding")
    def settle_funding(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._funding(market, qty, UInt64(1)))

    @arc4.abimethod(chunk="funding")
    def funding_owed(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._funding(market, qty, UInt64(2)))

    # ----- margin -----

    @arc4.abimethod(chunk="margin")
    def health_factor(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._risk(market, qty, UInt64(0)))

    @arc4.abimethod(chunk="margin")
    def liquidation_price(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._risk(market, qty, UInt64(1)))

    @arc4.abimethod(chunk="margin")
    def max_borrow(self, market: UInt64, qty: UInt64) -> UInt64:
        return self._emit(self._risk(market, qty, UInt64(2)))

    # ----- feature helpers (one per area; each embeds that area's model table) -----

    @subroutine
    def _price(self, market: UInt64, qty: UInt64, variant: UInt64) -> UInt64:
        return self._eval(market, qty, variant, Bytes.from_hex(_PRICE_MODEL))

    @subroutine
    def _funding(self, market: UInt64, qty: UInt64, variant: UInt64) -> UInt64:
        return self._eval(market, qty, variant, Bytes.from_hex(_FUNDING_CURVE))

    @subroutine
    def _risk(self, market: UInt64, qty: UInt64, variant: UInt64) -> UInt64:
        return self._eval(market, qty, variant, Bytes.from_hex(_RISK_WEIGHTS))

    # ----- shared low-level routines (kept in every chunk; tiny) -----

    @subroutine
    def _eval(self, market: UInt64, qty: UInt64, variant: UInt64, model: Bytes) -> UInt64:
        # deterministic fold of (inputs, model table) -> a uint64 "score"
        computed = op.itob(market) + op.itob(qty) + op.itob(variant)
        for _i in urange(4):
            computed = op.sha256(computed)
            computed = op.sha512_256(computed + model)
        return op.btoi(computed[:8])

    @subroutine
    def _emit(self, result: UInt64) -> UInt64:
        self.last_result = result
        self.last_caller = Txn.sender  # real top-level sender — no forwarding
        self.calls += 1
        return result
