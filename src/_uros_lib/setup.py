from algopy import (
    Box,
    BoxMap,
    Bytes,
    Global,
    OnCompleteAction,
    OpUpFeeSource,
    Txn,
    UInt64,
    arc4,
    ensure_budget,
    gtxn,
    itxn,
    op,
    subroutine,
    urange,
)

# Generic uros setup helper. It holds every chunk program in its own boxes and swaps the owning
# chunk into the `main` app via an inner UpdateApplication. The actual method call is a *separate*
# top-level txn the caller issues to `main` (so it runs with the real caller's context — no
# forwarding), and `prepare` is its leading `appl` arg. `main`'s program simply becomes that
# chunk; the next call swaps in whatever chunk it needs, and each method's grafted guard requires
# this prepare immediately before it, so a chunk loaded by an earlier call can't be invoked
# without a fresh swap. Reusable as-is: parameterised only by the `main` app id + loaded boxes.


class CodeboxSpec(arc4.Struct):
    key: arc4.DynamicBytes
    size: arc4.UInt64


class UrosSetup(arc4.ARC4Contract):
    def __init__(self) -> None:
        self.main_app = UInt64(0)
        # maps an ARC-4 method selector -> the codebox key of the chunk that owns it
        self.method_chunk = BoxMap(Bytes, Bytes, key_prefix=b"m")
        # codebox keys the creator registered as installable chunk programs (planned mode)
        self.chunk_registered = BoxMap(Bytes, UInt64, key_prefix=b"c")

    # --- deploy-time setup (creator only) ---

    @arc4.abimethod
    def set_main(self, main_app: UInt64) -> None:
        assert Txn.sender == Global.creator_address, "uros: only creator"
        self.main_app = main_app

    @arc4.abimethod
    def map_method(self, selector: Bytes, chunk_key: Bytes) -> None:
        assert Txn.sender == Global.creator_address, "uros: only creator"
        self.method_chunk[selector] = chunk_key

    @arc4.abimethod
    def register_chunk(self, chunk_key: Bytes) -> None:
        # only registered codeboxes may be installed by `install`; never the clear program box
        assert Txn.sender == Global.creator_address, "uros: only creator"
        assert chunk_key != Bytes(b"clear"), "uros: clear program is not a chunk"
        self.chunk_registered[chunk_key] = UInt64(1)

    @arc4.abimethod
    def create_codeboxes(self, specs: arc4.DynamicArray[CodeboxSpec]) -> None:
        # create every named codebox, zero-filled, at its exact program size. The AVM aborts if
        # a box with that key already exists at a *different* size. Each box must be in this
        # txn's box references, so a single call can create up to the 8-reference-per-txn cap.
        assert Txn.sender == Global.creator_address, "uros: only creator"
        # decoding the spec array + the per-box work can exceed the 700-op budget for a full
        # batch, so top it up via opup (fees drawn from surplus group credit)
        ensure_budget(
            specs.length * UInt64(120) + UInt64(200), fee_source=OpUpFeeSource.GroupCredit
        )
        for i in urange(specs.length):
            spec = specs[i].copy()
            Box(Bytes, key=spec.key.native).create(size=spec.size.native)

    @arc4.abimethod
    def write_box(self, key: Bytes, offset: UInt64, data: Bytes) -> None:
        assert Txn.sender == Global.creator_address, "uros: only creator"
        Box(Bytes, key=key).replace(offset, data)

    # --- runtime swap ---

    @arc4.abimethod
    def prepare(self) -> None:
        # the NEXT txn is the `main` method call; swap in the chunk that owns its selector.
        # `main`'s program simply becomes that chunk. Safety comes from the method's own guard:
        # each split method takes this prepare as its leading `appl` arg and asserts it, so a
        # chunk loaded by an earlier call can't be invoked without a fresh prepare right before.
        nxt = gtxn.ApplicationCallTransaction(Txn.group_index + 1)
        assert nxt.app_id.id == self.main_app, "uros: next txn must call main"
        self._swap(self.method_chunk[nxt.app_args(0)])

    @arc4.abimethod
    def install(self, chunk_key: Bytes) -> None:
        # planned mode: the caller's group installs the chunk its later calls need (including
        # calls that reach `main` as inner transactions from other apps). Any registered chunk of
        # the same contract may be installed by anyone: it changes which methods `main` serves,
        # never its state, and every group installs exactly what it needs before using it.
        assert chunk_key in self.chunk_registered, "uros: unregistered chunk"
        self._swap(chunk_key)

    @subroutine
    def _swap(self, approval_key: Bytes) -> None:
        # A chunk program may be up to 16 KiB (AVM 13: 1 + 7 extra pages), but one `bytes` value
        # tops out at 4096, so hand the inner UpdateApplication four <=4096-byte pages (the AVM
        # concatenates them; empty trailing pages contribute nothing).
        length, exists = op.Box.length(approval_key)
        assert exists, "uros: missing program box"
        assert length <= UInt64(16384), "uros: chunk program exceeds 16 KiB"
        clear = Box(Bytes, key=Bytes(b"clear")).value
        itxn.ApplicationCall(
            app_id=self.main_app,
            on_completion=OnCompleteAction.UpdateApplication,
            # carry main's guard selector so its router admits this UpdateApplication
            app_args=(arc4.arc4_signature("uros_guard()void"),),
            approval_program=(
                self._page(approval_key, length, UInt64(0)),
                self._page(approval_key, length, UInt64(1)),
                self._page(approval_key, length, UInt64(2)),
                self._page(approval_key, length, UInt64(3)),
            ),
            clear_state_program=(clear,),
        ).submit()


    @subroutine
    def _page(self, key: Bytes, length: UInt64, index: UInt64) -> Bytes:
        start = index * UInt64(4096)
        if start >= length:
            return Bytes()
        size = length - start
        if size > UInt64(4096):
            size = UInt64(4096)
        return op.Box.extract(key, start, size)
