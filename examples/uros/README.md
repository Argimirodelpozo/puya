# uros splitter (working around the AVM 8 KB program-size cap)

Author one Algorand Python contract whose methods total more than 8 KB; the compiler splits it
into **chunks**, and a separately deployed **setup** helper swaps the owning chunk into the
contract at call time. The heavy method runs as a real top-level txn (so `Txn.sender`/args are
the real caller's), the chunk code lives on-chain in setup's boxes (so other contracts can
compose it via inner txns), and a grafted entry guard rejects any heavy-method call not
preceded by `setup.prepare`.

The per-method `chunk="name"` groups methods into chunks: methods sharing a name land in one
chunk. The splitter also **force-merges** any chunks whose methods call each other — only one
chunk is live at a time, so a callee can't be left as a stub. (A merged chunk's box key is the
`+`-joined names, e.g. `fast+slow`.) Sizing each chunk under 8 KB is up to you; if a chunk
overshoots, the assembler rejects it — split it with finer `chunk=` names.

Driven entirely by **in-contract config** — no CLI flag:

```python
class MyContract(arc4.ARC4Contract, splitter="uros"):
    @arc4.abimethod(chunk="fast")
    def m_a(self, ...): ...
    @arc4.abimethod(chunk="slow")
    def m_b(self, ...): ...
```

```bash
cd examples/uros
puyapy contract.py  --out-dir out --output-arc56 --output-bytecode   # main shell + chunks + setup
puyapy consumer.py  --out-dir out --output-arc56                      # the contract client
python deploy_and_call.py

# regenerate the checked-in typed client if RiskEngine's ABI changes:
#   puyapy contract.py --output-client && cp out/client_RiskEngine.py .
```

Compiling `contract.py` emits the user's `main` shell + one program per chunk + the `setup`
helper in one go. (`consumer.py` is built from inside the dir so it can import the sibling
`client_RiskEngine.py`.)

The example (`contract.py`) is a toy DeFi **risk engine** (`RiskEngine`) with three feature
areas — **pricing / funding / margin** — three ABI methods each. The methods of an area all
call that area's helper, which embeds a 4 KB model table; the three tables push the contract
past the 8 KB cap. Each method is tagged `chunk="pricing"` / `"funding"` / `"margin"`, so the
splitter emits three chunks (one per area, ~4.8 KB / 3 pages each — over the 4096-byte
single-`bytes` limit, so the swap pages them in). The grouping is *motivated*: an area's methods
share its helper, so they belong in one chunk. `deploy_and_call.py` calls two pricing methods
(`mark_price`, `bid_price`) to
show one chunk program really holds several real methods, then one method from each other area.

**Calling a split method.** A method can't be called directly (that hits a stub / the wrong
chunk); the caller submits `[setup.prepare(), main.method(prepare, *args)]`, with `prepare` as
the method's leading `appl` arg, carrying the references the chunk swap needs.

This requirement shows up in the **generated client** (`puyapy --output-client` →
`out/client_RiskEngine.py`): the grafted guard surfaces as a typed transaction parameter, so a
caller is told to supply the preceding `setup.prepare`:

```python
class RiskEngine(algopy.arc4.ARC4Client, typing.Protocol):
    @algopy.arc4.abimethod
    def mark_price(self, _uros_prep: algopy.gtxn.ApplicationCallTransaction,
                   market: ..., qty: ...) -> ...: ...
```

`consumer.py` is a **contract client** that does exactly this on-chain — and it uses that
generated client to do it:

```python
result, _ = arc4.abi_call(RiskEngine.mark_price, prepare, market, qty, app_id=self.main_app)
```

`abi_call` composes the inner group for you: the `prepare` app-call is passed as the `_uros_prep`
arg, so it's placed first → `[setup.prepare, main.mark_price(prepare, ...)]`, submitted as one
atomic inner group. The consumer ships no chunk code (it's in `setup`'s boxes), and RiskEngine
sees the consumer app as the caller. The outer txn that drives it must carry the foreign-app refs
(`setup`, `main`) and the box refs the swap reads (inherited from the top-level txn) — see
`deploy_and_call.py` step 5.

---

## Context

The AVM caps a single deployed program at **8 KB** (4 × 2 KB pages via `extra_program_pages`).
Some contracts compile past that. We want to keep authoring one contract and have the toolchain
split it, rather than forcing authors to manually shard logic across helper apps.

The IR-level splitter runs **after SSA construction** (`awst_to_ir`, before `transform_ir`):
clones the contract into a **shell** (heavy method bodies stubbed) plus one **chunk** program
per group (that group's bodies real, the rest stubbed). Each emitted program is then
independently dead-code-pruned and lowered by puya's normal pipeline — no hand-rolled worklist,
no cross-program duplication.

### The one hard constraint

You cannot `UpdateApplication` an app **and run its new code in the same call** — an update
takes effect for the *next* call. So the app whose code is swapped must be **different** from
the one issuing the swap, and the swap + run must be driven as a **group from outside**. The
design below is a direct consequence.

### Why a single all-stubbed shell

The sum of the real method bodies is, by construction, > 8 KB — so **no single program can hold
every real body**. The contract still needs one program that (a) exposes the full ABI under the
original name (what clients compile their ARC-56 against) and (b) is ≤ 8 KB so it can actually
be deployed. The only program that satisfies both is the all-stubbed **shell**. It's also the
genesis program `main` is created from. Its stubbed bodies never execute: by the time any heavy
method runs, a chunk has been swapped in (the guard enforces that), so the live program is
always a chunk, never the shell.

---

## The two apps

| App | Role |
|---|---|
| **`main`** (your contract) | The user's split contract itself — created from the shell, its program swapped to the owning chunk per call. State lives here. The splitter **grafts** the uros plumbing onto it (so user contracts don't write it): `uros_guard` (kept real in every program, admits `UpdateApplication` only when `caller == setup`), `uros_set_setup`, and the `uros_setup` state. |
| **`setup`** (`UrosSetup`) | A *generic, reusable* swap helper. Holds every chunk in boxes (+ a `selector → chunk-key` map). `prepare()` reads the next txn's selector and swaps the owning chunk into `main` (inner `UpdateApplication`, multi-page). Shipped as embedded AWST and emitted automatically. |

**The group.** The caller submits a 2-txn top-level group; the method is a real txn they issue
to `main`, and it takes `setup.prepare` as a leading **`appl` transaction argument**:

```
[ setup.prepare()  ,  main.<method>(prepare, args) ]
   swap chunk in (inner)  REAL body, top-level
```

The splitter grafts a guard into each heavy method asserting that `appl` arg is exactly
`setup.prepare` (`app_id == uros_setup && selector == prepare`). So a chunk left loaded by an
earlier call can never be invoked without a fresh `prepare` immediately before it. Using an
`appl` arg means ABI clients **auto-place** the prepare call before the method — the convention
is machine-readable, not just documented.

Because the method is a separate top-level txn: `Txn.sender`/args are the **real** caller's (no
forwarding), the method can **inspect its own group** (e.g. require a payment), and the chunk
bytes live **on-chain in `setup`'s boxes**, so another contract can drive the same
`[prepare, method]` group via **inner** txns without shipping code.

The `setup` helper (and the grafted `infra`) ship like puyalib artifacts: source of truth in
`src/_uros_lib/` → `poe gen_uros_setup` pre-compiles them to embedded
`src/puya/ir/uros/lib.awst.json` → the `splitter=` kwarg on a Contract triggers detection,
infra grafting, guard injection, chunk resolution, and emission of `setup` alongside the shell +
chunks. The only compiler-specific output is `deploy.uros.json`, a tiny manifest carrying just
the post-pack **method → chunk** mapping (+ the main/setup contract names) — the one fact the
deploy can't get elsewhere. Everything else the deploy reads from the standard puya artifacts:
program bytes from `<name>.approval.bin` / `.clear.bin` (hence `--output-bytecode`), the state
schema from `<main>.arc56.json`, and `extra_program_pages` computed from the chunk sizes. Multi-
page swap reads each program from its box in ≤4096-byte pages and passes them as an
`ApprovalProgramPages` tuple, so chunks scale to the full 8 KB.

---

## Consequences

- **MBR is not a real limit.** Box MBR is ~400 µAlgo/byte (~3 ALGO per 7.4 KB chunk), one-time
  and refundable — negligible.
- **Box-read budget** (1024 B × refs, pooled across the group, ≤8 refs/txn) is the constraint
  that scales with chunk size; padded with empty refs. ~12 KB of model tables is split here.
- **`main`'s program is whatever chunk ran last** — there's no canonical on-chain program
  between calls. That's safe because the per-method guard rejects any heavy-method call not
  immediately preceded by `setup.prepare`, and it survives inner composition (the guard is a
  local check on the method's own group). The price: anything you *invoke* must be split (a
  non-split writable method would silently no-op against whatever chunk is loaded); plain state
  is read off-chain.

---

## Implementation map

| File | What |
|---|---|
| `stubs/algopy-stubs/_contract.pyi`, `arc4.pyi` | `Contract` `splitter="…"` class kwarg + `@arc4.abimethod(chunk="…")` method kwarg |
| `src/puyapy/awst_build/{module,arc4_decorators,contract}.py` | parses those kwargs into AWST `Contract.splitter` + `ARC4ABIMethodConfig.chunk` |
| `src/puya/compile.py` | calls `uros.detect_config` then dispatches `inject_setup` / `apply` / `write_manifest` |
| `src/puya/ir/uros/__init__.py` | all of it: `split_contract`, `detect_config` (hints), `_resolve_chunks` (force-merge call-connected hint groups), infra graft, guard injection, `apply`, `write_manifest` |
| `src/puya/ir/uros/lib.awst.json` | embedded `setup` + `infra` (regenerated via `poe gen_uros_setup`) |
| `src/_uros_lib/` (`setup.py`, `infra.py`), `scripts/generate_uros_setup.py` | source of truth for the uros lib |
| `examples/uros/` | this example: `contract.py` (the split `RiskEngine`), `client_RiskEngine.py` (checked-in `--output-client` typed client), `consumer.py` (a contract client that imports it and calls via inner txns), `deploy_and_call.py` (live localnet verification, incl. the contract-client leg) |
