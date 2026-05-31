#!/usr/bin/env python3
"""Live localnet verification of the uros splitter on a small DeFi risk engine.

`RiskEngine` has three feature areas — pricing / funding / margin — each with three ABI methods
that share that area's helper (which embeds a 4 KB model table). The three tables (+ code) push
the contract past the 8 KB program cap, so the compiler splits it into one chunk per feature:
  * a `main` "shell" — every heavy body stubbed; the genesis program `main` is created from, and
    the artifact carrying the full ABI under the original name. ~1 KB.
  * one program per chunk (pricing / funding / margin) — that area's 3 methods real, rest stubbed
    (~4.8 KB / 3 pages each; the shared model table lives once in the area's helper).
  * `setup` (UrosSetup) — a generic helper that holds every chunk in boxes and swaps the owning
    chunk into `main` via an inner UpdateApplication.

Per call the caller submits a 2-txn group; `setup.prepare` is the method's leading `appl` arg:
    [ setup.prepare()  ,  main.method(prepare, args) ]
       swap chunk in (inner)  real body, top-level
`prepare` reads the next txn's selector, picks the owning chunk, and swaps it into `main` (in
4096-byte pages, since a chunk exceeds the single-`bytes` cap). The method then runs top-level on
`main` (real Txn.sender / args), and the grafted guard asserts its `appl` arg is exactly
`setup.prepare` — so a chunk left loaded from a previous call can't be invoked without a swap.

Run against a running algokit localnet:  python examples/uros/deploy_and_call.py
"""
# ruff: noqa: T201

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

import algokit_utils as au

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
WRITE_CHUNK = 2000  # <= ABI byte[] arg budget per write_box

MAX_REFS = 8  # AVM allows 8 references per txn
WRITE_BUDGET = 4096  # box write budget granted per box reference

# the model tables, mirroring the module constants in contract.py
PRICE_MODEL = bytes.fromhex("a17f3c08" * 1000)
FUNDING_CURVE = bytes.fromhex("5e2db941" * 1000)
RISK_WEIGHTS = bytes.fromhex("c4093b7e" * 1000)
# method -> (model table, variant) — mirrors which helper each method calls + with which variant
MODEL = {
    "mark_price": (PRICE_MODEL, 0),
    "bid_price": (PRICE_MODEL, 1),
    "ask_price": (PRICE_MODEL, 2),
    "accrue_funding": (FUNDING_CURVE, 0),
    "settle_funding": (FUNDING_CURVE, 1),
    "funding_owed": (FUNDING_CURVE, 2),
    "health_factor": (RISK_WEIGHTS, 0),
    "liquidation_price": (RISK_WEIGHTS, 1),
    "max_borrow": (RISK_WEIGHTS, 2),
}


def evaluate(market: int, qty: int, variant: int, model: bytes) -> int:
    """Pure-Python mirror of the contract's `_eval` fold, to check on-chain results."""
    c = market.to_bytes(8, "big") + qty.to_bytes(8, "big") + variant.to_bytes(8, "big")
    for _ in range(4):
        c = hashlib.sha256(c).digest()
        c = hashlib.new("sha512_256", c + model).digest()
    return int.from_bytes(c[:8], "big")


def pubkey(address: str) -> bytes:
    return base64.b32decode(address + "=" * (-len(address) % 8))[:32]


def create_codeboxes(setup_client: au.AppClient, boxes: list[tuple[bytes, bytes]]) -> None:
    """Create every (key, data) box via setup.create_codeboxes, in as few calls as possible.
    box_create consumes write budget == the box size, and the per-txn write budget is
    4096 * (#box refs, max 8). algokit's resource population adds write-budget padding refs on
    top of the named boxes, so we keep each batch's total bytes under (8 - 4) * 4096 to leave
    room for that padding (~7 KB chunks pack ~2 per call)."""
    i = 0
    while i < len(boxes):
        batch: list[tuple[bytes, bytes]] = []
        total = 0
        while i < len(boxes) and len(batch) < MAX_REFS:
            size = len(boxes[i][1])
            if batch and total + size > WRITE_BUDGET * (MAX_REFS - 4):
                break
            batch.append(boxes[i])
            total += size
            i += 1
        setup_client.send.call(
            au.AppClientMethodCallParams(
                method="create_codeboxes",
                args=[[(k, len(d)) for k, d in batch]],
                box_references=[k for k, _ in batch],
                static_fee=au.AlgoAmount.from_micro_algo(6000),  # surplus funds ensure_budget opup
            ),
            send_params={"populate_app_call_resources": True},
        )


def main() -> None:
    # the manifest carries only the compiler-only method -> chunk mapping + contract names;
    # everything else (bytecode, schema, page count) comes from the standard puya artifacts, so
    # compile with `--output-arc56 --output-bytecode` (see this example's README).
    manifest = json.loads((OUT / "deploy.uros.json").read_text())
    main_name = manifest["main_contract"]
    setup_name = manifest["setup_contract"]
    methods = manifest["methods"]

    # `.bin` = raw assembled program bytes; main's shell is its own approval program
    shell_program = (OUT / f"{main_name}.approval.bin").read_bytes()
    clear_program = (OUT / f"{main_name}.clear.bin").read_bytes()
    sch = json.loads((OUT / f"{main_name}.arc56.json").read_text())["state"]["schema"]
    # chunk names (box keys) in first-seen order, and each chunk's program from its .bin
    chunk_names: list[str] = []
    for m in methods:
        if m["chunk"] not in chunk_names:
            chunk_names.append(m["chunk"])
    chunk_programs = {
        cn: (OUT / f"{main_name}__chunk_{cn}.approval.bin").read_bytes() for cn in chunk_names
    }
    # `main` must be created with enough extra pages to later receive the largest chunk
    main_extra_pages = max(math.ceil(len(p) / 2048) for p in chunk_programs.values()) - 1

    algorand = au.AlgorandClient.default_localnet()
    dispenser = algorand.account.localnet_dispenser()
    algorand.set_default_signer(dispenser.signer)
    sender = dispenser.addr

    # 1. deploy `setup` and fund it for the codeboxes (box MBR ~3 ALGO per ~7 KB chunk)
    setup_factory = au.AppFactory(
        au.AppFactoryParams(
            algorand=algorand,
            app_spec=au.Arc56Contract.from_json((OUT / f"{setup_name}.arc56.json").read_text()),
            default_sender=sender,
        )
    )
    setup_client, _ = setup_factory.send.bare.create()
    setup_app_id = setup_client.app_id
    print(f"setup app id = {setup_app_id}")
    algorand.send.payment(
        au.PaymentParams(
            sender=sender, receiver=setup_client.app_address, amount=au.AlgoAmount.from_algo(20)
        )
    )

    # 2. create `main` from its shell (the genesis program), with extra pages so it can later
    #    receive a multi-page chunk; then tell it which app (setup) may swap it
    main_create = algorand.send.app_create(
        au.AppCreateParams(
            sender=sender,
            approval_program=shell_program,
            clear_state_program=clear_program,
            extra_program_pages=main_extra_pages,
            schema={
                "global_ints": sch["global"]["ints"],
                "global_byte_slices": sch["global"]["bytes"],
                "local_ints": sch["local"]["ints"],
                "local_byte_slices": sch["local"]["bytes"],
            },
        )
    )
    main_app_id = main_create.app_id
    print(f"main app id = {main_app_id} (extra pages={main_extra_pages})")
    main_client = au.AppClient(
        au.AppClientParams(
            app_id=main_app_id,
            algorand=algorand,
            app_spec=au.Arc56Contract.from_json((OUT / f"{main_name}.arc56.json").read_text()),
            default_sender=sender,
        )
    )
    main_client.send.call(
        au.AppClientMethodCallParams(method="uros_set_setup", args=[setup_app_id])
    )
    setup_client.send.call(
        au.AppClientMethodCallParams(method="set_main", args=[main_app_id])
    )

    # 3. create all codeboxes (batched), then fill them with program bytes. setup holds the
    #    clear program + the chunks (the shell is only `main`'s genesis program, never boxed).
    boxes: list[tuple[bytes, bytes]] = [(b"clear", clear_program)]
    chunk_keys = {}  # chunk_name -> box key bytes
    for cn in chunk_names:
        key = cn.encode()
        chunk_keys[cn] = key
        boxes.append((key, chunk_programs[cn]))

    create_codeboxes(setup_client, boxes)
    for key, data in boxes:
        for off in range(0, len(data), WRITE_CHUNK):
            setup_client.send.call(
                au.AppClientMethodCallParams(
                    method="write_box", args=[key, off, data[off : off + WRITE_CHUNK]],
                    box_references=[key], static_fee=au.AlgoAmount.from_micro_algo(2000),
                )
            )
    print(f"created + loaded {len(boxes)} codeboxes into setup ({len(chunk_names)} chunks)")

    selectors = {m["name"]: bytes.fromhex(m["selector"][2:]) for m in methods}
    method_chunk = {m["name"]: m["chunk"] for m in methods}
    for m in methods:
        sel = selectors[m["name"]]
        setup_client.send.call(
            au.AppClientMethodCallParams(
                method="map_method", args=[sel, chunk_keys[m["chunk"]]],
                box_references=[b"m" + sel],
                static_fee=au.AlgoAmount.from_micro_algo(2000),
            )
        )
    print(f"mapped {len(selectors)} selectors -> chunks")

    def empties(n: int) -> list[au.BoxReference]:
        return [au.BoxReference(0, b"") for _ in range(n)]

    def prepare_call(method: str) -> au.AppCallMethodCallParams:
        sel = selectors[method]
        key = chunk_keys[method_chunk[method]]
        # prepare swaps the owning chunk in; pad box refs so the pooled read budget covers it.
        return setup_client.params.call(
            au.AppClientMethodCallParams(
                method="prepare", args=[],
                app_references=[main_app_id],  # 1 app + 7 box = 8 refs (the AVM cap)
                box_references=[b"m" + sel, key, b"clear", *empties(4)],
                static_fee=au.AlgoAmount.from_micro_algo(2000),
            )
        )

    def call_split(method: str, market: int, qty: int, expected_calls: int) -> None:
        # the method's leading `appl` arg IS setup.prepare; passing the prepare call as that arg
        # makes the ABI composer place it immediately before -> 2-txn group [prepare, method].
        result = main_client.send.call(
            au.AppClientMethodCallParams(
                method=method, args=[prepare_call(method), market, qty],
                box_references=empties(4),  # extra pooled read budget for the chunk swap
            )
        )
        model, variant = MODEL[method]
        expected = evaluate(market, qty, variant, model)
        state = algorand.app.get_global_state(main_app_id)
        calls = state["calls"].value
        last_result = state["last_result"].value
        raw_c = state["last_caller"].value
        last_caller = bytes.fromhex(raw_c) if isinstance(raw_c, str) else bytes(raw_c)
        ok = "OK" if last_result == expected else "MISMATCH"
        print(f"{method:18s} (chunk {method_chunk[method]!r}): -> {last_result:>20} {ok}")
        assert last_result == expected, f"{method}: result mismatch"
        assert calls == expected_calls, f"{method}: calls {calls}"
        assert last_caller == pubkey(sender), f"{method}: sender wrong"
        if result.abi_return is not None:
            assert result.abi_return == expected, f"{method}: abi return mismatch"

    # 4. call methods across feature areas. mark_price + bid_price share chunk 'pricing' — calling
    #    both proves a single chunk program holds several real methods (the grouping worked).
    print("=== calling methods (2-txn groups; pricing methods share one chunk) ===")
    call_split("mark_price", 1, 1000, 1)
    call_split("bid_price", 1, 1000, 2)  # same chunk as mark_price
    call_split("accrue_funding", 2, 5, 3)
    call_split("health_factor", 7, 250, 4)
    n_chunks = len(chunk_names)
    print(f"PASS: {len(methods)} methods across {n_chunks} feature chunks, via multi-page swaps")

    # 5. CONTRACT client: a separate on-chain app (Consumer) calls mark_price through uros,
    #    composing [setup.prepare, main.mark_price] as inner txns — shipping no chunk code.
    consumer_factory = au.AppFactory(
        au.AppFactoryParams(
            algorand=algorand,
            app_spec=au.Arc56Contract.from_json((OUT / "Consumer.arc56.json").read_text()),
            default_sender=sender,
        )
    )
    consumer_client, _ = consumer_factory.send.bare.create()
    algorand.send.payment(
        au.PaymentParams(
            sender=sender, receiver=consumer_client.app_address, amount=au.AlgoAmount.from_algo(1)
        )
    )
    consumer_client.send.call(
        au.AppClientMethodCallParams(method="configure", args=[setup_app_id, main_app_id])
    )
    sel = selectors["mark_price"]
    market, qty = 1, 1000
    res = consumer_client.send.call(
        au.AppClientMethodCallParams(
            method="best_quote", args=[market, qty],
            # the outer txn carries the refs the (nested) inner swap reads, on setup's boxes
            app_references=[setup_app_id, main_app_id],  # 2 apps + 6 boxes = 8 refs (the cap)
            box_references=[
                au.BoxReference(setup_app_id, b"m" + sel),
                au.BoxReference(setup_app_id, b"pricing"),
                au.BoxReference(setup_app_id, b"clear"),
                *empties(3),
            ],
            static_fee=au.AlgoAmount.from_micro_algo(8000),  # outer + inner tree
        )
    )
    expected = evaluate(market, qty, 0, PRICE_MODEL)
    main_state = algorand.app.get_global_state(main_app_id)
    raw_c = main_state["last_caller"].value
    seen_caller = bytes.fromhex(raw_c) if isinstance(raw_c, str) else bytes(raw_c)
    assert res.abi_return == expected, "consumer: returned quote mismatch"
    assert seen_caller == pubkey(consumer_client.app_address), "consumer: main didn't see consumer"
    print(f"contract client: Consumer.best_quote -> {res.abi_return} OK "
          "(RiskEngine saw the Consumer app as caller)")

    # 6. the guard rejects a method NOT preceded by setup.prepare. main currently holds the
    #    pricing chunk (from the Consumer call above), so mark_price is real and its guard fires;
    #    pass a non-prepare setup call as the `appl` arg -> selector mismatch fails the guard.
    #    `set_main` is creator-only and a no-op with the current main_app_id.
    def bad_guard() -> None:
        bogus = setup_client.params.call(
            au.AppClientMethodCallParams(method="set_main", args=[main_app_id])
        )
        main_client.send.call(
            au.AppClientMethodCallParams(
                method="mark_price", args=[bogus, 1, 1000], box_references=empties(4),
            )
        )

    try:
        bad_guard()
    except Exception:
        print("OK rejected: method not preceded by setup.prepare")
    else:
        raise AssertionError("guard accepted a method not preceded by setup.prepare")

    # 7. the check is on the txn IMMEDIATELY before — not "a prepare somewhere in the group".
    #    Group = [setup.prepare, mark_price, ask_price]: a real prepare IS present, mark_price
    #    runs (its predecessor is that prepare), but ask_price's predecessor is mark_price, so
    #    ask_price's guard rejects -> the whole group reverts. ask_price is built as a RAW app
    #    call so the composer doesn't auto-place its own prepare; its `appl` arg binds to the
    #    txn right before it (mark_price).
    def bad_adjacency() -> None:
        g = algorand.new_group()
        g.add_app_call_method_call(  # adds [prepare, mark_price]; prepare targets pricing
            main_client.params.call(
                au.AppClientMethodCallParams(
                    method="mark_price", args=[prepare_call("mark_price"), 1, 1000],
                    box_references=empties(4),
                )
            )
        )
        g.add_app_call(  # raw ask_price; its appl arg = the preceding mark_price, not a prepare
            au.AppCallParams(
                sender=sender,
                app_id=main_app_id,
                args=[selectors["ask_price"], (1).to_bytes(8, "big"), (1000).to_bytes(8, "big")],
                static_fee=au.AlgoAmount.from_micro_algo(1000),
            )
        )
        g.send()

    try:
        bad_adjacency()
    except Exception as e:
        # the failure must be the assert in txn[2] (ask_price) — whose predecessor is mark_price,
        # not the prepare at txn[0]. (The comment "must be preceded by setup.prepare" shows in
        # algokit's log; the exception text carries the txn index + assert.)
        msg = " ".join(str(e).split())
        assert "transaction [2]" in msg and "assert" in msg, f"wrong rejection: {msg[:160]}"
        print("OK rejected: prepare present but NOT immediately before the method")
        print("   txn[2]=ask_price hit its guard (predecessor is mark_price, not txn[0] prepare)")
    else:
        raise AssertionError("guard accepted a non-adjacent prepare")


if __name__ == "__main__":
    main()
