import random
from collections.abc import Sequence

import algokit_utils as au
import pytest

from tests import AWST_DIR
from tests.utils.deployer import Deployer, encode_bare_args

_BOX = b"box"


@pytest.mark.parametrize(
    ("args", "error"),
    [
        # each of these selects an operation whose result is not used
        ([0, 0, 1], "- would result negative"),
        ([1, 2**64 - 1, 1], r"\+ overflowed"),
        ([2, 2**32, 2**32], r"\* overflowed"),
        ([3, 2, 64], "overflow"),
        ([4, 1, 0], "/ 0"),
        ([5, 1, 64], "shl arg too big"),
        ([6, 0, 0, b"123456789"], "btoi arg too long"),
        ([7, 1, 2, b"ab"], "extraction end 3 is beyond length: 2"),
        ([8, 0, 0, bytes(65)], "math attempted on large byte-array"),
    ],
)
def test_unused_operation_that_fails(
    deployer_o: Deployer, args: Sequence[bytes | int], error: str
) -> None:
    # Without the fix, the optimizer removes the operation with its failure,
    # and the program is created.
    with pytest.raises(au.LogicError, match=error):
        deployer_o.create_bare(AWST_DIR / "unused_fallible_ops", args=args)


def test_unused_operation_that_does_not_fail(deployer_o: Deployer) -> None:
    deployer_o.create_bare(AWST_DIR / "unused_fallible_ops", args=[0, 1, 1])


def _create_and_fund(
    deployer: Deployer, localnet: au.AlgorandClient, account: au.AddressWithSigners, name: str
) -> au.AppClient:
    client = deployer.create_bare(AWST_DIR / name).client
    # for the boxes and the inner transactions of the app
    localnet.account.ensure_funded(
        account_to_fund=client.app_address,
        dispenser_account=account,
        min_spending_balance=au.AlgoAmount.from_micro_algo(1_000_000),
    )
    return client


def _call(client: au.AppClient, *args: bytes | int) -> None:
    client.send.bare.call(
        au.AppClientBareCallParams(
            args=encode_bare_args(args),
            box_references=[_BOX],
            # covers an inner transaction
            static_fee=au.AlgoAmount.from_micro_algo(2_000),
            note=random.randbytes(8),
        )
    )


@pytest.fixture
def box_exists_client(
    deployer_o: Deployer, localnet: au.AlgorandClient, account: au.AddressWithSigners
) -> au.AppClient:
    return _create_and_fund(deployer_o, localnet, account, "box_exists_asserts")


def test_box_exists_assert_after_box_del(box_exists_client: au.AppClient) -> None:
    # Without the fix, an exists flag read before the delete lets the optimizer take
    # the box to exist afterwards, and the last assert is removed.
    with pytest.raises(au.LogicError, match="box exists after box_del"):
        _call(box_exists_client, 0)


def test_box_exists_assert_of_flag_read_before_box_create(box_exists_client: au.AppClient) -> None:
    # Without the fix, the assert of a flag read before the box was created is removed.
    with pytest.raises(au.LogicError, match="box existed before box_create"):
        _call(box_exists_client, 1)


def test_box_exists_assert_after_inner_transaction(box_exists_client: au.AppClient) -> None:
    _call(box_exists_client, 2)


def test_box_exists_assert_of_box_known_to_exist(box_exists_client: au.AppClient) -> None:
    _call(box_exists_client, 3)


@pytest.fixture
def unobserved_writes_client(
    deployer_o: Deployer, localnet: au.AlgorandClient, account: au.AddressWithSigners
) -> au.AppClient:
    return _create_and_fund(deployer_o, localnet, account, "unobserved_writes")


def test_box_write_followed_by_write_of_another_size(
    unobserved_writes_client: au.AppClient,
) -> None:
    # Without the fix, the first put is removed, and the second creates the box.
    with pytest.raises(au.LogicError, match="box_put wrong size"):
        _call(unobserved_writes_client, 0, 0)


def test_box_write_followed_by_write_of_same_size(unobserved_writes_client: au.AppClient) -> None:
    _call(unobserved_writes_client, 1, 10)


def test_box_write_seen_by_minimum_balance(unobserved_writes_client: au.AppClient) -> None:
    # Without the fix, the first put is removed, and the minimum balance
    # is the same before and after it.
    _call(unobserved_writes_client, 2, 0)


def test_state_write_around_inner_transaction(unobserved_writes_client: au.AppClient) -> None:
    _call(unobserved_writes_client, 3, 0)


def test_state_write_followed_by_write_of_another_type(
    unobserved_writes_client: au.AppClient,
) -> None:
    # The schema has no room for a byte slice.
    # Without the fix, the write of the byte slice is removed, and the call succeeds.
    with pytest.raises(au.LogicError, match="exceeds schema"):
        _call(unobserved_writes_client, 4, 0, b"bytes")


def test_state_write_followed_by_write_of_same_type(
    unobserved_writes_client: au.AppClient,
) -> None:
    _call(unobserved_writes_client, 5, 7)
