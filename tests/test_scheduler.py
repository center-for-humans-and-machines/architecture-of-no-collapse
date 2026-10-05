"""Scheduler determinism and ordering tests."""

from renewal.core.scheduler import Scheduler


def test_same_seed_same_order():
    a = Scheduler(5, 42)
    b = Scheduler(5, 42)
    orders_a = [a.order() for _ in range(3)]
    orders_b = [b.order() for _ in range(3)]
    assert orders_a == orders_b


def test_different_seed_differs():
    a = Scheduler(5, 1)
    b = Scheduler(5, 2)
    assert a.order() != b.order()


def test_order_is_permutation():
    s = Scheduler(5, 42)
    for _ in range(20):
        assert sorted(s.order()) == [0, 1, 2, 3, 4]


def test_turn_index_sequential():
    s = Scheduler(3, 0)
    assert [s.next_turn() for _ in range(5)] == [0, 1, 2, 3, 4]


def test_needs_at_least_one_agent():
    import pytest

    with pytest.raises(ValueError):
        Scheduler(0, 0)
