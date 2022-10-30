from dataset_manipulation.exploit_symmetries import give_all_symmetries
import pytest


@pytest.mark.parametrize(
    "features, optimal_ordering, all_symmetries", (
     ([1, 2, 3], 3, [[2, 3, 1], [2, 1, 3], [3, 2, 1], [1, 2, 3], [3, 1, 2], [1, 3, 2]]),
     ([1, 2, 3], 1, [[1, 3, 2], [1, 2, 3], [3, 1, 2], [2, 1, 3], [3, 2, 1], [2, 3, 1]]))
    )


def test_give_all_symmetries(features, optimal_ordering, all_symmetries):
    assert give_all_symmetries(features, optimal_ordering) == all_symmetries