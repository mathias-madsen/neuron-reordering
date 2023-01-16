"""
This module contains a function for solving a neuron reordering problem.

It also contains test that show how to apply that solution in order to
increase the order-similarity between two pairs of linear layers.
"""

import torch
import numpy as np
from scipy.optimize import linprog

from matrix_reductions import get_row_col_sum_matrix
from reordering import get_reward_matrix, reorder_incoming, reorder_outgoing


def compute_neuron_permutation(in1: torch.nn.Linear,
                               out1: torch.nn.Linear,
                               in2: torch.nn.Linear,
                               out2: torch.nn.Linear) -> np.ndarray:

    # get the costs associated with each permutation:
    reward_matrix = get_reward_matrix(in1, out1, in2, out2)
    cost_vector = (-1) * reward_matrix.flatten().detach()

    # formulate the equality constraints:
    dim = in1.out_features
    A_eq = get_row_col_sum_matrix(dim, dim)
    b_eq = np.ones(dim + dim)

    solution = linprog(cost_vector,
                       A_eq=A_eq[:-1],  # last constr redundant
                       b_eq=b_eq[:-1],  # last constr redundant
                       bounds=(0, 1))

    return solution.x.round().reshape([dim, dim])


def reorder_like(mlp1: torch.nn.Sequential,
                 mlp2: torch.nn.Sequential) -> bool:
    """ Reorder all layers of mlp1 once to match mlp2 better. """

    affines1 = [lay for lay in mlp1 if type(lay) == torch.nn.Linear]
    affines2 = [lay for lay in mlp2 if type(lay) == torch.nn.Linear]

    # check that the architectures match:
    assert len(mlp1) == len(mlp2)
    assert len(affines1) == len(affines2)
    assert ([lay.in_features for lay in affines1] ==
            [lay.in_features for lay in affines2])
    assert ([lay.out_features for lay in affines1] ==
            [lay.out_features for lay in affines2])

    if len(affines1) < 2:
        return False  # no hidden layers to reorder

    changed = []
    pairs1 = zip(affines1[:-1], affines1[1:])
    pairs2 = zip(affines2[:-1], affines2[1:])
    for (a1, b1), (a2, b2) in zip(pairs1, pairs2):
        permhat = compute_neuron_permutation(a1, b1, a2, b2)
        permhat = torch.tensor(permhat, dtype=a1.weight.dtype)
        reorder_incoming(a1, permhat)
        reorder_outgoing(b1, permhat)
        eye = torch.eye(permhat.shape[0])
        changed.append(not torch.allclose(eye, permhat))
    
    return any(changed)


def _test_reorder_like() -> None:

    model1 = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 7),
        torch.nn.ReLU(),
        torch.nn.Linear(7, 2),
    )

    model2 = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 7),
        torch.nn.ReLU(),
        torch.nn.Linear(7, 2),
    )

    changes = [reorder_like(model1, model2) for _ in range(10)]
    assert changes == sorted(changes, reverse=True)  # True => False


def _test_compute_neuron_permutation_with_known_solution() -> None:

    n, k, m = 13, 7, 11

    a1 = torch.nn.Linear(n, k)
    b1 = torch.nn.Linear(k, m)

    a2 = torch.nn.Linear(n, k)
    b2 = torch.nn.Linear(k, m)

    # select a random permutation and modify network 1:

    perm = torch.eye(k)[np.random.permutation(k),]

    a1.load_state_dict({
        "weight": perm @ a2.weight,
        "bias": perm @ a2.bias,
        })

    b1.load_state_dict({
        "weight": b2.weight @ perm.T,
        "bias": b2.bias,
        })

    # check that it was shuffled -- will fail 1/13! times
    assert not torch.allclose(a1.weight, a2.weight)
    assert not torch.allclose(a1.bias, a2.bias)
    assert not torch.allclose(b1.weight, b2.weight)

    # the optimizer recovers the permutation matrix:

    permhat = compute_neuron_permutation(a1, b1, a2, b2)
    assert np.allclose(permhat.sum(axis=0), 1)
    assert np.allclose(permhat.sum(axis=1), 1)
    assert np.allclose(perm.numpy(), permhat.T)

    # the permutation matrix restores order:

    permhat = torch.tensor(permhat, dtype=a1.weight.dtype)

    reorder_incoming(a1, permhat)
    reorder_outgoing(b1, permhat)
    
    assert torch.allclose(a1.weight, a2.weight)
    assert torch.allclose(a1.bias, a2.bias)
    assert torch.allclose(b1.weight, b2.weight)


def _test_that_compute_neuron_permutation_decreases_cost() -> None:

    n, k, m = np.random.randint(1, 10, size=3)

    a1 = torch.nn.Linear(n, k)
    b1 = torch.nn.Linear(k, m)

    a2 = torch.nn.Linear(n, k)
    b2 = torch.nn.Linear(k, m)

    bias_1_sim = torch.sum(a1.bias * a2.bias)
    weight_1_sim = torch.sum(a1.weight * a2.weight)
    weight_2_sim = torch.sum(b1.weight * b2.weight)
    sim_before = bias_1_sim + weight_1_sim + weight_2_sim

    permutation = compute_neuron_permutation(a1, b1, a2, b2)
    permutation = torch.tensor(permutation, dtype=a1.weight.dtype)
    reorder_incoming(a1, permutation)
    reorder_outgoing(b1, permutation)

    bias_1_sim = torch.sum(a1.bias * a2.bias)
    weight_1_sim = torch.sum(a1.weight * a2.weight)
    weight_2_sim = torch.sum(b1.weight * b2.weight)
    sim_after = bias_1_sim + weight_1_sim + weight_2_sim

    assert sim_after >= sim_before


def _test_compute_neuron_permutation_with_zero_weights() -> None:

    n, k, m = np.random.randint(1, 10, size=3)

    a1 = torch.nn.Linear(n, k)
    b1 = torch.nn.Linear(k, m)

    a2 = torch.nn.Linear(n, k)
    b2 = torch.nn.Linear(k, m)

    a1.load_state_dict({"weight": 0 * a1.weight, "bias": a1.bias})
    b1.load_state_dict({"weight": 0 * b1.weight, "bias": b1.bias})
    a2.load_state_dict({"weight": 0 * a2.weight, "bias": a2.bias})
    b2.load_state_dict({"weight": 0 * b2.weight, "bias": b2.bias})

    bias_similarity_before = torch.sum(a1.bias * a2.bias)
    assert torch.allclose(torch.sum(a1.weight * a2.weight), torch.zeros([]))
    assert torch.allclose(torch.sum(b1.weight * b2.weight), torch.zeros([]))

    permutation = compute_neuron_permutation(a1, b1, a2, b2)
    permutation = torch.tensor(permutation, dtype=a1.weight.dtype)
    reorder_incoming(a1, permutation)
    reorder_outgoing(b1, permutation)

    bias_similarity_after = torch.sum(a1.bias * a2.bias)
    assert torch.allclose(torch.sum(a1.weight * a2.weight), torch.zeros([]))
    assert torch.allclose(torch.sum(b1.weight * b2.weight), torch.zeros([]))

    assert bias_similarity_after >= bias_similarity_before


def _test_compute_neuron_permutation_with_zero_bias() -> None:

    n, k, m = np.random.randint(1, 10, size=3)

    a1 = torch.nn.Linear(n, k, bias=False)
    b1 = torch.nn.Linear(k, m, bias=False)

    a2 = torch.nn.Linear(n, k, bias=False)
    b2 = torch.nn.Linear(k, m, bias=False)

    sim_before = (torch.sum(a1.weight * a2.weight) +
                  torch.sum(b1.weight * b2.weight))

    permutation = compute_neuron_permutation(a1, b1, a2, b2)
    permutation = torch.tensor(permutation, dtype=a1.weight.dtype)
    reorder_incoming(a1, permutation)
    reorder_outgoing(b1, permutation)

    sim_after = (torch.sum(a1.weight * a2.weight) +
                 torch.sum(b1.weight * b2.weight))

    assert sim_after >= sim_before


if __name__ == "__main__":

    _test_reorder_like()
    _test_compute_neuron_permutation_with_known_solution()
    _test_that_compute_neuron_permutation_decreases_cost()
    _test_compute_neuron_permutation_with_zero_bias()
    _test_compute_neuron_permutation_with_zero_weights()