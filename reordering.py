"""
This module contains the functions that define how two interpret a
permutation matrix as a reordering of the neurons between two affine
layers in a nerual network.

In particular, it settles conventions for whether P or P.T should be
multiplied onto the weight matrices, etc.
"""

import torch
import numpy as np


def get_in_weights_reward(in1: torch.nn.Linear,
                          in2: torch.nn.Linear) -> torch.Tensor:

    return in2.weight @ in1.weight.T


def get_in_biases_reward(in1: torch.nn.Linear,
                         in2: torch.nn.Linear) -> torch.Tensor:

    return torch.outer(in2.bias, in1.bias)


def get_out_weights_reward(out1: torch.nn.Linear,
                           out2: torch.nn.Linear) -> torch.Tensor:

    return out2.weight.T @ out1.weight


def get_reward_matrix(in1: torch.nn.Linear,
                      out1: torch.nn.Linear,
                      in2: torch.nn.Linear,
                      out2: torch.nn.Linear) -> torch.Tensor:
    """ Formulate the cost matrix for a neuron reordering problem. """

    weight_reward_in = get_in_weights_reward(in1, in2)
    weight_reward_out = get_out_weights_reward(out1, out2)

    if in1.bias is not None:
        bias_reward = get_in_biases_reward(in1, in2)
        return weight_reward_in + weight_reward_out + bias_reward
    else:
        return weight_reward_in + weight_reward_out


def reorder_incoming(linear_layer: torch.nn.Linear,
                     permutation_matrix: torch.Tensor) -> None:
    """ Permute neurons receiving activation from an incoming layer. """

    state = {"weight": permutation_matrix @ linear_layer.weight}

    if linear_layer.bias is not None:
        state.update({"bias": permutation_matrix @ linear_layer.bias})

    linear_layer.load_state_dict(state)


def reorder_outgoing(linear_layer: torch.nn.Linear,
                     permutation_matrix: torch.Tensor) -> None:
    """ Permute neurons feeding activation into an outgoing layer. """

    state = {"weight": linear_layer.weight @ permutation_matrix.T}

    if linear_layer.bias is not None:
        state.update({"bias": linear_layer.bias})

    linear_layer.load_state_dict(state)


def _test_get_in_weights_reward() -> None:

    n, k = np.random.randint(1, 10, size=2)

    a1 = torch.nn.Linear(n, k)
    a2 = torch.nn.Linear(n, k)

    reward_matrix = get_in_weights_reward(a1, a2)
    assert reward_matrix.shape == (k, k)

    dots_original = torch.sum(a1.weight * a2.weight)
    assert torch.allclose(dots_original, reward_matrix.trace())

    plist = np.random.permutation(k)
    pmat = torch.eye(k)[:, plist]
    reorder_incoming(a1, pmat)

    dots_permuted = torch.sum(a1.weight * a2.weight)
    coords = torch.sum(pmat * reward_matrix)
    assert torch.allclose(dots_permuted, coords)


def _test_get_in_bias_reward() -> None:

    n, k = np.random.randint(1, 10, size=2)

    a1 = torch.nn.Linear(n, k)
    a2 = torch.nn.Linear(n, k)

    reward_matrix = get_in_biases_reward(a1, a2)
    assert reward_matrix.shape == (k, k)

    dots_original = torch.sum(a1.bias * a2.bias)
    assert torch.allclose(dots_original, reward_matrix.trace())

    plist = np.random.permutation(k)
    pmat = torch.eye(k)[:, plist]
    reorder_incoming(a1, pmat)

    dots_permuted = torch.sum(a1.bias * a2.bias)
    coords = torch.sum(pmat * reward_matrix)
    assert torch.allclose(dots_permuted, coords)


def _test_get_out_weights_reward() -> None:

    k, m = np.random.randint(1, 10, size=2)

    b1 = torch.nn.Linear(k, m)
    b2 = torch.nn.Linear(k, m)

    reward_matrix = get_out_weights_reward(b1, b2)
    assert reward_matrix.shape == (k, k)

    dots_original = torch.sum(b1.weight * b2.weight)
    assert torch.allclose(dots_original, reward_matrix.trace())

    plist = np.random.permutation(k)
    pmat = torch.eye(k)[:, plist]
    reorder_outgoing(b1, pmat)

    dots_permuted = torch.sum(b1.weight * b2.weight)
    coords = torch.sum(pmat * reward_matrix)
    assert torch.allclose(dots_permuted, coords)


def _test_get_reward_matrix() -> None:

    n, k, m = np.random.randint(1, 10, size=3)

    a1 = torch.nn.Linear(n, k)
    b1 = torch.nn.Linear(k, m)

    a2 = torch.nn.Linear(n, k)
    b2 = torch.nn.Linear(k, m)

    reward_matrix = get_reward_matrix(a1, b1, a2, b2)

    bias_1_sim = torch.sum(a1.bias * a2.bias)
    weight_1_sim = torch.sum(a1.weight * a2.weight)
    weight_2_sim = torch.sum(b1.weight * b2.weight)
    sim_eye = bias_1_sim + weight_1_sim + weight_2_sim
    reward_eye = torch.sum(torch.eye(k) * reward_matrix)
    assert torch.isclose(sim_eye, reward_eye)

    permutation = torch.eye(k)[np.random.permutation(k),]
    reorder_incoming(a1, permutation)
    reorder_outgoing(b1, permutation)
    bias_1_sim = torch.sum(a1.bias * a2.bias)
    weight_1_sim = torch.sum(a1.weight * a2.weight)
    weight_2_sim = torch.sum(b1.weight * b2.weight)
    sim_perm = bias_1_sim + weight_1_sim + weight_2_sim
    reward_perm = torch.sum(permutation * reward_matrix)
    assert torch.isclose(sim_perm, reward_perm)


if __name__ == "__main__":

    _test_get_in_weights_reward()
    _test_get_in_bias_reward()
    _test_get_out_weights_reward()
    _test_get_reward_matrix()