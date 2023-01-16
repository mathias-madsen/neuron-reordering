"""
Wrappers for mixtures of multilayer perceptrons in parameter space.
"""

import torch


class LinearMixtures(torch.nn.Module):
    """ Convenience class for interpolating between two linear models. """

    def __init__(self, linear1, linear2, nsteps):
        super().__init__()
        self.linear1 = linear1
        self.linear2 = linear2
        self.nsteps = nsteps
        w1, b1 = linear1.weight, linear1.bias
        w2, b2 = linear2.weight, linear2.bias
        p = torch.linspace(1, 0, nsteps)
        q = 1 - p
        self.w = p[:, None, None]*w1 + q[:, None, None]*w2
        self.b = p[:, None]*b1 + q[:, None]*b2
    
    def __repr__(self):
        return ("LinearMixtures(linear1=%s, linear2=%s, nsteps=%s)" %
                (self.linear1, self.linear2, self.nsteps))

    def __call__(self, x):
        if len(x.shape) < 2:
            x = x[None,]
        return x @ self.w.swapaxes(-2, -1) + self.b[:, None, :]


class MLPMixtures(torch.nn.Sequential):
    """ Convenience class for interpolating between two MLPs. """

    def __init__(self, mlp1, mlp2, nsteps):
        layers = []
        for l1, l2 in zip(mlp1, mlp2):
            assert type(l1) == type(l2)
            if type(l1) == torch.nn.Linear:
                assert l1.in_features == l2.in_features
                assert l1.out_features == l2.out_features
                layers.append(LinearMixtures(l1, l2, nsteps))
            else:
                layers.append(l1)
        super().__init__(*layers)
   