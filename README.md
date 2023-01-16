# neuron-reordering
Functions for harmonizing the order of neurons in two neural networks

This reimplements the methods presented in the paper 
[``Git Re-Basin: Merging Models modulo Permutation Symmetries''](https://arxiv.org/abs/2209.04836).

That paper is motivated by the observation that you can permute the neurons
in a hidden layer of a neural network without changing the behavior of the
network. Given two neural networks of the same architecture, we can therefore
sort the neurons of each hidden layer in such a way that the parameters of
the two networks become as similar as possible.

Once you have reshuffled the order of each hidden layer in a network to make
it more similar to another network, you can take mixtures of the two
parameter sets. The claim of the paper is that such mixtures in parameter
space have good performance if both mixture components have been trained
to completion, and if they neurons have been permuted to maximize the
similarity of the two sets of network parameters.

I have not found that this actually holds. In the experiments I have
performed with the code presented here, there is occasionally an effect
of that sort, but it is not strong or reliable.