import torch
import numpy as np
from matplotlib import pyplot as plt

from ordersolve import reorder_like
from interpolations import MLPMixtures


if __name__ == "__main__":

    # ==== Define a ground-truth function and sample its values: ====

    def true_function(x):
        return torch.sin((np.pi * x)**2)

    xmin = -1.0
    xmax = +1.0

    trainx = (xmax - xmin)*torch.rand(512, 1) + xmin
    trainy = true_function(trainx)

    testx = (xmax - xmin)*torch.rand(512, 1) + xmin
    testy = true_function(testx)

    xspan = torch.linspace(xmin, xmax, 75)[:, None]
    yspan = true_function(xspan).flatten().detach().numpy()

    # ==== Create several identical networks and train them: ====

    networks = [
        torch.nn.Sequential(
            torch.nn.Linear(1, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 17),
            torch.nn.ReLU(),
            torch.nn.Linear(17, 19),
            torch.nn.ReLU(),
            torch.nn.Linear(19, 13),
            torch.nn.ReLU(),
            torch.nn.Linear(13, 1),
        )
        for _ in range(5)]

    for num, model in enumerate(networks):
        print("Training network %s . . ." % (num + 1,))
        opt1 = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
        for step in range(3000):
            xbatches = torch.split(trainx, 16)
            ybatches = torch.split(trainy, 16)
            for x, y in zip(xbatches, ybatches):
                opt1.zero_grad()
                yhat = model(x)
                loss = torch.mean((yhat - y) ** 2)
                loss.backward()
                opt1.step()
            if (step + 1) % 100 == 0:
                deteached_loss = float(loss)
                print(deteached_loss)
        print("Done.\n")

    plt.figure(figsize=(12, 5))
    plt.plot(trainx.flatten(), trainy.flatten(), "ko", alpha=0.5)
    modelx = torch.linspace(xmin, xmax, 1000)[:, None]
    for network in networks:
        modely = network(modelx).detach().numpy()
        plt.plot(modelx.flatten(), modely.flatten(), "-", alpha=0.5)
    plt.savefig("trained_models.png")
    plt.show()

    # ==== Interpolate model pairs with and without reordering: ====

    nsteps = 250

    pairs = [
        (networks[i], networks[j])
        for i in range(len(networks))
        for j in range(i + 1, len(networks))
        ]

    mixs1 = [MLPMixtures(*p, nsteps) for p in pairs]
    yhats1 = [f(xspan).squeeze().detach().numpy() for f in mixs1]
    mses1 = [np.mean((yspan - yhat) ** 2, axis=1) for yhat in yhats1]

    for network1, network2 in pairs:
        for num_step in range(10):
            anything_changed = reorder_like(network1, network2)
            if not anything_changed:
                print("Breaking off after %s steps" % (num_step + 1,))
                break
    print()

    mixs2 = [MLPMixtures(*p, nsteps) for p in pairs]
    yhats2 = [f(xspan).squeeze().detach().numpy() for f in mixs2]
    mses2 = [np.mean((yspan - yhat) ** 2, axis=1) for yhat in yhats2]

    # ==== Visualize the effects of neuron reordering: ====

    for mse in mses1:
        plt.plot(np.linspace(0, 1, nsteps), mse, "r-", alpha=0.5)
    plt.plot([], [], "r-", label="without reordering")
    for mse in mses2:
        plt.plot(np.linspace(0, 1, nsteps), mse, "b-", alpha=0.5)
    plt.plot([], [], "b-", label="with reordering")
    plt.xlabel("mixture proportion")
    plt.ylabel("$MSE$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mses_with_and_without_reordering.png")
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(xspan.flatten(), yspan.flatten(), "k-", lw=5,
            alpha=0.8, label="true function")
    for yhat in yhats1:
        plt.plot(xspan.flatten(), yhat[nsteps // 2,], "r-", lw=3, alpha=0.5)
    plt.plot([], [], "r-", alpha=0.3, label="without reordering")
    for yhat in yhats2:
        plt.plot(xspan.flatten(), yhat[nsteps // 2,], "b-", lw=3, alpha=0.5)
    plt.plot([], [], "b-", alpha=0.3, label="with reordering")
    plt.legend()
    plt.title("50/50 parameter mixtures")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.savefig("midway_functions.png")
    plt.show()

    ymix_without = yhats1[0]
    ymix_with = yhats2[0]

    plt.close("all")
    vmin = min(ymix_without.min(), ymix_with.min())
    vmax = max(ymix_without.max(), ymix_with.max())
    figure, (left, right) = plt.subplots(figsize=(12, 5), ncols=2)
    left.imshow(ymix_without.T, interpolation="nearest",
                vmin=vmin, vmax=vmax, aspect="auto")
    left.set_xlabel("proportion network 1 vs network 2")
    left.set_title("without reordering")
    left.set_xticks([0, nsteps - 1])
    left.set_xticklabels(["0.0", "1.0"])
    right.imshow(ymix_with.T, interpolation="nearest",
                vmin=vmin, vmax=vmax, aspect="auto")
    right.set_xlabel("proportion network 1 vs network 2")
    right.set_title("with reordering")
    right.set_xticks([0, nsteps - 1])
    right.set_xticklabels(["0.0", "1.0"])
    plt.tight_layout()
    plt.savefig("interpolations.png")
    plt.show()
