import torch


def build_optimizer(network, optimizer, lr):
    if optimizer == 'sgd':
        optim = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.95)

    elif optimizer == 'adam':
        # set default optimizer to adam
        optim = torch.optim.Adam(network.parameters(), lr=lr)

    else:
        raise ValueError('Optimizer is not initialized to "sgd" or "adam"')

    return optim
