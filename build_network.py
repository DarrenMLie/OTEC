import torch.nn as nn


def build_network(layers, neurons, activation):
    """
  A function that builds a network with nn.append and nn.sequential.

  layers      -     number of fully connected layers
  neurons     -     number of neurons in each fully connected layer
  activation  -     nonlinear activation function 
  """

    # initialize activation function
    if activation == 'ReLU':
        nl_act = nn.ReLU()

    elif activation == 'Tanh':
        nl_act = nn.Tanh()

    elif activation == 'Sigmoid':
        nl_act = nn.Sigmoid()
    else:
        raise ValueError('Activation is not - ReLU, Tanh, or Sigmoid')

    # add layers to the network
    lay = []
    sizes = [10, 30, 50, 60, 70, 90, 90, 70, 60, 50, 30, 10]

    # input takes 4 variables (lat, long, depth, time)
    lay.append(nn.Linear(4, neurons))
    lay.append(nl_act)

    for l in range(len(sizes)-1):
        lay.append(nn.Linear(sizes[l], sizes[l+1]))
        lay.append(nl_act)

    # append fully connected layers with activation
    # for l in range(layers):
    #     lay.append(nn.Linear(neurons, neurons))
    #     lay.append(nl_act)

    # output gives 1 variable (temp)
    lay.append(nn.Linear(neurons, 1))

    net = nn.Sequential(*lay)
    return net
