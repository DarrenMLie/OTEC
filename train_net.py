import wandb


def train_net(net, device, optimizer, loss_function, batch_loss, val_loss, loader):
    """
    A function that trains a regression neural network using stochatic gradient
    descent and returns the trained network. The loss function being minimized is
    `loss_func`.

    Parameters:

    loader     -    Data for one large batch of data
    """

    for X_batch, y_batch in loader:
        # put all data on device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = net(X_batch).to(device)

        # zero the optimizer for each batch
        optimizer.zero_grad()

        # loss = loss_function(y_batch, y_pred, config.reg_weight, net.parameters())
        # compute the loss and step the optimizer
        loss = loss_function(y_batch, y_pred)
        loss.backward()
        optimizer.step()

        # log the batch loss
        wandb.log({"batch loss": loss.item()})
        batch_loss.append(loss.item())

    # Return everything we need to analyze the results
    return batch_loss, val_loss
