import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


class EarlyStopper:
    def __init__(self, patience=5, delta=0., output_dir="../models"):
        # create output directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.patience = patience
        self.counter = 0
        self.delta = delta
        self.min_val_loss = np.inf

    def save_model(self, state_dict, model_name):
        """
        save the model

        Parameters
        ----------
        state_dict: dict
            a state_dict of the model that records model's parameters
        model_name: str
            name of the model
        """
        checkpoint_path = self.output_dir + f"/{model_name}_best.pt"
        torch.save(state_dict, checkpoint_path)

    def early_stop(self, model, val_loss, epoch, model_name):
        """
        check if it needs to early stop the training process.
        save current model if it is the best so far

        Parameters
        ----------
        model: PyTorch Model
            the model to test & save
        val_loss: PyTorch Float Tensor
            validation loss
        epoch: int
            current epoch index
        model_name: str
            name of the model

        Returns
        -------
        output: bool
            True if it needs to early stop the training process
        """
        if val_loss < self.min_val_loss + self.delta:
            print(f'=> Model at epoch {epoch + 1} is the best according to validation loss')
            self.save_model(state_dict=model.state_dict(), model_name=model_name)
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def test(model, criterion, dataloader, device):
    """
    calculate validation/test loss

    Parameters
    ----------
    model: PyTorch Model
        model to test
    criterion: PyTorch Loss Function
        the loss function to use
    dataloader: PyTorch DataLoader
        validation/test set dataloader
    device: PyTorch Device (torch.device)
        device to use, either CPU or GPU

    Returns
    -------
    test_loss: PyTorch Float Tensor
        validation/test loss
    """
    num_batches = len(dataloader)

    # Put the model in evaluation mode - no drop out during eval
    model.eval()

    # validation/test loss
    test_loss = 0

    with torch.no_grad():
        for img, img_id in dataloader:
            # move tensors to GPU if possible
            img = img.to(device)

            pred = model(img)
            # pred = torch.squeeze(pred)  # remove extra dim
            test_loss += criterion(pred, img).item()

    test_loss /= num_batches

    return test_loss


def train(model, criterion, optimizer, epochs, train_dataloader, val_dataloader, early_stopper, device, model_name):
    """
    train the model

    Parameters
    ----------
    model: PyTorch Model
        model to test
    criterion: PyTorch Loss Function
        the loss function to use
    optimizer: PyTorch optimizer
        the optimizer to use
    epochs: int
        number of epochs
    train_dataloader: PyTorch DataLoader
        training set dataloader
    val_dataloader: PyTorch DataLoader
        validation set dataloader
    early_stopper: Early Stopper
        an early stopper. It has a 'early_stop' method to check whether it needs to early stops the training process
    device: PyTorch Device (torch.device)
        device to use, either CPU or GPU
    model_name: str
        the name of the model

    Returns
    -------
    train_stats: list
        stats of the entire training process, include training loss and validation loss
    """
    num_batches = len(train_dataloader)

    # training stats
    train_stats = []

    # move model to GPU if possible
    model.to(device)

    for epoch in range(epochs):
        # print current epoch stats
        print("")
        print(f"======== Epoch {epoch + 1} / {epochs} ========")

        # Set the model to training mode
        model.train()

        # training loss
        train_loss = 0

        for batch, (img, img_id) in enumerate(train_dataloader):
            # move tensors to GPU if possible
            img = img.to(device)

            # Compute prediction and loss
            pred = model(img)
            # pred = torch.squeeze(pred)  # remove extra dim
            loss = criterion(pred, img)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print current batch stats
            if batch % 10 == 0:
                print(f"loss: {loss.item():>7f}  [{batch + 1:>2d}/{num_batches:>2d}]")

        # calculate training loss & validation loss
        train_loss /= num_batches
        test_loss = test(model, criterion, val_dataloader, device)

        # print current epoch stats
        print(f"Training Loss: {train_loss}")
        print(f"Validation Loss: {test_loss}")

        # record stats
        train_stats.append({"epoch": epoch + 1, "train loss": train_loss, "validation loss": test_loss})

        # check if it needs to early stop the training process
        if early_stopper.early_stop(model, test_loss, epoch, model_name):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_stats


def plot_train_val_loss(train_stats, figure_path):
    """
    plot training & validation loss vs. epoch

    Parameters
    ----------
    train_stats: list
        a list that records stats in the training process
    figure_path: str


    Returns
    -------

    """
    # transform to dataframe
    data = pd.DataFrame(data=train_stats)

    # plot
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.plot(data['train loss'], '-o', label="Training Loss")
    plt.plot(data['validation loss'], '-o', label="Validation Loss")

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(figure_path)
    plt.show()
