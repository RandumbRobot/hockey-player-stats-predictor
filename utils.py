import torch
from torch import nn, sqrt, div
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



"""##############################
TRAINING AND VALIDATION
"""##############################


def val_epoch(model, dataloader, loss, device):
  val_losses = []

  for idx,batch in enumerate(dataloader): # for each validation batch

    x, y = batch[0].to(device), batch[1].to(device)

    preds = model(x)

    val_loss = loss(preds, y)
    val_losses.append(val_loss.item())

  return val_losses



def train_and_validate(model, criterion, optimizer, dataloader_train, dataloader_test, epochs=100, loss_interval=25, device="cpu"):
    """
    Train and validate and model

    Arguments:
        model: model to train and validate
        criterion: loss function for the model
        optimizer: optimizer for the model
        dataloader_train: dataloader for train data
        dataloader_test: dataloader for test data
        epochs: number of epochs to train
        loss_interval: number of epochs between saving the training and validation loss.
        device: device on which to run the network
    

    Returns:
        (trained_model, train_loss, val_loss)
    """
    N=5 # TODO: change dataloader dynamically based on multiple N. Create a list of dataloader and mix-n-match the N accross batches

    #training loop
    # initialize the hidden state.
    #hidden = (torch.zeros(1, batch_size, hidden_size))

    train_losses = []
    val_losses = []

    for i in range(epochs): # for each epoch
        t_losses = []
        for idx,batch in enumerate(dataloader_train): # for each batch

            x, y = batch[0].to(device), batch[1].to(device)

            # clear gradients
            model.zero_grad()

            # predict
            #y_pred, hidden = model(batch[0], hidden)
            y_pred = model(x)

            # get loss
            loss = criterion(y_pred, y)
            t_losses.append(loss.item())
            loss = loss.mean()

            # propagate and train
            loss.backward()
            optimizer.step()
            

        if i%(loss_interval*2) == 0:
            print(i,"th epoch : ", np.mean(t_losses))

        # Validation and losses
        if i%loss_interval == 0:
            vlosses = val_epoch(model, dataloader_test, criterion, device)
            val_losses.append(np.mean(vlosses))
            train_losses.append(np.mean(t_losses))
            print(f"Validation loss for epoch {i}: {np.mean(vlosses)}")

    return model, train_losses, val_losses







"""##############################
PLOTTING
"""##############################


def plot_train_v_loss(name, train_losses, val_losses, loss_epoch_step_size, baseline_loss=None):
    """
    Plots the train VS loss curves

    name: name of the model
    train_loss: list of losses during training
    val_loss: list of validation losses
    loss_epoch_step_size: epoch step size for sampling the loss

    NOTE: train_loss and val_loss must be the same size

    """
    #@title
    f, axs = plt.subplots(1,1, figsize=(15,5))
    font = {'size' : 14}
    matplotlib.rc('font', **font)

    axs = (axs, 0)

    xticks = np.arange(0, len(val_losses)*loss_epoch_step_size, step=loss_epoch_step_size)

    axs[0].plot(xticks, train_losses, lw=3, ms=8, marker='o',
            color='orange', label='Train')
    axs[0].set_ylabel("Loss")
    axs[0].plot(xticks, val_losses, lw=3, ms=10, marker='^',
            color='purple', label='Validation')
    
    axs[0].set_title(f'{name}\nTrain/Val Loss Over Time')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylim(max(0, min(train_losses+val_losses)-0.1), min(np.mean(train_losses+val_losses)+0.2, max(train_losses+val_losses)))
    axs[0].grid()
    
    if baseline_loss is not None:
        axs[0].axhline(y=baseline_loss, color='r', linestyle='--', label='Averaging Baseline Loss')

    plt.legend()
    plt.show()



def get_targets_preds_pairs(dataset, model, device):
    # Get all the target/pred pairs for the current model
    preds_n_targets = {}
    for N in dataset.NL:
        preds_n_targets[N] = []

        preds = []
        targets = []
        for team in dataset.data:
            for sample in dataset.data[team][N]:
                # Batch of size 1 because of LSTM bug
                x = dataset.data[team][N][sample][0]
                x = x.reshape((1,x.size(0),x.size(1))).to(device)

                pred = model(x)
                preds.append(pred)
                targets.append(dataset.data[team][N][sample][1])
        
        preds = torch.stack(preds).to('cpu').detach()
        preds = preds.reshape((preds.size(0), preds.size(2)))
        targets = torch.stack(targets).to('cpu').detach()
        preds_n_targets[N] = (preds, targets)

    return preds_n_targets


def plot_divergence(NL, pred_n_targets_dict, criterion):
    """
    Plots the divergence between the predictions and the targets. It includes the max positive and negative deviations.

    NL: list of the N values used for targets/preds
    pred_n_targets_dict: dictionary where each entry is an N from NL and contains the list of all target/prediction pairs
    criterion: loss function object

    NOTE: NL, targets and preds must all have the same size
    """

    for N in NL:
        y_hat, y = pred_n_targets_dict[N] # get targets for batches of size N

        for stat in range(len(y[0])):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f'Predictions for N={N}, stat: {stat}', fontsize=18)

            plt.xlabel('Actual Results', fontsize=14)
            plt.ylabel('Predicted Results', fontsize=14)

            target_stats = y[:, stat]
            preds_stats = y_hat[:, stat]

            #print(target_stats)
            #print(preds_stats)
            
            # Absolute value difference
            diffs = target_stats - preds_stats
            yval = target_stats + diffs
            #yval = diffs
            # Plot overshoot/undershoot vs actual
            ax1.scatter(target_stats, yval, c="b",s=10)
            ax1.plot(target_stats,target_stats)
            ax1.set_xlim(min(target_stats),max(target_stats))
            ax1.set_ylim(min(yval),max(yval))

            # Relative value difference
            diffs = div(target_stats - preds_stats, target_stats)*100
            yval = diffs
            ax2.scatter(target_stats, yval, c="b",s=10)
            ax2.set_xlim(min(target_stats),max(target_stats))
            ax2.set_ylim(min(yval),max(yval))

            #ax1.annotate('loss =', xy=(5, 93),fontsize=16)
            #ax1.annotate('loss =', fontsize=16)
            loss=criterion(preds_stats, target_stats)
            #ax1.annotate(loss, xy=(5, 82),fontsize=16)
            #ax1.annotate(loss,fontsize=16)
            plt.grid(True)
            plt.show()
            

"""##############################
CUSTOM LOSS FUNCTIONS
"""##############################

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return sqrt(self.mse(yhat,y))