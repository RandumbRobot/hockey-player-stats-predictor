from torch import nn, sqrt
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


def plot_train_v_loss(name, train_losses, val_losses, loss_epoch_step_size):
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

    plt.legend()
    plt.show()



def plot_divergence(NL, pred_n_targets_dict, criterion):
    """
    Plots the divergence between the predictions and the targets. It includes the max positive and negative deviations.

    NL: list of the N values used for targets/preds
    pred_n_targets_dict: dictionary where each entry is an N from NL and contains the list of all target/prediction pairs
    criterion: loss function object

    NOTE: NL, targets and preds must all have the same size
    """

    for N in NL:
        y, y_hat = pred_n_targets_dict[N] # get targets for batches of size N

        for sid,stat in enumerate(y):
            fig1 = plt.figure(figsize=(16,5))
            az = fig1.add_axes([0.36, 0.36, 0.3, 0.3]) # left, bottom, width, height (range 0 to 1)

            diffs = y-y_hat
            yval = y+diffs

            # Plot overshoot/undershoot vs actual
            az.scatter(y, yval, c="b",s=10)
            az.plot([0,50,120],[0,50,120])
            az.set_xlim(-5,110)
            az.set_ylim(-5,110)
            az.set_aspect('equal')
            plt.xlabel('Actual Results', fontsize=14)
            plt.ylabel('Predicted Results', fontsize=14)
            plt.title(f'Predictions for N={N}, stat: {sid}', fontsize=18)
            az.annotate('loss =', xy=(5, 93),fontsize=16)
            loss=criterion(y_hat[sid],y[sid])
            az.annotate(loss, xy=(5, 82),fontsize=16)
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