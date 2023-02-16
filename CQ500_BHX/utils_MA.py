import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import fastprogress
import time
from sklearn import metrics
import seaborn as sb


# Get device
def get_device(cuda_preference=True):
    """Gets pytorch device object. If cuda_preference=True and 
        cuda is available on your system, returns a cuda device.
    """
    
    print('cuda available:', torch.cuda.is_available(), 
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count())
    
    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device



# Custom Data Set for CT Scans
class CustomData(Dataset):
  def __init__(self, X, Y, normalize = False, mean = 0, std = 1):

    #X = X.copy()
    self.X = X
    self.Y = Y
    self.normalize = normalize
    self.mean = mean
    self.std = std
  
  def __len__(self):
    return len(self.Y)

  def __getitem__(self, idx):
    if self.normalize == True:
      norm = transforms.Normalize(self.mean, self.std)
      x = norm(self.X[idx])
    elif self.normalize == False:
      x = self.X[idx]
    return x, self.Y[idx]




# function to compute mean and std of image data
def mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std

# custom weight initilization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


# Functions to train the network

def accuracy(correct, total): 
    """Compute accuracy as percentage."""
    return float(correct)/total



def train2(dataloader, optimizer, model, loss_fn, device, master_bar, threshold):
    """Run one training epoch.

    Arguments:
        dataloader (DataLoader): Torch DataLoader object to load data
        optimizer: Torch optimizer object
        model (nn.Module): Torch model to train
        loss_fn: Torch loss function
        device (torch.device): Torch device to use for training
        master_bar (fastprogress.master_bar): Will be iterated over for each
            epoch to draw batches and display training progress
        threshold: threshold value that indicates from where observations are predicted as 1

    """
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0

    for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
        
        x = x.to(device)
        y = y.type(torch.FloatTensor).to(device)
        
        optimizer.zero_grad()
        model.train()
        
        # Forward pass
        y_pred = model(x).squeeze()
        
        
        # For calculating the accuracy, save the number of correctly classified 
        # patients and the total number
        epoch_correct += sum(y == torch.where(y_pred>threshold,1,0))
        epoch_total += len(y)


        # Compute loss
        loss = loss_fn(y_pred.squeeze(),y)

        
        # Backward pass
        loss.backward()
        optimizer.step()


        # For plotting the train loss, save it for each sample
        epoch_loss.append(loss.item())

    # Return the mean loss and the accuracy of this epoch
    return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total)


def validate2(dataloader, model, loss_fn, device, master_bar, threshold):
    """Compute loss, accuracy and confusion matrix on validation set.

    Arguments:
        dataloader (DataLoader): Torch DataLoader object to load data
        model (nn.Module): Torch model to train
        loss_fn: Torch loss function
        device (torch.device): Torch device to use for training
        master_bar (fastprogress.master_bar): Will be iterated over to draw 
            batches and show validation progress
        threshold: threshold value that indicates from where observations are predicted as 1

    """
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    y_vec = np.array([])
    y_pred_vec = np.array([])
    confusion_matrix = torch.zeros(2, 2)    

    model.eval()
    with torch.no_grad():
        for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
            
            x = x.to(device)
            y = y.type(torch.FloatTensor).to(device)
            
            # make a prediction on validation set
            y_pred = model(x).squeeze()
            
            # For calculating the accuracy, save the number of correctly 
            # classified samples and the total number
            epoch_correct += sum(y == torch.where(y_pred>threshold,1,0))
            epoch_total += len(y)

            # safe y and y_pred for AUC computation
            y_vec = np.append(y_vec, y.detach().cpu().numpy())
            y_pred_vec =  np.append(y_pred_vec, torch.where(y_pred>threshold,1,0).detach().cpu().numpy())

            # Fill confusion matrix
            for (y_true, y_p) in zip(y, torch.where(y_pred>threshold,1,0)):
                confusion_matrix[int(y_true), int(y_p)] +=1

            # Compute loss
            loss = loss_fn(y_pred.squeeze(),y)


            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total), metrics.roc_auc_score(y_vec,y_pred_vec), confusion_matrix, y_vec,y_pred_vec


def run_training2(model, optimizer, loss_function, device, num_epochs, 
                train_dataloader, val_dataloader,lr, threshold = 0.5, early_stopper=True, verbose=False):
   
    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses, train_accs, val_accs, val_aucs = [],[],[],[],[]
    confusion_matrix_list, y_list, y_preds_list = [],[],[]

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_acc = train2(train_dataloader, optimizer, model, 
                                                  loss_function, device, master_bar, threshold = threshold)
        # Validate the model
        epoch_val_loss, epoch_val_acc, epoch_val_auc, confusion_matrix, y,y_pred = validate2(val_dataloader, 
                                                                   model, loss_function, 
                                                                   device, master_bar, threshold = threshold)
        
        #scheduler.step()
        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        val_aucs.append(epoch_val_auc)
        confusion_matrix_list.append(confusion_matrix)
        y_list.append(y)
        y_preds_list.append(y_pred)
        
        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}, train acc: {epoch_train_acc:.3f}, val acc {epoch_val_acc:.3f},val AUC {epoch_val_auc:.3f}')
            
        if early_stopper:
            ####################
           
           early_stopper.update(epoch_val_acc, model, optimizer)
           if early_stopper.early_stop:
             model,optimizer = early_stopper.load_checkpoint(model,lr)
             break


            ####################
        
   

            
    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')
    return train_losses, val_losses, train_accs, val_accs, val_aucs, epoch, confusion_matrix_list, y_list,y_preds_list


def plot(title, label, train_results, val_results, yscale='linear', save_path=None, 
         extra_pt=None, extra_pt_label=None):
    """Plot learning curves.

    Arguments:
        title (str): Title of plot
        label (str): x-axis label
        train_results (list): Results vector of training of length of number
            of epochs trained. Could be loss or accuracy.
        val_results (list): Results vector of validation of length of number
            of epochs. Could be loss or accuracy.
        yscale (str, optional): Matplotlib.pyplot.yscale parameter. 
            Defaults to 'linear'.
        save_path (str, optional): If passed, figure will be saved at this path.
            Defaults to None.
        extra_pt (tuple, optional): Tuple of length 2, defining x and y coordinate
            of where an additional black dot will be plotted. Defaults to None.
        extra_pt_label (str, optional): Legend label of extra point. Defaults to None.
    """
    
    epoch_array = np.arange(len(train_results)) + 1
    train_label, val_label = "Training "+label.lower(), "Validation "+label.lower()
    
    sb.set(style='ticks')

    plt.plot(epoch_array, train_results, epoch_array, val_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Validation results']
    
    if extra_pt:
        ####################
        plt.plot(extra_pt[0],extra_pt[1],marker = '*', color = 'k')
        plt.annotate(extra_pt_label,extra_pt)
        ####################
        
        
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)
    
    sb.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
    plt.show()  



def test(valdl, model, device,loss_fn, threshold = 0.5):
    """Compute accuracy and confusion matrix on validation set """
    ####################
    epoch_test_loss, epoch_test_acc, auc, confusion_matrix, y, y_pred = validate(valdl, 
                                                                   model, loss_fn, 
                                                                   device, master_bar = None, threshold = threshold)
    

    return epoch_test_loss, epoch_test_acc, auc, confusion_matrix, y, y_pred
    ####################



# Early Stopper
class EarlyStopper:
    """Early stops the training if validation accuracy does not increase after a
    given patience.
    """
    def __init__(self, verbose=False, path='checkpoint.pt', patience=3):
        """Initialization.

        Args:
            verbose (bool, optional): Print additional information. Defaults to False.
            path (str, optional): Path where checkpoints should be saved. 
                Defaults to 'checkpoint.pt'.
            patience (int, optional): Number of epochs to wait for increasing
                accuracy. If accyracy does not increase, stop training early. 
                Defaults to 1.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.__early_stop = False
        self.val_acc_max = -np.Inf
        self.path = path

        
    @property
    def early_stop(self):
        """True if early stopping criterion is reached.

        Returns:
            [bool]: True if early stopping criterion is reached.
        """
        ####################

        if self.patience == self.counter:
          return True
        else:
          return(False)

        ####################

        
        
    def update(self, val_acc, model, optimizer):
        """Call after one epoch of model training to update early stopper object.

        Args:
            val_acc (float): Accuracy on validation set
            model (nn.Module): torch model that is trained
        """
        ####################
        if val_acc > self.val_acc_max:
          self.save_checkpoint(model, optimizer, val_acc)
          self.counter = 0
        else:
          self.counter = self.counter + 1
        return
        ####################


            
    def save_checkpoint(self, model, optimizer, val_acc):
        """Save model checkpoint.

        Args:
            model (nn.Module): Model of which parameters should be saved.
        """
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        ####################

        self.val_acc_max = val_acc
        torch.save({
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                   }, self.path)
        return
        ####################

        
        
        
    def load_checkpoint(self, model, lr):
        """Load model from checkpoint.

        Args:
            model (nn.Module): Model that should be reset to parameters loaded
                from checkpoint.

        Returns:
            nn.Module: Model with parameters from checkpoint
        """
        if self.verbose:
            print(f'Loading model from last checkpoint with validation accuracy {self.val_acc_max:.6f}')
        ####################
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

        ####################


# Custom Data Set for Patient Data
class CustomPatientData(Dataset):
  def __init__(self, X, Y):

    #X = X.copy()
    self.X = X
    self.Y = Y
  
  def __len__(self):
    return len(self.Y)

  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]
    
    
    
# MLP model
class MLP(nn.Module):
    """Deep neural network torch model."""
    def __init__(self, num_features, dropout, num_hidden_units, num_hidden_layers):
        

        """Initialize model."""

        ####################
        super(MLP, self).__init__()
        
        self.relu = torch.nn.ReLU()
        self.linear_in = torch.nn.Linear(num_features, num_hidden_units)
        self.drop = nn.Dropout(dropout)
        self.hidden = torch.nn.ModuleList([torch.nn.Linear(num_hidden_units, num_hidden_units)  for _ in range(num_hidden_layers)])
        self.hidden_relu = torch.nn.ModuleList([torch.nn.ReLU()  for _ in range(num_hidden_layers)])  #when computing DeepLIFT attributions one cannot use the same ReLU module more than once
        self.linear_out = torch.nn.Linear(num_hidden_units, 1)

        # weight initialization
        self.hidden.apply(init_weights)
        torch.nn.init.xavier_normal_(self.linear_in.weight)
        torch.nn.init.xavier_normal_(self.linear_out.weight)
                 
        ####################

    
    def forward(self, x):
        """Compute model predictions."""

        ####################
       
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.drop(x)
        for i in range(len(self.hidden)):
          x = self.hidden[i](x)
          x = self.hidden_relu[i](x)
          x = self.drop(x)
        x = self.linear_out(x)
        output = torch.sigmoid(x)
        return output  
        ####################
        
        

def train(dataloader, optimizer, model, loss_fn, device, master_bar, threshold):
    """Run one training epoch.

    Arguments:
        dataloader (DataLoader): Torch DataLoader object to load data
        optimizer: Torch optimizer object
        model (nn.Module): Torch model to train
        loss_fn: Torch loss function
        device (torch.device): Torch device to use for training
        master_bar (fastprogress.master_bar): Will be iterated over for each
            epoch to draw batches and display training progress
        threshold: threshold value that indicates from where observations are predicted as 1

    """
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0

    for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
        
        x = x.to(device)
        y = y.type(torch.FloatTensor).to(device)
        
        optimizer.zero_grad()
        model.train()
        
        # Forward pass
        y_pred = model(x).squeeze()
        
        # For calculating the accuracy, save the number of correctly classified 
        # patients and the total number
        epoch_correct += sum(y == torch.where(y_pred>threshold,1,0))
        epoch_total += len(y)

        # Compute loss
        loss = loss_fn(y_pred.squeeze(),y)

        
        # Backward pass
        loss.backward()
        optimizer.step()


        # For plotting the train loss, save it for each sample
        epoch_loss.append(loss.item())

    # Return the mean loss and the accuracy of this epoch
    return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total)



def validate(dataloader, model, loss_fn, device, master_bar, threshold):
    """Compute loss, accuracy and confusion matrix on validation set.

    Arguments:
        dataloader (DataLoader): Torch DataLoader object to load data
        model (nn.Module): Torch model to train
        loss_fn: Torch loss function
        device (torch.device): Torch device to use for training
        master_bar (fastprogress.master_bar): Will be iterated over to draw 
            batches and show validation progress
        threshold: threshold value that indicates from where observations are predicted as 1

    """
    epoch_loss = []
    epoch_correct, epoch_total = 0, 0
    y_vec = np.array([])
    y_pred_vec = np.array([])
    confusion_matrix = torch.zeros(2, 2)    

    model.eval()
    with torch.no_grad():
        for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
            
            x = x.to(device)
            y = y.type(torch.FloatTensor).to(device)
            
            # make a prediction on validation set
            y_pred = model(x).squeeze()
            
            # For calculating the accuracy, save the number of correctly 
            # classified samples and the total number
            epoch_correct += sum(y == torch.where(y_pred>threshold,1,0))
            epoch_total += len(y)

            # safe y and y_pred for AUC computation
            y_vec = np.append(y_vec, y.detach().cpu().numpy())
            y_pred_vec =  np.append(y_pred_vec, torch.where(y_pred>threshold,1,0).detach().cpu().numpy())

            # Fill confusion matrix
            for (y_true, y_p) in zip(y, torch.where(y_pred>threshold,1,0)):
                confusion_matrix[int(y_true), int(y_p)] +=1

            # Compute loss
            loss = loss_fn(y_pred.squeeze(),y)


            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total), metrics.roc_auc_score(y_vec,y_pred_vec), confusion_matrix, y_vec,y_pred_vec


def run_training(model, optimizer, loss_function, device, num_epochs, 
                train_dataloader, val_dataloader,lr,scheduler, threshold = 0.5, early_stopper=True, verbose=False):
   
    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses, train_accs, val_accs, val_aucs = [],[],[],[],[]
    confusion_matrix_list, y_list, y_preds_list = [],[],[]

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_acc = train(train_dataloader, optimizer, model, 
                                                  loss_function, device, master_bar, threshold = threshold)
        # Validate the model
        epoch_val_loss, epoch_val_acc, epoch_val_auc, confusion_matrix, y,y_pred = validate(val_dataloader, 
                                                                   model, loss_function, 
                                                                   device, master_bar, threshold = threshold)
        
        scheduler.step()
        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        val_aucs.append(epoch_val_auc)
        confusion_matrix_list.append(confusion_matrix)
        y_list.append(y)
        y_preds_list.append(y_pred)
        

        
        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}, train acc: {epoch_train_acc:.3f}, val acc {epoch_val_acc:.3f},val AUC {epoch_val_auc:.3f}')
            
        if early_stopper:
            ####################
           
           early_stopper.update(epoch_val_acc, model, optimizer)
           if early_stopper.early_stop:
             model,optimizer = early_stopper.load_checkpoint(model,lr)
             break


            ####################
        
   

            
    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')
    return train_losses, val_losses, train_accs, val_accs, val_aucs, epoch, confusion_matrix_list, y_list,y_preds_list