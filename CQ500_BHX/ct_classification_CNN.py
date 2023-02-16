import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler, WeightedRandomSampler
from torchvision import transforms, datasets, models
from torchvision.models import alexnet, AlexNet_Weights
import fastprogress
from datetime import date


path0 = '/scratch1/users/mheiden/data/'
#path0 = '/Volumes/NONAME/Classification_rotated/data_test/'

# check if folder exists, if not create it
if not os.path.exists(path0 + "Results"):
    os.makedirs(path0 + "Results")


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



def accuracy(correct, total): 
    """Compute accuracy as percentage."""
    return float(correct)/total



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
    return np.mean(epoch_loss), accuracy(epoch_correct, epoch_total), confusion_matrix, y_vec,y_pred_vec


def run_training(model, optimizer, loss_function, device, num_epochs, 
                train_dataloader, val_dataloader,lr,scheduler, threshold = 0.5, early_stopper=True, verbose=False):
   
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses, train_accs, val_accs = [],[],[],[]
    confusion_matrix_list, y_list, y_preds_list = [],[],[]

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_acc = train(train_dataloader, optimizer, model, 
                                                  loss_function, device, master_bar, threshold = threshold)
        # Validate the model
        epoch_val_loss, epoch_val_acc, confusion_matrix, y,y_pred = validate(val_dataloader, 
                                                                   model, loss_function, 
                                                                   device, master_bar, threshold = threshold)
        
        scheduler.step()
        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        confusion_matrix_list.append(confusion_matrix)
        y_list.append(y)
        y_preds_list.append(y_pred)
        

        
        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}, train acc: {epoch_train_acc:.3f}, val acc {epoch_val_acc:.3f}')
            
        if early_stopper:
            ####################
           
           early_stopper.update(epoch_val_acc, model, optimizer)
           if early_stopper.early_stop:
             model,optimizer = early_stopper.load_checkpoint(model,lr)
             break


            ####################
        
   

    return train_losses, val_losses, train_accs, val_accs, epoch, confusion_matrix_list, y_list,y_preds_list




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




class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000, dropout = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),    
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

my_AlexNet = AlexNet()
orig_AlexNet = models.alexnet(weights = AlexNet_Weights.DEFAULT)
my_AlexNet.load_state_dict(orig_AlexNet.state_dict())

class extendedAlexNet(nn.Module):
    def __init__(self):
        super(extendedAlexNet,self).__init__()
        self.layer1 = nn.Linear(1000,100)
        self.layer2 = nn.Linear(100,1)
        self.net = my_AlexNet
        # net.features parameters are taken from AlexNet and will not be trained
        for p in self.net.features.parameters():
            p.requires_grad=False

    def forward(self,x):
        x1 = self.net(x)
        y = self.layer1(x1)
        y = self.layer2(y)
        return y


def save_model(model):
    day = date.today().strftime("%m%d")
    model_save_name = 'AlexNet_%s.pth' % day 
    path = path0 + F"Results/{model_save_name}"
    torch.save(model.state_dict(), path)



device = get_device()



# set options
training_samples = 300000

# batch size
batch_size = 128

# loss function
loss_fn = nn.MSELoss()

# patience parameter for early stopper
patience_param = 3

# learning rate
lr = 0.001

# number of epochs
n_epochs = 10


#elastic_transformer = transforms.ElasticTransform(alpha=100.0)
#transforms_elastic = transforms.RandomApply(
#    torch.nn.ModuleList([elastic_transformer]), p=0.3)

transforms_ccrop = transforms.RandomApply(
    torch.nn.ModuleList([
    transforms.CenterCrop((300,200)),
    transforms.Resize((512,512))]), p=0.3)


blurrer = transforms.RandomApply(
    torch.nn.ModuleList([
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 5))])
    ,p=0.3)



transform = transforms.Compose([
    transforms_ccrop,
    blurrer,
    transforms.ToTensor()    
])




transform_val = transforms.Compose([
    transforms.ToTensor(),

])



####### Loading the data ########

# Load full training data set
full_trainset = datasets.ImageFolder(path0 + 'Train/', transform=transform)

# Load full validation data set
full_valset = datasets.ImageFolder(path0 + 'Val/', transform=transform_val)


# Remove duplets in data that somehow appear when uploading to google drive
doppelte = []
doppelte_idx = []
i = 0
for t in full_trainset.imgs:
  if ' ' in t[0]:
    doppelte.append(t)
    doppelte_idx.append(i)
  i=i+1

for d in doppelte:
  full_trainset.imgs.remove((d[0], d[1]))

full_trainset.targets = np.delete(full_trainset.targets, doppelte_idx, axis=0)

doppelte = []
for t in full_valset.imgs:
  if ' ' in t[0]:
    doppelte.append(t)

for d in doppelte:
  full_valset.imgs.remove((d[0], d[1]))

#print('##### Data Loaded #######')
#print('Length of Train data:',len(full_trainset))
#print('Length of validation data:',len(full_valset))



train_accs, train_losses, val_accs, val_losses, epochs_stopped = [], [], [], [], []

RANDOM_SEED = 42
RANDOM = np.random.RandomState(RANDOM_SEED)

  
# Oversampling of images with hemorrages to obtain a more balanced dataset 
train_class_sample_count = np.array([len(np.where(full_trainset.targets == t)[0]) for t in np.unique(full_trainset.targets)])
train_weight0 = 1. / train_class_sample_count

train_weight1 = np.array([train_weight0[t] for t in full_trainset.targets])

train_samples_weights = torch.from_numpy(train_weight1)
  
# Sample only a certain number of samples from training data
train_sampler = WeightedRandomSampler(train_samples_weights.type('torch.DoubleTensor'), training_samples, generator=torch.Generator().manual_seed(RANDOM_SEED))
  

trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=batch_size , sampler = train_sampler, drop_last = True)
valloader = torch.utils.data.DataLoader(full_valset,batch_size=batch_size, shuffle = True,drop_last = True)

# instantiate model and optimizer
model = extendedAlexNet().to(device)

# Optiizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr = lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# instantiate early stopper
stopper = EarlyStopper(patience=patience_param)

# run training
train_loss, val_loss, train_acc, val_acc, epoch_stopped, _, _, _ = run_training(model, optimizer, loss_fn, device, n_epochs, trainloader, valloader,lr, exp_lr_scheduler, early_stopper=stopper, verbose = True)
  
save_model(model)

metrics_df = pd.DataFrame({'Train_ACC':train_acc,'Train_LOSS':train_loss, 'VAL_ACC':val_acc, 'VAL_LOSS':val_loss, 'EPOCH':epoch_stopped})
day = date.today().strftime("%m%d")
metrics_df_save_name = 'metrics_results_%s.csv' % day
p = path0 + 'Results/%s' %metrics_df_save_name
metrics_df.to_csv(p)
