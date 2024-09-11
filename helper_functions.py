# Imports:

# allows you to handle csv files
import pandas as pd

# cv2 to read images
import cv2
import os
import re
import matplotlib.pyplot as plt

# # multiprocessing: cpu tasks
# from multiprocessing import Pool

# # multithreading: input/output
# from concurrent.futures import ThreadPoolExecutor

# plot
import matplotlib.pyplot as plt
import seaborn as sns

# numpy
import numpy as np

# sklearn confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# from sklearn.model_selection import train_test_split

# PIL (Pillow: Python Imaging Library used by Pytorch)
import PIL

# # import timm
# import timm

# import pytorch
import torch

# import torchvision for image transformation
# import torchvision
# from torchvision.models import (
#     resnet50, ResNet50_Weights, 
#     vgg16, VGG16_Weights, 
#     mobilenet_v3_large, MobileNet_V3_Large_Weights, 
#     efficientnet_v2_l, EfficientNet_V2_L_Weights
#     )
import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data.sampler import SubsetRandomSampler
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# import Pytorch's neural network and optimization library
# import torch.nn as nn 
# import torch.nn.functional as F
# import torch.optim as optim

# dataloader
from torch.utils.data import DataLoader, Dataset

# # math
# import math

# # import streamlit
# import streamlit as st


# # GPU available
# print(f'GPU available: {torch.cuda.is_available()}')

# use GPU is cuda is available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


################ Generate DataFrame of Samples ###############

# create df and append important details in new columns
def generate_df(path):
    
    drop = []
    card_id = []
    title = []
    file_names = []
    
    card = ['Card', 'Cards', 'CARD', 'CARDS']
    
    for file in os.listdir(path):

        # only valid jpg file names will be saved. Filters out corrupted save files
        if file.endswith('.jpg'):
            file_name = re.split('_|\.', file)

            # appends file name of img that contains cards
            if any (x in file_name for x in card):
                file_names.append(file_name)
                card_id.append(file_name[1])
                title.append(file)

            # appends card id for those img that are non-cards
            # to track how many img out of the original data has been dropped
            else:
                drop.append(file_name[1])
    
    df = pd.DataFrame(file_names)


    # setting various variables to filter for
    grade_number = ['1','2','3','4','5','6','7','8','9','10']
    grade_type = ['MT', 'MINT+', 'MINT', 'NM-MT+', 'NM-MT', 'NM+', 'NM', 'EX-MT+', 'EX-MT', 'EX+', 'EX', 'VG-EX+', 'VG-EX', 'VG+', 'VG', 'GOOD+', 'GOOD', 'FR', 'PR']
    side_text = ['Front', 'Back']
    card = ['Card', 'Cards', 'CARD', 'CARDS']
    card_type = ['TCG', 'FOOTBALL', 'BASEBALL', 'WRESTLING', 'SOCCER', 'BASKETBALL', 'HOCKEY', 'NON-SPORT', 'MULTI-SPORT' ,'LEAGUE' ,'MISC', 'GOLF']

    grade_num_list =[]
    side_list =[]
    card_list = {}

    END_COLUMN = len(df.columns)

    for index, row in df.iterrows():
        for x in range(0,END_COLUMN):
            if row[x] in card_type and df.iloc[index, 1] not in card_list:
                if row[x+1] in card:
                    card_list[df.iloc[index, 1]] = row[x]
            if row[x] in side_text:
                side_list.append(row[x])


    print(f'Samples to drop: {set(drop)}') # print unique values    
    print(f'Samples dropped: {len(drop)/2}')


    for index, row in df.iterrows():
    # check if row has been appended. 0 for no, 1 for yes
        row_checked = 0

        # looping through each column in a row, given the maximum length of columns
        for x in range(0,END_COLUMN):
            # if row has not been checked, proceed on
            if row_checked == 0:
                # check if the grade number is followed by another number, in the case of half point grade, and check if the preceding data is a grade type
                # append grade, and indicate row has been checked.
                if row[x] in grade_number and row[x+1] in grade_number and row[x-1] in grade_type:
                    grade_num_list.append(row[x]+"."+row[x+1])
                    row_checked += 1
                # if there is no number after grade, append grade, and indicate that the row has been checked.
                elif row[x] in grade_number and row[x-1] in grade_type:
                    grade_num_list.append(row[x])
                    row_checked += 1
            else:
                pass

    df['File_name'] = title
    df['Company'] = df[0]
    df['Id'] = df[1]
    df['Product'] = df[8]
    df['Grade'] = grade_num_list
    df['Side'] = side_list
    df['Type'] = df.iloc[:, 1].map(lambda x: card_list[x])

    return df



############################## Crop Images ##############################

# standardise img size and crop img
def crop_img(from_path, to_path):

    exception_list = []

    card = ['Card', 'Cards', 'CARD', 'CARDS']
    
    for file in os.listdir(from_path):
        if file.endswith('.jpg'):

            # restrict file len of file name to be less than 181 else open(path, "rb") will error out during dataiter:
            if len(file) < 180:

                file_name = re.split('_|\.', file)

                if any(x in file_name for x in card): # checking if any item in card is in file_name. Use to skip any non card items
                    try: 
                        image = cv2.imread(f'{from_path}\{file}')

                        # resize all images to width: 1400, height: 2400 for crop
                        image = cv2.resize(image, (1400, 2400))
                        
                        # set the area variables (x,y and width, height) to crop the image (e.g rectangle)
                        # resize img with torchvision.transforms.Resize()
                        x = 125 # original
                        y = 610 # original
                        w = 1160 # original
                        h = 1625 # original

                        cropped_image = image[y:y+h, x:x+w]

                        # save cropped image to designated folder
                        cv2.imwrite(f'{to_path}/{file}', cropped_image)

                    except Exception:
                        exception_list.append(file)
                        pass
                else:
                    exception_list.append(file) # append the non card items
                    pass
            else:
                exception_list.append(file) # append card with len(file) > 180
                pass

    df = pd.DataFrame(exception_list)
        
    return df


############################## Concat Images ##############################

# concat imgs with same id
def concat_img(from_path, to_path):
    
    exception_list = []

    concat_list = []

    check_list = []

    test = []

    for file in os.listdir(from_path):
        try:
            concat_list.append(file)

            id = int(re.split('_|\.', file)[1])

            check_list.append(id)

            # check if id are the same, and concat the imgs
            if len(concat_list) == 2 and check_list[0] == check_list[1]:
                img_1 = cv2.imread(f'{from_path}\{concat_list[0]}')
                img_2 = cv2.imread(f'{from_path}\{concat_list[1]}')

                try:
                    h_img = cv2.hconcat([img_1, img_2])
                    #  save concat image to designated folder
                    cv2.imwrite(f'{to_path}/{file}', h_img)
                    
                    concat_list =[]
                    check_list =[]

                except cv2.error as error:
                    exception_list.append(file)
                    pass

            # if there are 2 imgs but ids are not the same, replace the first img with second img and remove the second img. 
            elif len(concat_list) == 2 and check_list[0] != check_list[1]:

                concat_list[0] = concat_list[1]
                check_list[0] = check_list[1]
                concat_list.pop()
                check_list.pop()

        except Exception:
            exception_list.append(file)
            pass
    
    df = pd.DataFrame(exception_list)
    
    return df


############################## Define Seed ##############################

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


################ Transform Image to tensor and attribute targets ###############

# merge features and image
class CardData(Dataset):
    """ Dataset connecting card images to the score and annotations

        Args:
            img_dir (string): Path to image directory
            csv_dir (string): Path to csv directory
            transform (optional): Transformation to be applied to object
    """
    def __init__ (self, img_dir, df, transform = transforms.ToTensor()):

        # set df
        self.df = df

        # img dir
        self.img_dir = img_dir
        
        # apply transformations
        self.transform = transform

    # return length of dataset
    def __len__(self):
        return len(self.df)

    # modify load and preprocess images with __getitem__ to avoid the need to load the entire dataset in to memory upfront
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # this code loads each image retrieved from disk
        # join img and annotations
        img_name =  os.path.join(self.img_dir, self.df.iloc[idx, 0])

        # load image in PIL format required for compatability
        img = PIL.Image.open(img_name)

        # grade defined in df column 1. Set data as float
        grade = np.array(self.df.iloc[idx, 1])
        # .view(1) to set data in correct shape for MSE loss function
        grade = torch.tensor(grade).view(1).to(torch.float32)

        # side defined in df column 2. Set data as int
        side = np.array(self.df.iloc[idx, 2])
        # set data as tensor. for classification
        side = torch.tensor(side).to(torch.int64)

        # side defined in df column 3. Set data as int
        type = np.array(self.df.iloc[idx, 3])
         # set data as tensor. for classification
        type = torch.tensor(type).to(torch.int64)

        # grade class define in df column 4. Set data as int
        grade_class = np.array(self.df.iloc[idx, 4])
         # set data as tensor. for classification
        grade_class = torch.tensor(grade_class).to(torch.int64)
    

        # apply transformation
        img = self.transform(img)

        # img and targets
        sample = [img, grade, side, type, grade_class]

        return sample    



############################## Dataloader Function ##############################

# train val test split data
def loader(train_data, val_data, test_data, sampler=None):

    """ Split data set in to train and test, and perform data augmentation accordingly to train and test data
    
        Args:
            train_val_img_dir (string): Path to train and validation image directory
            train_val_csv_dir (string): Path to train and validation csv directory
            test_img_dir (string): Path to test image directory
            test_csv_dir (string): Path to test csv directory
            val_size (float): proportion of sample for validation. Default: 0.2
    """

    # 32 rule of thumb. 64 works as well. Use powers of 2
    batch_size = 128

    train_loader = DataLoader(
                            train_data, # training transformation w data augmentation
                            # sampler = sampler
                            shuffle = True,  # Shuffle data
                            batch_size = batch_size, # split sampler in to # batches according to batch_size
                            num_workers = 0, # specify how many CPU cores to utilise for multi process loading. Set 0 for main core
                            pin_memory = True # speed up host (CPU) to device (GPU) transfer for large dataset.
    )


    val_loader = DataLoader(
                            val_data, # validation transformation w/o data augmentation
                            batch_size = batch_size, # split sampler in to # batches according to batch_size
                            num_workers = 0, # specify how many CPU cores to utilise for multi process loading. Set 0 for main core
                            pin_memory = True # speed up host (CPU) to device (GPU) transfer for large dataset.
    )


    test_loader = DataLoader(
                            test_data, # validation transformation w/o data augmentation
                            batch_size = batch_size, # split sampler in to # batches according to batch_size
                            num_workers = 0, # specify how many CPU cores to utilise for multi process loading. Set 0 for main core
                            pin_memory = True # speed up host (CPU) to device (GPU) transfer for large dataset.
    )

    # print(f'Total samples: {len(train_data)}')
    print(f'Train samples: {len(train_data)}')
    print(f'Val samples: {len(val_data)}')
    print(f'Test samples: {len(test_data)}')

    return train_loader, val_loader, test_loader


############### Visualise data ###############
# plot RatingNum Frequency (Regression workflow)

# n, bins, patches = plt.hist(train_df.iloc[:, 1], 10, density=True, facecolor='g', alpha=0.75)

# plt.xlabel('Card Rating')
# plt.ylabel('Frequency')
# plt.title('Rating Histogram')
# plt.xlim(0, 10)
# # plt.ylim(0, 0.03)
# plt.grid(True)
# plt.show()


############################## Classification side train function ##############################

def classification_model_train(target, save_model, model, train_loader, val_loader, optimiser, scheduler, loss_criterion, epochs=10, patience=7):

    """ Function to train model """

    """
    Args:
        objective (string) : objective for running the train model
        model : model used for training
        loader : train loader
        optimiser : choice of optimiser
        loss_criterion: choice of loss measure
        epochs : Default 10
    """
    # set model to run on GPU
    model.to(device)

    # model name
    model_name = model.__class__.__name__

    # create empty array to store values for plotting
    epoch_log = []
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_accuracy = []
    
    # set early_stopping
    counter = 0
    best_score = None

    # iterate over epochs:
    for epoch in range(epochs):
        print(f'Starting Epoch: {epoch+1}')

        # save train loss per batch
        batch_train_loss = []

        # train mode
        model.train()

        # # accumulate loss after each mini batch
        # train_loss = 0.0

        # iterate over trainloader iterator in mini-batches
        for i, data in enumerate(train_loader,0):

            # learning rate scheduler
            iter = len(train_loader)

            # data is a list [inputs, grade, sides, types]
            inputs, grades, labels, types, grades_class = data

            # move data to GPU
            inputs = inputs.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            
            # clear gradients before each mini-batch training
            optimiser.zero_grad()

            # forward propagate
            outputs = model(inputs)

            if target.lower() == 'side':
                # loss = difference between prediction and labels
                loss = loss_criterion(outputs, labels)

                # back propagate to obtain gradients for all nodes
                loss.backward()

                # update gradients/ weights
                optimiser.step()

                # update scheduler
                scheduler.step(epoch + i/iter)

                # update running_loss
                batch_train_loss.append(loss.item())

            elif target.lower() == 'type':
                # loss = difference between prediction and labels
                loss = loss_criterion(outputs, types)

                # back propagate to obtain gradients for all nodes
                loss.backward()

                # update gradients/ weights
                optimiser.step()

                # update scheduler
                scheduler.step(epoch + i/iter)

                # update running_loss
                batch_train_loss.append(loss.item())

            elif target.lower() == 'gradeclass':
                # loss = difference between prediction and labels
                loss = loss_criterion(outputs, grades_class)

                # back propagate to obtain gradients for all nodes
                loss.backward()

                # update gradients/ weights
                optimiser.step()

                # update scheduler
                scheduler.step(epoch + i/iter)

                # update running_loss
                batch_train_loss.append(loss.item())

        # save epoch train loss
        epoch_train_loss.append(np.array(batch_train_loss).mean())

        # evaluation loop after each epoch
        model.eval()

        # store batch val loss to get epoch loss
        batch_val_loss = []

        # store val loss to save model
        val_loss = 0.0

        # for prediction accuracy
        correct = 0
        total = 0

        for data in val_loader:

            # save train loss per batch
            images, grades, labels, types, grades_class = data

            # move data to GPU
            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            # forward propagate test data through model
            outputs = model(images)

            if target.lower() == 'side':
                loss = loss_criterion(outputs, labels)

                batch_val_loss.append(loss.item())

                val_loss += loss.item()

                # predicted class
                _, predicted = torch.max(outputs.data, dim=1)

                # returns number of labels per batch
                total += labels.size(0)

                # keep running total of # of predictions predicted correctly
                correct += (predicted == labels).sum().item()

            elif target.lower() == 'type':
                loss = loss_criterion(outputs, types)

                batch_val_loss.append(loss.item())

                val_loss += loss.item()

                # predicted class
                _, predicted = torch.max(outputs.data, dim=1)

                # returns number of labels per batch
                total += types.size(0)

                # keep running total of # of predictions predicted correctly
                correct += (predicted == types).sum().item()

            elif target.lower() == 'gradeclass':
                loss = loss_criterion(outputs, grades_class)

                batch_val_loss.append(loss.item())

                val_loss += loss.item()

                # predicted class
                _, predicted = torch.max(outputs.data, dim=1)

                # returns number of labels per batch
                total += grades.size(0)

                # keep running total of # of predictions predicted correctly
                correct += (predicted == grades_class).sum().item()

        # store epoch
        epoch_log.append(epoch+1)
        
        epoch_val_loss.append(np.array(batch_val_loss).mean())

        # mean val loss
        val_loss /= len(val_loader)

        # accuracy in %
        accuracy = correct / total * 100

        # store accuracy for plot
        epoch_val_accuracy.append(np.array(accuracy))

        if best_score is None:
            best_score = val_loss
            # save the first run
            PATH = f'./model/{model_name}_{target}_epoch_{epoch+1}_val_loss_{best_score:.4f}_val_acc_{accuracy:.4f}.pth'
            # save using model.module.state.dict() if model might be using DataParallel
            torch.save(model.state_dict(), PATH)
            print("Model Saved")
        elif val_loss < best_score:
            # Check if val_loss improves or not. if val_loss improves, we update the latest best_score, and save the current model weights
            best_score = val_loss

            # reset counter for early stoping
            counter = 0
            
            if save_model.lower() == 'yes':
                PATH = f'./model/{model_name}_{target}_epoch_{epoch+1}_val_loss_{best_score:.4f}_val_acc_{accuracy:.4f}.pth'
                # save using model.module.state.dict() if model might be using DataParallel
                torch.save(model.state_dict(), PATH)
                print("Model Saved")
            else:
                pass
        
        else:
            # val_loss does not improve, we increase the counter, 
            # stop training if it exceeds the amount of patience
            counter += 1
            if counter == patience:
                break

        print(f'Epoch: {epoch+1}, Current loss: {val_loss:.4f}, Current acc:{accuracy:.4f}')    
        print(f'Best loss: {best_score:.4f}')

    print("Training Completed")


    # plot loss and accuracy
    # create subplots
    fig, ax1 = plt.subplots()

    # create twinx to plot secondary y-axis
    ax2 = ax1.twinx()

    # create plot for loss_log and accuracy_log wrt to epoch_log
    ax1.plot(epoch_log, epoch_val_loss, 'r-')
    ax1.plot(epoch_log, epoch_train_loss, 'b-')
    ax2.plot(epoch_log, epoch_val_accuracy, 'g-')

    # set title and labels
    plt.title('Accuracy and Losses vs Epoch')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Val Loss', color = 'r')
    ax1.set_ylabel('Train Loss', color = 'b')
    ax2.set_ylabel('Val Accuracy', color = 'g')

    return [model_name, epoch_log, epoch_train_loss, epoch_val_loss]


################################## Regression train function ##################################

def regression_model_train(target, save_model, model, train_loader, val_loader, optimiser, scheduler, loss_criterion, epochs=10, patience=7):

    """ Function to train model """

    """
    Args:
        purpose (string) : objective for running the train model
        model : model used for training
        mode (string) : 'save' = save model
        loader : train loader
        optimiser : choice of optimiser
        loss_criterion: choice of loss measure
        epochs : Default 10
    """
    # set model to run on GPU
    model.to(device)

    # model name
    model_name = model.__class__.__name__

    # create empty array to store values for plotting
    epoch_log = []
    epoch_train_loss = []
    epoch_val_loss = []

    # set early_stopping
    counter = 0
    best_score = None
    # PATH = # user defined path to save model

    # iterate over epochs:
    for epoch in range(epochs):
        print(f'Starting Epoch: {epoch+1}')

        # train mode
        model.train()

        # accumulate loss after each mini batch
        batch_train_loss = []
        

        # iterate over trainloader iterator in mini-batches
        for i, data in enumerate(train_loader,0):

            # learning rate scheduler
            iter = len(train_loader)

            # data is a list [inputs, grades, labels, types, grades_class]
            inputs, grades, labels, types, grades_class = data

            # move data to GPU
            inputs = inputs.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)
            
            # clear gradients before each mini-batch training
            optimiser.zero_grad()

            # forward propagate
            # multiply by 10 the sigmoid output 0 - 10
            outputs = model(inputs) * 10

            # loss = difference between prediction and labels. RMSE
            loss = torch.sqrt(loss_criterion(outputs, grades))

            # back propagate to obtain gradients for all nodes
            loss.backward()

            # update gradients/ weights
            optimiser.step()

            # update scheduler
            scheduler.step(epoch + i/iter)

            # update running_loss
            batch_train_loss.append(loss.item())

        # save epoch train loss
        epoch_train_loss.append(np.array(batch_train_loss).mean())

        # evaluation loop after each epoch
        model.eval()

        # store batch val loss to get epoch loss
        batch_val_loss = []

        # for epoch validation loss
        val_loss = 0.0
        
        for data in val_loader:
            images, grades, labels, types, grades_class = data

            # move data to GPU
            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            # forward propagate test data through model
            outputs = model(images) * 10

            loss = torch.sqrt(loss_criterion(outputs, grades))

            batch_val_loss.append(loss.item())

            val_loss += loss.item()
        
        epoch_log.append(epoch+1)

        epoch_val_loss.append(np.array(batch_val_loss).mean())

        val_loss /= len(val_loader)
        
        if best_score is None:
            best_score = val_loss
            PATH = f'./model/{model_name}_{target}_epoch_{epoch}_val_loss_{best_score:.4f}.pth'
            # save using model.module.state.dict() if model might be using DataParallel
            torch.save(model.state_dict(), PATH)
            print("Model Saved")
        elif val_loss < best_score:
            # Check if val_loss improves or not.
            # val_loss improves, we update the latest best_score, 
            # and save the current model
            best_score = val_loss

            # reset counter for early stoping
            counter = 0

            if save_model.lower() == 'yes':
                PATH = f'./model/{model_name}_{target}_epoch_{epoch}_val_loss_{best_score:.4f}.pth'
                # save using model.module.state.dict() if model might be using DataParallel
                torch.save(model.state_dict(), PATH)
                print("Model Saved")
            else:
                pass

        else:
            # val_loss does not improve, we increase the counter, 
            # stop training if it exceeds the amount of patience
            counter += 1
            if counter == patience:
                break
            
        print(f'Epoch: {epoch+1}, Current RMSE: {val_loss:.4f}')
        print(f'Best RMSE: {best_score:.4f}')

    print("Training Completed")

    # plot loss and accuracy
    # create subplots
    fig, ax1 = plt.subplots()

    # create twinx to plot secondary y-axis
    ax2 = ax1.twinx()

    # create plot for loss_log and accuracy_log wrt to epoch_log
    ax1.plot(epoch_log, epoch_train_loss, 'b-')
    ax2.plot(epoch_log, epoch_val_loss, 'r-')

    # set title and labels
    plt.title('Train Loss and Val Loss vs Epoch')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color = 'b')
    ax2.set_ylabel('Val Loss', color = 'r')

    return [model_name, epoch_log, epoch_train_loss, epoch_val_loss]


################################# Data Accuracy #################################

# define function for val accuracy
def data_acc(target, model, loader):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, grades, labels, types, grades_class = data

            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            # propagate data through the CNN
            outputs = model(images) # call on the neural network model that is loaded prior
            _, predicted = torch.max(outputs.data, 1)
            
            if target.lower() == 'side':
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            elif target.lower() == 'type':
                total += types.size(0)
                correct += (predicted == types).sum().item()
            elif target.lower() == 'gradeclass':
                total += types.size(0)
                correct += (predicted == grades_class).sum().item()  
            
    accuracy = correct / total * 100
    print(f'Model accuracy: {accuracy:.2f}%')


################################# Classification Report and Confusion Matrix #################################

# define function for classification report and confusion matrix

# predictions are in torch tensor, need to convert to numpy to use sklearn confusion matrix

# initialise blank tensor to store predictions and labels
# torch.zeros creates a list of 0, specific by n number of elements, with dtype = .long (same as .int64)
# print(torch.zeros(0, dtype = torch.long)) = tensor([], dtype=torch.int64)
# print(torch.zeros(5, dtype = torch.long)) = tensor([0,0,0,0,0], dtype=torch.int64)

def eval_report(target, model,loader):
    model.to(device)
    model.eval()

    pred_list = torch.zeros(0, dtype = torch.long, device = 'cpu')
    label_list = torch.zeros(0, dtype = torch.long, device = 'cpu')
    type_list = torch.zeros(0, dtype = torch.long, device = 'cpu')
    grade_list = torch.zeros(0, dtype = torch.long, device = 'cpu')

    with torch.no_grad():
        for i, (images, grades, labels, types, grades_class) in enumerate(loader):

            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)
            
            if target.lower() == 'side':
                # enumerate through the validation set and concatenate the predictions and labels tensor to their respective lists
                # view(-1) is to reshape/ flatten tensor. e,g tensor size (3,4) is flatten to size (12, )
                pred_list = torch.cat([pred_list, predictions.view(-1).cpu()])
                label_list = torch.cat([label_list, labels.view(-1).cpu()])
            elif target.lower() == 'type':
                # enumerate through the validation set and concatenate the predictions and labels tensor to their respective lists
                # view(-1) is to reshape/ flatten tensor. e,g tensor size (3,4) is flatten to size (12, )
                pred_list = torch.cat([pred_list, predictions.view(-1).cpu()])
                type_list = torch.cat([type_list, types.view(-1).cpu()])
            elif target.lower() == 'gradeclass':
                # enumerate through the validation set and concatenate the predictions and labels tensor to their respective lists
                # view(-1) is to reshape/ flatten tensor. e,g tensor size (3,4) is flatten to size (12, )
                pred_list = torch.cat([pred_list, predictions.view(-1).cpu()])
                grade_list = torch.cat([grade_list, grades_class.view(-1).cpu()])

    if target.lower() == 'side':
        # print out classification report
        print(classification_report(y_true = label_list.numpy() , y_pred = pred_list.numpy()))

        # sklearn confusion matrix (actual vs predicted)
        con_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())

        # print(con_mat)
        plt.figure(figsize=(5, 4))
        sns.heatmap(con_mat, annot=True, cmap='Blues', fmt='g',
                    xticklabels=['Predicted Back', 'Predicted Front'],
                    yticklabels=['Actual Back', 'Actual Front'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # plot ROC curve
        fpr, tpr, thresholds = roc_curve(label_list.numpy(), pred_list.numpy())

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        # auc value
        auc_value = auc(fpr, tpr)
        print(f'AUC: {auc_value:.4f}')

    elif target.lower() == 'type':
        # print out classification report
        print(classification_report(y_true = type_list.numpy() , y_pred = pred_list.numpy()))

        # sklearn confusion matrix (actual vs predicted)
        con_mat = confusion_matrix(type_list.numpy(), pred_list.numpy())

        # print(con_mat)
        plt.figure(figsize=(5, 4))
        sns.heatmap(con_mat, annot=True, cmap='Blues', fmt='g',
                    # xticklabels=['Predicted Back', 'Predicted Front'],
                    # yticklabels=['Actual Back', 'Actual Front']
                    )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # plot One vs Rest ROC curve
        num_classes = np.unique(type_list.numpy())

        type_num = {
            0 : 'BASEBALL',
            1 : 'BASKETBALL',
            2 : 'FOOTBALL',
            3 : 'HOCKEY',
            4 : 'LEAGUE',
            5 : 'MISC',
            6 : 'MULTI-SPORT',
            7 : 'NON-SPORT',
            8 : 'SOCCER',
            9 : 'TCG',
            10 : 'WRESTLING',
            11 : 'GOLF',
        }

        for i in num_classes:
            # Create binary classification problem for class i
            y_true_binary = (type_list.numpy() == i).astype(int)
            y_pred_binary = (pred_list.numpy() == i).astype(int)
            # y_pred_binary = pred_list.numpy()[:, i]

            fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
            
            # Plot ROC curve for class i
            plt.plot(fpr, tpr, label=f'Class {i} ({type_num[i]})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('One-vs-Rest ROC Curve')
            plt.legend(bbox_to_anchor=(1.48,0), loc='lower right')

            # auc value
            auc_value = auc(fpr, tpr)
            print(f'AUC: {auc_value:.4f}')

        plt.show()

    elif target.lower() == 'grade':
        # print out classification report
        print(classification_report(y_true = grade_list.numpy() , y_pred = pred_list.numpy()))

        # sklearn confusion matrix (actual vs predicted)
        con_mat = confusion_matrix(grade_list.numpy(), pred_list.numpy())

        # print(con_mat)
        plt.figure(figsize=(5, 4))
        sns.heatmap(con_mat, annot=True, cmap='Blues', fmt='g',
                    # xticklabels=['Predicted Back', 'Predicted Front'],
                    # yticklabels=['Actual Back', 'Actual Front']
                    )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # plot One vs Rest ROC curve
        num_classes = np.unique(grade_list.numpy())

        for i in num_classes:
            # Create binary classification problem for class i
            y_true_binary = (grade_list.numpy() == i).astype(int)
            y_pred_binary = (pred_list.numpy() == i).astype(int)
            # y_pred_binary = pred_list.numpy()[:, i]

            fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)
            
            # Plot ROC curve for class i
            plt.plot(fpr, tpr, label=f'Class {i}')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('One-vs-Rest ROC Curve')
            plt.legend()
        plt.show()



################################# Prit Misclassified Images #################################

# function to print out misclassified images (predicted != actual)

def print_misclass(target, model, loader, num=5):
    model.to(device)
    model.eval()

    count = 0

    # use torch.no_grad() to save memory
    with torch.no_grad():
        for data in loader:
            images, grades, labels, types, grades_class = data

            # move data to GPU
            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            # output from network
            outputs = model(images)

            # torch.max() returns the max value, torch.argmax() returns the index of the max value
            # dim = 1 => columns, dim = 0 => rows
            predictions = torch.argmax(outputs, dim=1)

            for i in range(data[0].shape[0]):
                
                while count < num:

                    if target.lower() == 'side':
                        pred = predictions[i].item()
                        label = labels[i]

                        if(label !=  pred):
                            
                            print(f'Actual Label: {label}, Predicted label: {pred}')

                            mean = [0.485, 0.456, 0.406]
                            std = [0.229, 0.224, 0.225]
                            
                            # revert img back before normalisation
                            unnormalized_image = transforms.Normalize(mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])(images[i])

                            # to cpu for converting back to numpy array
                            unnormalized_image = unnormalized_image.to("cpu").numpy()

                            # tensor shape (channel, H, W) transpose to imshow shape (H, W, channel)
                            # 0: channels (color channels, typically 3 for RGB images)
                            # 1: height (vertical axis)
                            # 2: width (horizontal axis)
                            unnormalized_image = unnormalized_image.transpose((1, 2, 0))

                            # Clip pixel values to [0, 1] range
                            unnormalized_image = np.clip(unnormalized_image, 0, 1)

                            count += 1

                            plt.imshow(unnormalized_image)

                            plt.show()

                    elif target.lower() == 'type':
                        pred = predictions[i].item()
                        type = types[i]

                        if(type !=  pred):
                            
                            print(f'Actual Label: {type}, Predicted label: {pred}')

                            mean = [0.485, 0.456, 0.406]
                            std = [0.229, 0.224, 0.225]
                            
                            # revert img back before normalisation
                            unnormalized_image = transforms.Normalize(mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])(images[i])

                            # to cpu for converting back to numpy array
                            unnormalized_image = unnormalized_image.to("cpu").numpy()

                            # tensor shape (channel, H, W) transpose to imshow shape (H, W, channel)
                            # 0: channels (color channels, typically 3 for RGB images)
                            # 1: height (vertical axis)
                            # 2: width (horizontal axis)
                            unnormalized_image = unnormalized_image.transpose((1, 2, 0))

                            # Clip pixel values to [0, 1] range
                            unnormalized_image = np.clip(unnormalized_image, 0, 1)

                            count += 1

                            plt.imshow(unnormalized_image)

                            plt.show()

                    elif target.lower() == 'grade':
                        pred = predictions[i].item()
                        grade = grades_class[i]

                        if(grade !=  pred):
                            
                            print(f'Actual Label: {grade}, Predicted label: {pred}')

                            mean = [0.485, 0.456, 0.406]
                            std = [0.229, 0.224, 0.225]
                            
                            # revert img back before normalisation
                            unnormalized_image = transforms.Normalize(mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])(images[i])

                            # to cpu for converting back to numpy array
                            unnormalized_image = unnormalized_image.to("cpu").numpy()

                            # tensor shape (channel, H, W) transpose to imshow shape (H, W, channel)
                            # 0: channels (color channels, typically 3 for RGB images)
                            # 1: height (vertical axis)
                            # 2: width (horizontal axis)
                            unnormalized_image = unnormalized_image.transpose((1, 2, 0))

                            # Clip pixel values to [0, 1] range
                            unnormalized_image = np.clip(unnormalized_image, 0, 1)

                            count += 1

                            plt.imshow(unnormalized_image)

                            plt.show()

                else:
                    break


################################# Function to Predict #################################

# define function to predict

def classification_predict(model, loader):
    model.to(device)

    model.eval()

    predict = []

    # use torch.no_grad() to save memory
    with torch.no_grad():
        for data in loader:
            images, grades, labels, types, grades_class = data

            # move data to GPU
            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            # output from network
            outputs = model(images)

            # torch.max() returns the max value, torch.argmax() returns the index of the max value
            # dim = 1 => columns, dim = 0 => rows

            # argmax() returns index of the  maxmium logit or probability along class dimension, of the last linear layer
            # e.g outputs = torch.tensor([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]])
            # e.g torch.argmax(outputs, dim=1) = torch.tensor([1, 0])
            predictions = torch.argmax(outputs, dim=1)

            for i in range(data[0].shape[0]):
                
                pred = predictions[i]

                pred = pred.to("cpu").numpy()

                predict.append(pred.tolist())

    return predict    


################################# Function to Predict Regression #################################

# define function to predict

def regression_predict(model, loader):
    model.to(device)

    model.eval()

    predict = []

    # use torch.no_grad() to save memory
    with torch.no_grad():
        for data in loader:
            images, grades, labels, types, grades_class = data

            # move data to GPU
            images = images.to(device)
            grades = grades.to(device)
            labels = labels.to(device)
            types = types.to(device)
            grades_class = grades_class.to(device)

            # output from network
            # scale sigmoid output by x10 to get values from 0 to 10.
            outputs = model(images) * 10

            # .cpu() move output to cpu
            # .detact() detach tensor from computational graph, no longer tracked for gradients (is this necessary?)
            # .numpy() convert to numpy array
            # .reshape(len(outputs),) reshape to a 1 dimension array of rows and no columns
            # output is Sigmoid, so scale by 10x, and np.round() to nearest whole number and divide by 2 to get 0.5
            grade = np.round((outputs.cpu().detach().numpy().reshape(len(outputs),))*2)/2

            # list() converts numpy in to python list
            # store list in to defined list for df concat
            predict.extend(list(grade))

    return predict    
