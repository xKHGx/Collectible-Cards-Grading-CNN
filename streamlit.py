# run streamlit in terminal in the correct Project folder: streamlit run streamlit.py [-- script args]
# exit streamlit from terminal: ctrl + c

import PIL.Image
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
from torchvision.models import (
    resnet50,
    resnet152,
    vgg16,
    mobilenet_v3_large, 
    efficientnet_v2_l,
    )
import PIL
import torchvision.transforms as transforms
import cv2
import timm
from time import sleep


# use GPU is cuda is available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# concat images
def concat_img(img_1, img_2):
    img_1 = PIL.Image.open(img_1)
    img_1 = img_1.resize((1440,2400))
    img_1 = np.array(img_1)
    # img_1 = img_1.save('img1.jpg')
    img_2 = PIL.Image.open(img_2)
    img_2 = img_2.resize((1440,2400))
    img_2 = np.array(img_2)
    # img_2 = img_2.save('img2.jpg')
    h_img = cv2.hconcat([img_1, img_2])
    h_img = PIL.Image.fromarray(h_img)

    return h_img

############### Side Classification Model ###############

# load classification model from local repository
model_path = r'..\Project_5\models'
load_model = 'ResNet50_Side_epoch_1_val_loss_0.0466_val_acc_98.8000.pth'

# instantiate model
side_model = resnet50()

# 'Back' and 'Front' classes
num_classes = 2

# modify model head for classification with num_classes at the final layer.
in_features = side_model.fc.in_features
side_model.fc = nn.Sequential(
                    nn.Linear(in_features, 640),
                    nn.ReLU(),
                    nn.Linear(640, num_classes))

# load weights from save model
side_model.load_state_dict(torch.load(f'{model_path}\{load_model}', weights_only=True))

 
############### Side Classification function ###############
def side_classification(model, image_path):
   model.to(device)

   # open image with Pillow
#    img = PIL.Image.fromarray(image_path)
   img = PIL.Image.open(image_path)

   # ImageNet mean and std
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]

    # transform image for model eval
   transform_norm = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)])
   
   img_normalized = transform_norm(img).float()

   # unsqueeze_(0) performs operation inplace. re-assignment not required. saves memory as temp instance isnt created
   img_normalized.unsqueeze_(0)

    # move image to gpu
   img_normalized = img_normalized.to(device)

   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)

      predictions = torch.argmax(output, dim=1)

      # convert numpy to python int
      predict = int(predictions.to("cpu").numpy().tolist()[0])
      
      side_num = {
        0 : 'Back',
        1 : 'Front'
        }
      
      side = side_num[predict]

      return side


############### Type Classification Model ###############

# load regression model from local repository
model_path = r'..\Project_5\models'
load_model = 'ResNet152_Type_epoch_8_val_loss_0.1983_val_acc_95.3839.pth'

# 12 types classes
num_classes = 12

# load pretrained resnet50 from torchvision
type_model = resnet152()

in_features = type_model.fc.in_features
type_model.fc = nn.Sequential(
                    nn.Linear(in_features, 640),
                    nn.ReLU(),
                    nn.Linear(640, num_classes))

# load weights from save model
type_model.load_state_dict(torch.load(f'{model_path}\{load_model}', weights_only=True))


############### Type Classification function ###############

# image classification function
def type_classification(model, image_path):
   model.to(device)

#    open image with Pillow
#    img = PIL.Image.open(image_path)

   # ImageNet mean and std
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]

    # transform image for model eval
   transform_norm = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)])
   
   img_normalized = transform_norm(image_path).float()

   # unsqueeze_(0) performs operation inplace. re-assignment not required. saves memory as temp instance isnt created
   img_normalized.unsqueeze_(0)

    # move image to gpu
   img_normalized = img_normalized.to(device)

   with torch.no_grad():
        model.eval()  
        output =model(img_normalized)

        predictions = torch.argmax(output, dim=1)

        # convert numpy to python int
        predict = int(predictions.to("cpu").numpy().tolist()[0])

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

        type = type_num[predict]

        return type


############### Grade Regression Model ###############

# load regression model from local repository
model_path = r'..\Project_5\models'
load_model = 'RexNet_RatingNum_epoch_8_val_loss_1.4761.pth'

# 'Back' and 'Front' classes
num_classes = 1

# load pretrained resnet50 from torchvision
grade_model = timm.create_model('rexnet_200', pretrained=False, num_classes=num_classes)

# modify model head for classification with num_classes at the final layer.
# Sigmoid to restrict output [0,1]. Scale it by 10x, and round to nearest whole number and divide by 2 to get 0.5
in_features = grade_model.head.fc.in_features
grade_model.head.fc = nn.Sequential(
                            nn.Linear(in_features, 640),
                            nn.ReLU(),
                            nn.Linear(640, num_classes),
                            nn.Sigmoid())


# load weights from save model
grade_model.load_state_dict(torch.load(f'{model_path}\{load_model}', weights_only=True))


############### Grade Regression function ###############

# image classification function
def image_regression(model, image_path):
   model.to(device)

   # open image with Pillow
#    img = PIL.Image.open(image_path)

   # ImageNet mean and std
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]

    # transform image for model eval
   transform_norm = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)])
   
   img_normalized = transform_norm(image_path).float()

   # unsqueeze_(0) performs operation inplace. re-assignment not required. saves memory as temp instance isnt created
   img_normalized.unsqueeze_(0)

    # move image to gpu
   img_normalized = img_normalized.to(device)

   with torch.no_grad():
      model.eval()  
      output =model(img_normalized) * 10

      grade = np.round((output.cpu().detach().numpy().reshape(len(output),))*2)/2

      return grade
   

############### Streamlit UI ###############

# user streamlit image upload
MAX_IMAGES = 2

uploaded_files = st.file_uploader("Choose an image file", accept_multiple_files=True)

if len(uploaded_files) > 2:
    st.warning(f"Maximum number of files reached. Only the first {MAX_IMAGES} will be processed.")
    uploaded_files = uploaded_files[:MAX_IMAGES]

image_list = []
caption = []

# looping through uploaded images and classify them in to front and back
for uploaded_file in uploaded_files:
    predict = side_classification(side_model, uploaded_file)
    image_list.append(uploaded_file)
    caption.append(predict)
    
st.image(image_list, caption=caption, width=200)

# concat img for regression
# try except so that streamlit will not display error messages during initial page load
try: 
    img_concat = concat_img(image_list[0], image_list[1])
except Exception:
    pass

# predict type classification
try: 
    type = type_classification(type_model, img_concat)
    st.write(f'Predicted Type: {type}')
except Exception:
    pass

# predict grade
try: 
    grade = int(image_regression(grade_model, img_concat))
    st.write(f'Predicted Grade: {grade}')
except Exception:
    pass