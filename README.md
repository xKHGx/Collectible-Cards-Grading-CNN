# GA DSIF 13 - Capstone

## Topic Introduction - Authenication and Grading of collectible cards
Collector Universe, founded in 1986, was the first company to provide authentication and grading services to collectors of rare coins, stamps, autographs and sports cards. Its sports card grading division, Professional Sports Authenticator (PSA), was established in 1991 and grew to become the industry's gold standard in authentication and grading services.

Over the years, PSA expanded its services to include other types of products, such as tickets, packs, memorabilia and more. The growth of the industry attracted many new entrants, such as Beckett Grading Services ('Beckett'), Certified Guaranty Company ('CGC') and Sportscard Guarantee Corporation ('SGC') amongst many others. These companies shall be collectively term 'grading companies' from here on.

Collectible cards are actively traded on online market places such as eBay, facebook, Amazon, TCGplayer, and many others. Prices are determined base on a number of factors, such as rarity, supply and demand and grade. 

Cards are graded based on the following standards/ metrics (in no particular order):
- Surface:
  - Free of stains, smudges, scratches
  - Free of manufacturer print defects, ink smears
- Corners
  - Free of dings, creases, bends and discoloration
- Edges
  - Free of frays, chips and dings 
- Centering
  - Even border on all sides

Cards are graded on an ascending numerical scale of 1-10, with increments of either 0.5 or 1, depending on the grading company.

More information is available in these links:
- https://comics.ha.com/tutorial/tcg-card-grading.s
- https://public.com/learn/psa-grading-bgs-beckett-grading-card-guide?wpsrc=Organic+Search&wpsn=www.google.com
- https://www.beckett.com/grading-standards

## Opportunity Statement
Card grading is an expensive and long-drawn process. From PSA's services page, the regular non-member price per card is US$ 74.99, with an estimated turnaround time of 45 business days. The cheapest price available is US$ 14.99, which is available only to members, who have to incur additional membership fees. Other grading companies have similar fee structures.

In addition to high price and long delivery time, grading companies also take on the risk of phyiscal cards being damaged whilst in their posession. With cards value soaring in recent years, grading and authentication prices are likely to increase further to compensate such risk.

In view of the above inefficiencies and risk, this capstone explores the use of convolutional neural network techniques to provide a low-cost and efficient way to grade cards accurately, without the risk of phyiscal damage.

The intention is not to replace the need for physical grading. Rather, it serves to provide users a quick and efficient way to decide on which cards have an expected high grade, and thus should be sent for phyiscal grading, minimising cost while maximising user's profitability and return. 

## Model Approach
This capstone leverages on various pre-trained torchvision and timm models to speed up training time. Both regression and classification techniques are implemented to compare which technique predicts the grade with the lowest error rate and highest accuracy.

In addition to grade prediction, type and side predictions are also explored in this capstone.

End-user interaction is provided by the Streamlit application.
    
## Data Collection
Image and text data were scrapped from website using Selenium. Note that data was scrapped from only one company, hence the data is likely skewed towards the grading approach of the company. Nonetheless, given that the company has graded the majority of the slabs worldwide, the sample data collected is representative of the grade population.

## Data Preprocessing
The following steps were taken to preprocess image data:

1. Images were resize to (W) 1,400 x (H) 2,400 to standardize the crop area to extract card image, without card details, for model training, validation and testing.
2. The crop function excluded non-card images (e.g packs, tickets)
3. Images with file name longer than 180 characters were excluded as open(path, "rb") errors out during dataiter. 
4. Cropped images were concatenated on same card IDs to form complete images (e.g front and back concate to form one single image).
5. Target information were extracted from the text data and exported to csv.


## Exploratory Data Analysis (EDA) on Processed Data
Total samples (Front and Back are separately counted): 107,682\
Total samples collected (Front and Back are merged as one): 53,684

Sample grade distribution table: Each count represents a single card.

**Note:** Frequency did not sum to 100% due to rounding differences.

| Grade  | Count    | Frequency |
|--------|----------|-----------|
| 1      | 188      | 0.35%     |
| 1.5    | 68       | 0.13%     |
| 2      | 309      | 0.58%     |
| 2.5    | 36       | 0.07%     |
| 3      | 464      | 0.86%     |
| 3.5    | 35       | 0.07%     |
| 4      | 780      | 1.45%     |
| 4.5    | 28       | 0.05%     |
| 5      | 1,177    | 2.19%     |
| 5.5    | 21       | 0.04%     |
| 6      | 1,926    | 3.59%     |
| 6.5    | 17       | 0.03%     |
| 7      | 2,500    | 4.66%     |
| 7.5    | 21       | 0.04%     |
| 8      | 6,506    | 12.12%    |
| 8.5    | 103      | 0.19%     |
| 9      | 16,087   | 29.97%    |
| 10     | 23,418   | 43.62%    |
|--------|----------|-----------|
| Total  | 53,684   | 100%      |


| Type          | Count    | Frequency |
|---------------|----------|-----------|
| BASEBALL      | 10,763   | 20.05%    |
| BASKETBALL    | 7,618    | 14.19%    |
| FOOTBALL      | 8,585    | 15.99%    |
| GOLF          | 38       | 0.07%     |
| HOCKEY        | 1,662    | 3.10%     |
| LEAGUE        | 120      | 0.22%     |
| MISC          | 347      | 0.65%     |
| MULTI-SPORT   | 275      | 0.51%     |
| NON-SPORT     | 2,792    | 5.20%     |
| SOCCER        | 2,291    | 4.27      |
| TCG           | 18,617   | 34.68%    |
| WRESTLING     | 576      | 1.07%     |
|---------------|----------|-----------|
| Total         | 53,684   | 100%      |

**Findings:**
- Data collected is highly unbalanced.
- Grades 9 and 10 accounted for 74%. 
- TCG and Baseball accounted for 55%.
- Grade table suggests that users are likely able to discern grades < 8 and grades >= 8.


## Data transformation
The following steps were taken to transform the image data:

1. Image data was resized to model(s) input specification: (224 x 224)
2. Image data was normalised with ImageNet's mean of (0.485,0.456,0.406) and standard deviation of (0.229,0.224,0.225). 
    - This can be improved by finding and using the custom dataset's mean and standard deviation to normalise instead.
3. Data augmentations were applied to simulate real world picture taken at varied angles, distance and lighting, with objects in both foreground and background.
4. Image data was converted to tensor for model training and evaluation.


## Model Selection and Training
The following steps were taken to select the appropriate models for both Type classification and Grade regression, using a subset of the raw data (2,848 samples):

1. Select three pre-trained models based on the model's accuracy @ 1%, 5% and number of parameters. All train paramenters used were the same.
    - ResNet50-DEFAULT (torchvision - accuracy @ 1%: 80.858, @ 5%: 95.434, params: 25.6M) - baseline
    - EfficientNet_V2_L-DEFAULT (torchvision - accuracy @ 1%: 85.808, 5%: 97.788, params: 118.5M)
    - RexNet200-DEFAULT (timm - accuracy @ 1%: 83.168, @5%: 96.654, params: 16.52M)
    - ResNet152-DEFAULT (torchvision - accuracy @ 1%: 82.284, @ 5%: 96.002, params: 60.2M)

2. Findings:
    - For Type classification, Resnet50d achieved the Best loss.
        - ResNet50: Best loss: 1.4423
        - EfficientNet_V2_L: Best loss: 1.5966
        - RexNet200: Best loss: 1.9279
        
        Further test was done with ResNet152, which acheived Best loss of 1.2146. ResNet152 was selected as choice model for Type classification.
        - ResNet50: Best loss: 1.4423
        - **ResNet152: Best loss: 1.2146**

    - For Grade regression, Rexnet_200 achieve the best loss. It was selected as choice model for Grade Regression.
        - ResNet50: Best loss: 1.1027
        - EfficientNet_V2_L: Best loss: 1.2469
        - **RexNet200: Best loss: 1.090**

## Model Training
Each Type and Grade category were limited to a maximum of 2,000 randomly selected samples, due to: 
  - imbalanced dataset
  - computationally time-consuming for RTX 3060 GPU

Data was split in to Train, Validation and Test in 0.75: 0.15: 0.10 ratio, stratified by target.

**Model heads** were modified to learn the new targets. Sigmoid activation function was implemented for regression and scaled by 10x to obtain the grade.

**Unfreeze layers**: Model head and preceding model layer were unfreezed to update weights and bias for better results.

**Optimizer**: AdamW 
- Adam weight decay implented in Pytorch is incorrect. Refer to https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay

**Loss criterion**: 
- Classification: Cross Entropy
- Regression: MSELoss(reduction='mean')

**Scheduler**: Cosine Annealing Warm Restart
- Avoid getting stuck at local minima
- Explore new optima
- Improve model performance
- Reduce overfitting

**Epochs (early stop)**: 100 (7)


## Model Evaluation

The following steps were taken to train and evaluate the results

1. Side Classification
   - Best validation accuracy: 98.93%
   - Best validation loss: 0.0466
   - Best test accuracy: 99.00%
   - | Side     | Precision | Recall |F1-Score|
     |----------|-----------|--------|--------|
     | 0 (Back) | 0.99      | 0.99   | 0.99   |
     | 1 (Front)| 0.99      | 0.99   | 0.99   |

2. Type Classification
   - Best validation accuracy: 94.36%
   - Best validation loss: 0.2886
   - Best test accuracy: 92.21% 
   - | Type           | Precision | Recall |F1-Score|
     |--------------  |-----------|--------|--------|
     | 0 (Baseball)   | 0.86      | 0.94   | 0.94   |
     | 1 (Basketball) | 0.97      | 0.89   | 0.93   |
     | 2 (Football)   | 0.94      | 0.94   | 0.94   |
     | 3 (Hockey)     | 0.98      | 0.95   | 0.96   |
     | 4 (League)     | 1.00      | 0.33   | 0.50   |
     | 5 (Misc)       | 0.88      | 0.80   | 0.84   |
     | 6 (Multi-sport)| 0.59      | 0.48   | 0.53   |
     | 7 (Non-sport)  | 0.94      | 0.94   | 0.94   |
     | 8 (Soccer)     | 0.85      | 0.96   | 0.90   |
     | 9 (TCG)        | 1.00      | 0.99   | 0.99   |
     | 10 (Wrestling) | 0.90      | 0.79   | 0.84   |
     | 11 (Golf)      | 1.00      | 0.75   | 0.86   |

3. Grade Regression
    - Test samples achieved a RMSE of 1.560. Considering the grades are between 1 to 10, the error is too large for the model to reliability predict accurate grades.

4. Grade Classification (trial)
   - Best validation accuracy: 0.7591%
   - Best validation loss: 16.4965
   - It is evident that Grade classification performs worse (e.g higher loss) than Grade Regression. As such, the trial was put to a stop.


## Findings

- The selected models were further tested with images extracted from eBay, to assess models' performance on real world images taken by users. (e.g with noises)
- **Side classification models** trained without data augmentations perform very well on eBay data.
- **Type classification models** trained without data augmentations performed poorly on eBay data, likely due to angles, distance, and objects in both foreground and background.
- **Retrained type classification models** with data augmentation perfomed much better while mantaining similar loss and accuracy scores.
- **Grade regression models** performed poorly on eBay data even with data augmentation. Given that the RMSE is 1.560 and the grade range is 1 - 10, it is hardly useful to users to make sound economic decisions.

## Improvements
- Obtain more data especially on the lower range grades (e.g 1-7) and half-point grades.
- Train larger models and unfreeze more layers to fine tune models' weights and biases.
- Ensemble models trained on the four grading standards


## Presentation Slide
[Project Presentation](https://docs.google.com/presentation/d/1ayRNaXmC1KZ2o21sywcMrnGyUqF42Ifm_ZBHtPsVIXM/edit?usp=sharing)

## Dataset
The dataset is available on request.

## Running the Code
To run the code in this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries listed in the `requirements.txt` file.
3. Description of the following files:

- **helper_functions.py**
  - Contains all the functions to run the notebooks below.
- **Image Scrapper**
  - Scrape image and text data from website via XPATHs.
- **Image Preprocessor**
  - Filter away non-card images such as tickets, packs, etc
  - Resize images to a standard size, and crop out the card image
  - Concat images with same ID number
  - Seperate text with regex on delimiter, and extract required targets
  - Save image filename and targets in a csv file
- **Image Classification and Regression with Pytorch**
  - Define image and data paths
  - Define image transformation pipeline
  - Encode categorical data to numerical data
  - Define the target, sample size, train/validate/test split ratio and dataloader
  - Define models for the task, and tweak the training parameters accordingly
  - Evaluate models with test data and choose the best model

## Running the Streamlit App
To run the Streamlit app, follow these steps:

1. Ensure you have Streamlit installed. If not, install it using the following command:
   ```bash
   pip install streamlit
   ```
2. Navigate to the directory containing the `streamlit.py` file.
3. Run the Streamlit app using the following command in the terminal:
   ```bash
   streamlit run streamlit.py [-- script args]
   ```
4. The app will open in your default web browser. You can upload the front and back image of the card and the app will predict the following:
   - **Side**: Front or Back
   - **Type**: Football, Basketball, TCG, etc
   - **Grade**: Card grade between 1 to 10

## Repository Structure
- `helper_functions.py`: Contains all functions used in the capstone.
- `Image Scrapper.ipynb`: Image and text data web scrapper 
- `Image Preprocessor.ipynb`: Image and text data preprocessor
- `Image Classification and Regression with Pytorch.ipynb`: Model training and evaluation
- `streamlit.py`: Streamlit app script.
- `Raw Data/`: Sample scrapped images.
- `Raw Crop/`: Sample cropped images.
- `Raw Concat/`: Sample concatenated images.
- `models/`: Trained models with best loss.


