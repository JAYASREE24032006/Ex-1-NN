# EX-1 : INTRODUCTION TO KAGGLE AND DATA PREPROCESSING
#### ENTER YOUR NAME : JAYASREE R
#### ENTER YOUR REGISTER NO : 212223040074
#### DATE : 17-09-2025

## AIM :

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED :
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT :

**Kaggle :**

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing :**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM :

STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM :
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("Churn_Modelling.csv")
df
df.isnull().sum()
df.fillna(0)
df.isnull().sum()
df.duplicated()
df['EstimatedSalary'].describe()
scaler = StandardScaler()
inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
scaled_values = scaler.fit_transform(df[inc_cols])
df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
df
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
print("X Values")
x
print("Y Values")
y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
print("X Training data")
x_train
print("X Testing data")
x_test
```

## OUTPUT :

#### Read the dataset from drive
<img width="1040" height="332" alt="image" src="https://github.com/user-attachments/assets/6f5430f6-c6b7-4d48-8b4c-28f90b07bda5" />

#### Finding Missing Values
<img width="148" height="413" alt="image" src="https://github.com/user-attachments/assets/781d5e22-3bbd-435e-91ca-5d06b4335338" />

#### Handling Missing values
<img width="131" height="423" alt="image" src="https://github.com/user-attachments/assets/da105877-fc03-4f72-ad8b-95287b89aceb" />

#### Check for Duplicates
<img width="175" height="365" alt="image" src="https://github.com/user-attachments/assets/df82fa29-7bab-4a85-b616-0e35e1d11ef7" />

#### Detect Outliers
<img width="192" height="254" alt="image" src="https://github.com/user-attachments/assets/0bd33084-40b1-4a10-b1cc-996a5bf8f222" />

#### Normalize the dataset
<img width="1039" height="367" alt="image" src="https://github.com/user-attachments/assets/387a55ee-0c4d-4b0a-a07c-d470a914c2bf" />

#### Split the dataset into input and output
<img width="998" height="347" alt="image" src="https://github.com/user-attachments/assets/d74e4626-38d3-4e0e-8db1-334827bdb0f1" />

<img width="275" height="379" alt="image" src="https://github.com/user-attachments/assets/6ddec14f-d5bb-4ffe-b201-3a5c9fc15ce3" />

#### Print the training data and testing data
<img width="1003" height="351" alt="image" src="https://github.com/user-attachments/assets/fd13c8c1-1cae-4d83-8517-f693ef75baa3" />

<img width="967" height="322" alt="image" src="https://github.com/user-attachments/assets/abab8d74-09fd-49cc-8d4c-d5a0a9ba86ce" />

## RESULT :
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


