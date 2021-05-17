# Logistic-Regression
Explanation of logistic regression with a dataset Advertising Dataset

Def : Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables 

## Using Logistic Regression Algorithm on Advertising Dataset:

In the beginning, we import all required libraries
 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```
Pandas is used for data analysis. The library allows various data manipulation operations such as merging, reshaping, selecting, as well as data cleaning, and data wrangling features

Numpy is used in the industry for array computing

Seaborn is a Python library used for enhanced data visualization

## Importing Dataset

```python
df = pd.read_csv('/content/advertising.csv')
```
importing dataset is done using .read_csv


## Getting More Information
```python
df.head()
df.info()
df.describe()
```
![12](https://user-images.githubusercontent.com/82372055/118535133-b6b8a200-b767-11eb-9206-00cc0d745d46.png)


## Exploratory Data Analysis:
Exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods
Let's use seaborn to explore the data!
```python
sns.histplot(data=df, x='Age',bins=30)
```
Using this we get Histogram plot of data Age
![Screenshot 2021-05-17 233028](https://user-images.githubusercontent.com/82372055/118535271-e23b8c80-b767-11eb-9ff9-c9f434483c0f.png)

```python
sns.jointplot(x='Age', y='Area Income',data= df)
```
Using this we get jointplot between Age and Income   
![Screenshot 2021-05-17 233125](https://user-images.githubusercontent.com/82372055/118535377-05663c00-b768-11eb-9a11-3a2db0cfc425.png)

```python
sns.jointplot(x= 'Age', y= 'Daily Time Spent on Site', data= df,kind='kde', color='red')
```
Using this we get a join plot with the KDE distributions of Daily Time spent on-site vs. Age.
![Screenshot 2021-05-17 233206](https://user-images.githubusercontent.com/82372055/118535445-1d3dc000-b768-11eb-9741-208dc5420e09.png)

```python
sns.jointplot(x= 'Daily Time Spent on Site', y='Daily Internet Usage',data=df)
```
Using this we get joint plot between Daily Time Spent on Site and Daily Internet Usage   
![Screenshot 2021-05-17 233236](https://user-images.githubusercontent.com/82372055/118535520-2fb7f980-b768-11eb-89d8-e894bae6b15d.png)
```python
sns.pairplot(data=df, hue='Clicked on Ad')
```
this is a pair plot with the hue defined by the 'Clicked on Ad' column feature
pic

![download](https://user-images.githubusercontent.com/82372055/118535592-48c0aa80-b768-11eb-8177-3967e1f9a768.png)

## Now its time for Training LinearRegression Our Model:
first, we import train test split 
```python
from sklearn.model_selection import train_test_split
```
now at we split Features (X) and Labels (y) as below.
```python
X = df[["Daily Time Spent on Site", 'Age', 'Area Income','Daily Internet Usage', 'Male']]
```

Then we split into test and train datasets using the following code
```python
x_train, x_test, y_train, y_test = train_test_split(X,df["Clicked on Ad"],test_size=0.30,random_state=42)
```

now we are ready to start fitting the Data in Model using LinearRegression
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegressionRegression()
lm.fit(x_train,y_train)
```
Training our logistic regression model is Done. by fitting a best linear is done by using lr.fit(x_train,y_train) method.

## Predictions and Evaluations
We predict the values for our testing set (x_test) and save it in the predictions variable 

```python
predictions= lm.predict(x_test)
```
now we can check the accuracy score to check how good our model has done training.
importing classification_report,confusion_matrix
```python
from sklearn.metrics import classification_report,confusion_matrix

```
confusion matrix
```python

print(confusion_matrix(y_test,predictions))

```
classification report
```python
print(classification_report(y_test,predictions))
```

![Screenshot 2021-05-17 233403](https://user-images.githubusercontent.com/82372055/118535704-6a219680-b768-11eb-9d10-06227cc049ce.png)
