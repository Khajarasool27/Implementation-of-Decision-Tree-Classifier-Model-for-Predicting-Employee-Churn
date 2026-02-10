# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.  

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:  B.Khaja Rasool
RegisterNumber: 212224230040
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 


data = pd.read_csv("C:\\Users\\admin\\Downloads\\Employee (1).csv")


data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()


x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()
y = data["left"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)


y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy


dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:
<img width="1259" height="493" alt="image" src="https://github.com/user-attachments/assets/73b81df4-af34-4e6c-a4f1-1728ce60c282" />

<img width="1633" height="822" alt="image" src="https://github.com/user-attachments/assets/489a55ef-2447-4cdc-b2e7-eb59e35f026d" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
