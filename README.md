## Machine Learning: Project 2

### Chayton Hamric

### Group 8

---

# Table of Content
- [Table of Content](#table-of-content)
- [Background](#background)
- [Statisical Summary](#statisical-summary)
- [Summary of Classification Results](#summary-of-classification-results)
- [Methods](#methods)
  - [KNN](#knn)
  - [Logistical Regression](#logistical-regression)
  - [Supported-Vector Machine](#supported-vector-machine)
- [A Brief Look at Code](#a-brief-look-at-code)
- [Conclusion](#conclusion)
- [References](#references)

---

# Background

For project 2 our group decided to use the data set that looked at bankruptcies in companies. We have two different categories with one being `Bankrupt` and the other being `Not Bankrupt`. With these two categories, we can train our data set using three different algorithms to compare each one. We used KNN, Logistical Regression and Support Vector Machine to figure our weather we could predict if a business was Bankrupt or Not Bankrupt. The data were collected from the Taiwan Economic Journal for the years 1999 to 2009. Company bankruptcy was defined based on the business regulations of the Taiwan Stock Exchange.

To help the algorithm determining the best classification we used 12 attributes:

`'Bankrupt?': 'A',
    'ROA(C) before interest and depreciation before interest',
    'ROA(A) before interest and % after tax',
    'ROA(B) before interest and depreciation after tax,
    'Operating Gross Margin',
    'Realized Sales Gross Margin',
    'Operating Profit Rate',
    'Pre-tax net Interest Rate',
    'After-tax net Interest Rate',
    'Non-industry income and expenditure/revenue',
    'Continuous interest rate (after tax)',
    'Operating Expense Rate'`

---


# Statisical Summary

With our dataset we had 96 different attributes and 7000 different companies data. Because of this we had to cut down on the amount of attributes we used and we found after testing 12 ended up giving the highest accuracy. We then had to trim our companies down dramatically this was because out of our 7000 companies we only had 222 businesses that was `bankrupt`. This made our data very unbalanced at first which so the machine ended up just assuming every new tested company was `Not Bankrupt`. For this reason we found the best consistency when we had our data be around a 50/50 split between companies that are bankrupt and those that were not. The percentage we used to train our data was 75% and then we used 25% to test. Below is all the statistical data for our Min, Max, Median, Mode, and Standard Deviation.

<center>

Figure 1
![Min](/StatSum/Min.png)

Figure 2
![Max](/StatSum/Max.png)

Figure 3
![Median](/StatSum/Median.png)

Figure 4
![Mode](/StatSum/Mode.png)

Figure 5
![Standard Deviation](/StatSum/STD.png)

</center>


---


# Summary of Classification Results

For the results we compared three different algorithms to see who would preform best by giving the most accurate data in testing weather a company is `Bankrupt` or `Not Bankrupt`.

We tested with KNN first to see how our data set would line up and for this we found that KNN was the worst preforming algorithm. The reason we say this is because KNN had only a 65% chance of guessing if the company was `Not Bankrupt`. This of coarse is bad because not that skews the data making the accuracy of predicting if a company is `Bankrupt`. We believe this is because KNN is better for large datasets and because our data set is only 222 `Bankrupt` and `Not Bankrupt` companies each it had a hard time training. Below is the confusion matrix for KNN checking 15 nearest neighbors.

<center>

Figure 1
![KNN confusion matrix](/Graphs/KNN.png)

</center>

Next we used Logistical Regression and Support Vector Machine. Both of these had great accuracy with them both having ~80% predictability if a company was `Bankrupt` or `Not Bankrupt`. Below is the two confusion matrixes for Logistical Regression and Support Vector Machine.

<center>

Logistical Regression
![Logistical Regression confusion matrix ](/Graphs/LR.png)

Support Vector Machine
![Support Vector Machine confusion matrix](/Graphs/SVM.png)

</center>

---

---
# Methods

## KNN

Machine learning algorithm that searches and compares a specific number of the closest neighbors to determine an output that best represents its nearest neighbors. Does not support outliers very well.

## Logistical Regression

Machine learning algorithm that uses probability to determine the output. This can be used with several classes. Does not support outliers very well.

## Supported-Vector Machine

Machine learning algorithm that uses classification that supports both liner and non-linear. Does support outliers very well.

---

# A Brief Look at Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

Data = pd.read_csv('data.csv')

Data.rename(columns={'Bankrupt?': 'A',
    'ROA(C) before interest and depreciation before interest': 'B',
    'ROA(A) before interest and % after tax': 'C',
    'ROA(B) before interest and depreciation after tax': 'D',
    'Operating Gross Margin': 'E',
    'Realized Sales Gross Margin': 'F',
    'Operating Profit Rate': 'G',
    'Pre-tax net Interest Rate': 'H',
    'After-tax net Interest Rate': 'I',
    'Non-industry income and expenditure/revenue': 'J',
    'Continuous interest rate (after tax)': 'K',
    'Operating Expense Rate': 'L'}, inplace=True)

plt.style.use('dark_background')
pd.plotting.scatter_matrix(Data)

X = Data.iloc[:, [1-11]]
x = scale(X)
Y = Data["A"]
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=42)
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, Y_test)))

# KNN Confusion Matrix
knn = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=1, weights='distance')
knn.fit(X_train, Y_train)
pcm(knn, X_test, Y_test, normalize='true', display_labels=['Not Bankrupt', 'Bankrupt'])


# LR Confusion Matrix
lrc = LogisticRegression(solver='liblinear', random_state=0)
lrc.fit(X_train, Y_train)
pcm(lrc, X_test, Y_test, normalize='true', display_labels=['Not Bankrupt', 'Bankrupt'])

# SVM Confusion Matrix
svc = SVC(C=10, degree=1, kernel='poly')
svc.fit(X_train, Y_train)
pcm(svc, X_test, Y_test, normalize='true', display_labels=['Not Bankrupt', 'Bankrupt'])
plt.show()

```

----

# Conclusion

In conclusion we found that we were able to predict up to ~80% accuracy if a company is `Bankrupt` or `Not Bankrupt`. We also found out that at least for our dataset KNN was not the best in predicting with high accuracy but if we were to have a much larger dataset with a lot more of the two categories we could see an increase in accuracy. But for KNN the accuracy for predicting if a company is `Not Bankrupt` was too low. On the other hand with the data we have the two best algorithms were Logistical Regression and Supported-Vector Machine. These both gave ~80% accuracy in both predicting if the company was `Bankrupt` or `Not Bankrupt`.

---

# References

* [Wikipedia On Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
* [Kaggle Bankruptcy Prediction Dataset](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction)
* [Wikipedia on KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* [Wikipedia on Logistical Regression](https://en.wikipedia.org/wiki/Logistic_regression)
* [Wikipedia on Support-Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine)