import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

Data = pd.read_csv('data.csv')
# print(Data)

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
#plt.show()

X = Data.iloc[:, [1-11]]
x = scale(X)
Y = Data["A"]
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.25, random_state=42)
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)
# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}".format(lr.score(X_train, Y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, Y_test)))

# Use the trained logistic repression model to predict a new, previously unseen object
# ROAC = 0.3909228294
# ROAB = 0.4361582526
# OER = 0.0001157521304
# Bankrupt_prediction = lr.predict([[ROAC, ROAB, OER]])
# Bankrupt_probability = lr.predict_proba([[ROAC, ROAB, OER]])
# print("pass: {}".format(Bankrupt_prediction[0]))
# print("fail/pass probability: {}".format(Bankrupt_probability[0]))

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
