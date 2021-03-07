import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

Data = pd.read_csv('data.csv')

print(f'Features and Attributes (Total): 12')
print(f'Features and Attributes (Type): Bankrupt?, ROA(C) before interest and depreciation before interest, ROA(A) before interest and % after tax, ROA(B) before interest and depreciation after tax, Operating Gross Margin, Realized Sales Gross Margin, Operating Profit Rate, Pre-tax net Interest Rate, After-tax net Interest Rate, Non-industry income and expenditure/revenue, Continuous interest rate (after tax), Operating Expense Rate')
print(f' ')
print(f'Number of Classes: 2')
print(f'Names of Classes: Bankrupt, Not Bankrupt')
print(f' ')
print(f'Dataset Partition (Training): 75%')
print(f'Dataset Partition (Testing): 25%')
print(f' ')
print(f'Min:')
print(f'{round(Data.min(),3)}')
print(f' ')
print(f'Max:')
print(f'{round(Data.max(),3)}')
print(f' ')
print(f'Mean:')
print(f'{round(Data.mean(),3)}')
print(f' ')
print(f'Median:')
print(f'{round(Data.median(),3)}')
print(f'Mode:')
print(f'{round(Data.mode(axis=1, numeric_only=True, dropna=False),3)}')
print(f'STD: ')
print(f'{round(Data.std(),3)}')

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
