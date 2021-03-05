import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv('data.csv')

print(Data)

# plt.plot(Data["Operating Profit Rate"], Data["Bankrupt?"], 'ro')
# plt.ylabel('Operating Expense Rate')
# plt.xlabel('Bankrupt Status')
# plt.show()

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
plt.show()

