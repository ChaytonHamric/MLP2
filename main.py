import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', 
    header=None, 
    names=[
        'ROA(C) before interest and depreciation before interest',
        'ROA(A) before interest and % after tax',
        'ROA(B) before interest and depreciation after tax',
        'Operating Gross Margin',
        'Realized Sales Gross Margin',
        'Operating Profit Rate',
        'Pre-tax net Interest Rate',
        'After-tax net Interest Rate',
        'Non-industry income and expenditure/revenue',
        'Continuous interest rate (after tax)'
        'Operating Expense Rate'
        ])
print(data)

