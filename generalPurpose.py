#!/home/nicholasjw/anaconda3/bin/python3.6

print('starting analysis')
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

print('required libraries loaded')

data = pd.read_csv('loan.csv')

purpose = data['purpose'].value_counts()
print(purpose)

plt.figure(figsize=(8,6)) 
patches, texts = plt.pie(purpose, radius=1, center=(0.1, 0))
plt.legend(patches, purpose.index, loc='center left')
plt.title('Purposes of loans', fontsize=16)
# plt.show()
plt.savefig('purpose_pie_chart')

purpose_status_frame = data[['purpose', 'loan_status']]
groups = purpose_status_frame.groupby('purpose')
purpose_status_sum = pd.DataFrame(columns=data['loan_status'].value_counts().index)
# print(list(purpose_status_frame))
for name,df in  groups:
    print(name)
    # print(df)
    purpose_status_sum.loc[name] = np.zeros(7, dtype=int)
    for x in list(purpose_status_sum):
        if x in df['loan_status'].value_counts().index:
            purpose_status_sum.loc[name][x] = df['loan_status'].value_counts()[x]

# print(purpose_status_sum)

purpose_status_sum['Sum'] = purpose_status_sum.sum(axis=1)
purpose_status_sum['Good Loans'] = purpose_status_sum['Current'] + purpose_status_sum['Fully Paid']
purpose_status_sum['Bad Loans'] = purpose_status_sum['Sum'] - purpose_status_sum['Good Loans']
purpose_status_sum['Good Rate'] = purpose_status_sum['Good Loans']/purpose_status_sum['Sum']
purpose_status_sum.to_csv('Purpose_on_Status_full.csv', sep=',')
purpose_status_sum.drop(['Late (31-120 days)'], axis=1, inplace=True)
purpose_status_sum.drop(['Late (16-30 days)'], axis=1, inplace=True)
purpose_status_sum.drop(['Default'], axis=1, inplace=True)
purpose_status_sum['Good Rate'] = ['{0:1.2f} %'.format(i*100) for i in purpose_status_sum['Good Rate']]

print(purpose_status_sum)
purpose_status_sum.to_csv('Purpose_on_Status.csv', sep=',')