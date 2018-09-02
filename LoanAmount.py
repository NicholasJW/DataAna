#!/home/nicholasjw/anaconda3/bin/python3.6

print('starting analysis')
import additionalFunc as adfun
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

print('required libraries loaded')

data = pd.read_csv('loan.csv')
# print(data)

print('data loaded and starting general analysis')

status = data['loan_status'].value_counts()
# status['percentage'] = status/sum(status)
legend_index = status.index
legend_colunms = ['count', 'percentage']
legend = pd.DataFrame(index = legend_index, columns=legend_colunms)
legend['count'] = status
legend['percentage'] = 100*status/sum(status)
print(legend)
# print(set(data['loan_status']))
colors = ['yellow', 'green', 'red', 'blue', 'pink', 'lightcoral', 'black']
# plt.figure(figsize=(10,6))
patches, texts = plt.pie(status, colors=colors, labeldistance=1.2, radius=0.8, center=(0.4, 0.4))
plt.title('Overall status of all small loans in the past 5 years', fontsize=16)
legend_labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(legend_index, legend['percentage'])]
# print(legend_labels)
plt.legend(patches, legend_labels, loc='lower left')
# print(sum(status))
# print(status.index)
# print(type(status))
# plt.show()
plt.savefig('overall_status_pie')
print("pie chart saved")

plt.close()

# print(list(data))
loan_status_frame = data[['loan_amnt', 'loan_status']]
print('general histogram')
print(loan_status_frame[loan_status_frame.loan_amnt == 35000].shape[0])
print(loan_status_frame[loan_status_frame.loan_amnt == 10000].shape[0])
groups = loan_status_frame.groupby('loan_status')
# print(loan_status_frame.head())
# print(type(loan_status_frame))
bins = np.linspace(0, 35000, 15)
for name,df in groups:
    # print(name)
    # print(df.head())
    plt.hist(df['loan_amnt'], bins=bins, label=name)
plt.title('Histogram of loan amount', fontsize=16)
plt.xlabel('Loan amount')
plt.ylabel('Count')
plt.legend()
plt.savefig('loan_amount_hist')
print('Histogram saved')
plt.close()

status_sum = pd.DataFrame(columns=legend_index)
# print(status_sum)

for i in range(8):
    secname = str(i*5000) + '-' + str((i+1)*5000)
    df = loan_status_frame[(loan_status_frame.loan_amnt >= 5000*i) & (loan_status_frame.loan_amnt < 5000*(i+1))]
    status_sum.loc[secname] = np.zeros(7, dtype=int)
    for x in list(status_sum):
        print(x)
        if x in df['loan_status'].value_counts().index:
            status_sum.loc[secname][x] = df['loan_status'].value_counts()[x]
    # status_sum.iloc[i]['sum'] = status_sum.loc[i].sum()

status_sum['Sum'] = status_sum.sum(axis=1)
status_sum['Good Loans'] = status_sum['Current'] + status_sum['Fully Paid']
status_sum['Bad Loans'] = status_sum['Sum'] - status_sum['Good Loans']
status_sum['Good Rate'] = status_sum['Good Loans']/status_sum['Sum']
status_sum.to_csv('Loan_Amount_on_Status_full.csv', sep=',')
status_sum.drop(['Late (31-120 days)'], axis=1, inplace=True)
status_sum.drop(['Late (16-30 days)'], axis=1, inplace=True)
status_sum.drop(['Default'], axis=1, inplace=True)
status_sum['Good Rate'] = ['{0:1.2f} %'.format(i*100) for i in status_sum['Good Rate']]
# status_sum['bad_rate'] = status_sum['bad_loans']/status_sum['sum']
# for i in range(8):
    # print(status_sum.iloc[i]['sum'])
print(status_sum)
status_sum.to_csv('Loan_Amount_on_Status.csv', sep=',')