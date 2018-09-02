#!/home/nicholasjw/anaconda3/bin/python3.6

print('starting analysis')
from scipy.stats import norm
import additionalFunc as adfun
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

print('required libraries loaded')

# print(adfun.p2f('33.33%'))
data = pd.read_csv('loan.csv')
print(data.head())

print('data loaded and starting general analysis')

insrate_status_frame = data[['int_rate', 'loan_status']]
insrate_status_frame['int_rate'] = insrate_status_frame['int_rate'].str.rstrip('%').astype('float') / 100.0
groups = insrate_status_frame.groupby('loan_status')
bins = np.linspace(0, 0.3, 15)
ints = insrate_status_frame['int_rate'].values
ints = ints[np.isfinite(ints)]
# print(type(ints))
# mu, std = np.nanmean(ints), np.nanstd(ints)
mu, std = norm.fit(ints)
plt.hist(ints, bins=bins, density=True)

plt.title('Histogram of interest rate', fontsize=15)
# plt.legend()
plt.xlabel('Interest rate')
plt.yticks([])
# plt.ylabel('Count')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, linewidth=2)
title = "Fit results: mean = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
# plt.show()
plt.savefig('intrate_hist')

insrate_sum = pd.DataFrame(columns=data['loan_status'].value_counts().index)
# insrate_status_frame['int_rate'] = insrate_status_frame['int_rate'].str.rstrip('%').astype('float') / 100.0

for i in range(15):
    secname =  str(i*0.02) + '-' + str((i+1)*0.02)
    df = insrate_status_frame[(insrate_status_frame.int_rate >= i*0.02) & (insrate_status_frame.int_rate < (i+1)*0.02)]
    insrate_sum.loc[secname] = np.zeros(7, dtype=int)
    for x in list(insrate_sum):
        if x in df['loan_status'].value_counts().index:
            insrate_sum.loc[secname][x] = df['loan_status'].value_counts()[x]

insrate_sum['Sum'] = np.zeros(15)
insrate_sum['Sum'] = insrate_sum.sum(axis=1)
insrate_sum['Good Loans'] = insrate_sum['Current'] + insrate_sum['Fully Paid']
insrate_sum['Bad Loans'] = insrate_sum['Sum'] - insrate_sum['Good Loans']
# insrate_sum['good_rate'] = np.zeros(15)
# result = insrate_sum.copy()
insrate_sum['Sum'] += 0.001
insrate_sum['Good Rate'] = insrate_sum['Good Loans']/insrate_sum['Sum']
insrate_sum['Sum'] -= 0.001
# insrate_sum[['Sum', 'Bad Loans']] -= 0.001
insrate_sum.to_csv('Interest_Rate_on_Status_full.csv', sep=',')
insrate_sum.drop(['Late (31-120 days)'], axis=1, inplace=True)
insrate_sum.drop(['Late (16-30 days)'], axis=1, inplace=True)
insrate_sum.drop(['Default'], axis=1, inplace=True)
insrate_sum['Good Rate'] = ['{0:1.2f} %'.format(i*100) for i in insrate_sum['Good Rate']]

print(insrate_sum)
insrate_sum.to_csv('Interest_Rate_on_Status.csv', sep=',')