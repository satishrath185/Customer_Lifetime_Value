#Importing Necessary Dependencies

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Loading Dataset
data=pd.read_excel("Online Retail.xlsx")
print(data.head())
print("Shape of Data:",data.shape)

#Data Clean Up
#Negative Quantity
data=data[data['Quantity']>0]

#Cleaning Missing CustomerID
print("Number Null CustmerID:",pd.isnull(data["CustomerID"]).sum())
data=data[pd.notnull(data['CustomerID'])]

print("Shape:",data.shape)

#Number of Months of data
print("Date Range:",data["InvoiceDate"].min(),"-",data["InvoiceDate"].max())

#Last month of Data is incomplete, so better to remove it
data[data["InvoiceDate"]>"2011-12-01"].shape

data=data[data["InvoiceDate"]<"2011-12-01"]
print("Shape:",data.shape)

#Total Sales
data["Sales"]= data['Quantity'] * data['UnitPrice']
print(data.head())

#Orders Data
orders = data.groupby(["CustomerID","InvoiceNo"]).agg({'Sales': sum,'InvoiceDate': max})
print("Orders Shape:",orders.shape)
print(orders.head())

#Exploratory Data Analysis
def groupby_mean(x):
    return x.mean()

def groupby_count(x):
    return x.count()

def purchase_duration(x):
    return (x.max() - x.min()).days

def avg_frequency(x):
    return (x.max() - x.min()).days/x.count()

groupby_mean.__name__ = 'avg'
groupby_count.__name__ = 'count'
purchase_duration.__name__ = 'purchase_duration'
avg_frequency.__name__ = 'purchase_frequency'

processed_data = orders.reset_index().groupby('CustomerID').agg({
    'Sales': [min, max, sum, groupby_mean, groupby_count],
    'InvoiceDate': [min, max, purchase_duration, avg_frequency]
})
processed_data.columns = ['_'.join(col).lower() for col in processed_data.columns]

print(processed_data.shape)

#Taking into Account only repeat customers
processed_data = processed_data.loc[processed_data['invoicedate_purchase_duration'] > 0]
print(processed_data.shape)

#Visualizing How many customers belong to each category
ax = processed_data.groupby('sales_count').count()['sales_avg'][:20].plot(
    kind='bar',
    color='skyblue',
    figsize=(12,7),
    grid=True
)
ax.set_xlabel('Sales Count')
ax.set_ylabel('Count')
plt.show()

ax = processed_data['invoicedate_purchase_frequency'].hist(
    bins=20,
    color='skyblue',
    rwidth=0.7,
    figsize=(12,7)
)

ax.set_xlabel('avg. number of days between purchases')
ax.set_ylabel('count')
plt.show()

#Predicting 3 Month CLV
#Data Preparation
clv_freq = '3M'

clean_data = orders.reset_index().groupby([
    'CustomerID',
    pd.Grouper(key='InvoiceDate', freq=clv_freq)
]).agg({
    'Sales': [sum, groupby_mean, groupby_count],
})
clean_data.columns = ['_'.join(col).lower() for col in clean_data.columns]
clean_data = clean_data.reset_index()

date_month_map = {
    str(x)[:10]: 'M_%s' % (i+1) for i, x in enumerate(
        sorted(clean_data.reset_index()['InvoiceDate'].unique(), reverse=False)
    )
}

print(date_month_map)
clean_data['M'] = clean_data['InvoiceDate'].apply(lambda x: date_month_map[str(x)[:10]])

#Final Processed Dataset for Modelling
#Features Set
features_df = pd.pivot_table(
    clean_data.loc[clean_data['M'] != 'M_5'],
    values=['sales_sum', 'sales_avg', 'sales_count'],
    columns='M',
    index='CustomerID'
)
features_df = features_df.fillna(0)
print(features_df.head())

response_df = clean_data.loc[
    clean_data['M'] == 'M_5',
    ['CustomerID', 'sales_sum']
]

response_df.columns = ['CustomerID', 'CLV_'+clv_freq]

sample_set_df = features_df.merge(
    response_df,
    left_index=True,
    right_on='CustomerID',
    how='left'
)

sample_set_df = sample_set_df.fillna(0)

sample_set_df['CLV_'+clv_freq].describe()

#Model Building and Training
from sklearn.model_selection import train_test_split

target_var = 'CLV_'+clv_freq
all_features = [x for x in sample_set_df.columns if x not in ['CustomerID', target_var]]

x_train, x_test, y_train, y_test = train_test_split(
    sample_set_df[all_features],
    sample_set_df[target_var],
    test_size=0.3
)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
reg_fit = LinearRegression()
reg_fit.fit(x_train, y_train)
coef = pd.DataFrame(list(zip(all_features, reg_fit.coef_)))
coef.columns = ['feature', 'coef']
print(coef)

#Model Evaluation
from sklearn.metrics import r2_score, median_absolute_error
train_preds =  reg_fit.predict(x_train)
test_preds = reg_fit.predict(x_test)

#R Squared
print('In-Sample R-Squared: %0.4f' % r2_score(y_true=y_train, y_pred=train_preds))
print('Out-of-Sample R-Squared: %0.4f' % r2_score(y_true=y_test, y_pred=test_preds))

#Mean Absolute Error
print('In-Sample MSE: %0.4f' % median_absolute_error(y_true=y_train, y_pred=train_preds))
print('Out-of-Sample MSE: %0.4f' % median_absolute_error(y_true=y_test, y_pred=test_preds))

#Scatter Plot
plt.scatter(y_train, train_preds)
plt.plot([0, max(y_train)], [0, max(train_preds)], color='gray', lw=1, linestyle='--')

plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('In-Sample Actual vs. Predicted')
plt.grid()

plt.show()
plt.scatter(y_test, test_preds)
plt.plot([0, max(y_test)], [0, max(test_preds)], color='gray', lw=1, linestyle='--')

plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('Out-of-Sample Actual vs. Predicted')
plt.grid()

plt.show()

