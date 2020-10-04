import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error



data = pd.read_excel('~/Downloads/ANZ/cleaned.xlsx', index='customer_id')
#print (data.head())

data.drop_duplicates(['customer_id'], keep='first', inplace=True)
data['annual_salary'] = data['amount'] * 52
salary = data[data['txn_description'] =='PAY/SALARY']


age_salary = salary[['age', 'annual_salary']]
#print(age_salary)
X = age_salary.iloc[:, :-1].values
#X = X.reshape(X.shape[1:])
#X = X.transpose()
y = age_salary.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

ax_train = plt
ax_train.scatter(X_train, y_train, color='red')
ax_train.plot(X_train, regressor.predict(X_train), color='blue')
ax_train.title('Annual Salary vs Customer Age (Data Training)')
ax_train.xlabel('Age')
ax_train.ylabel('Annual Salary')
ax_train.show()

ax_test = plt
ax_test.scatter(X_test, y_test, color='red')
ax_test.plot(X_test, regressor.predict(X_test), color='blue')
ax_test.title('Annual Salary vs Customer Age (Data Test)')
ax_test.xlabel('Age')
ax_test.ylabel('Annual Salary')
ax_test.show()

#y_predicted = regressor.predict([[27]])
y_pred = regressor.predict(X_test)
#print(y_predicted)

#evaluate
#rmse = mean_squared_error(y_train, y_predicted)
#r2 = r2_score(y_train, y_predicted)

lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)

lin_mae = mean_absolute_error(y_pred, y_test)

print('Slope: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)
print('Liner Regression R squared: %.4f' % regressor.score(X_test, y_test))
print('Liner Regression RMSE: %.4f' % lin_rmse)
print('Liner Regression MAE: %.4f' % lin_mae)