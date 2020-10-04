import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_excel('~/Downloads/ANZ/cleaned.xlsx', index='customer_id')

data.drop_duplicates(['customer_id'], keep='first', inplace=True)
data['annual_salary'] = data['amount'] * 52
salary = data[data['txn_description'] =='PAY/SALARY']

age_salary = salary[['age', 'annual_salary']]
X = age_salary.iloc[:, :-1].values
y = age_salary.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#y_pred = regressor.predict([[27]])
y_pred = regressor.predict(X_test)

#print(y_pred)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Annual Salary vs Customer Age')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

lin_mse = mean_squared_error(y_pred, y_test)
lin_rmse = np.sqrt(lin_mse)

print('Liner Regression RMSE: %.4f' % lin_rmse)
