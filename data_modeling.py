
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
print(now)

df = pd.read_csv('eda_data.csv')

# choose relevant columns
# print(df.columns)
df_model = df[['average_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_competitors',
               'job_state', 'job_town', 'age', 'sql_yn', 'python_yn', 'excel_yn', 'ML_yn', 'aws_yn', 'spark_yn', 'job_simple', 
               'seniority', 'desc_length']]

# create dummy variables for categorical data
df_full = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

x = df_full.drop('average_salary', axis = 1)
y = df_full.average_salary.values

# Note: df.attribute.values creates an array of that attribute while df.attribute creates a series with indices for all of the values of that attribute
#       The array is generally more recommended for models

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Model types
#    1. multiple linear regression

### SM Model ###
import statsmodels.api as sm

x_sm = x = sm.add_constant(x)
sm_model = sm.OLS(y,x_sm)
sm_model.fit().summary()

### SK Learn Model ###
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score # cross validation

lm = LinearRegression()
lm.fit(x_train, y_train)

print("Linear Fit = " + str(np.mean(cross_val_score(lm, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))))

now = datetime.now()
print(now)

#    2. lasso regression

lm_L = Lasso()
print("Lasso = " + str(np.mean(cross_val_score(lm_L, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))))

now = datetime.now()
print(now)

alpha = []
error = []

alpha_prime = 0.0
min_error = -20.0

for i in range(1,100):
    alpha_num = i/100
    alpha.append(alpha_num)
    lm_L_loop = Lasso(alpha = alpha_num)
    temp_err = np.mean(cross_val_score(lm_L_loop, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))
    error.append(temp_err)
    ### Ken, this is how you find Min Error with less work
    if min_error < temp_err:
        min_error = temp_err
        alpha_prime = alpha_num

plt.plot(alpha, error)
print("Alpha Prime = " + str(alpha_prime) + "\nMin Error = " + str(min_error))
plt.show()

now = datetime.now()
print(now)

#    3. random forest -- don't have to worry about multicollinearity with this type of model??? reasoning behind this?
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

print("Random Forest = " + str(np.mean(cross_val_score(rf, x_train, y_train, scoring='neg_mean_absolute_error', cv=3))))


"""
# Other Model Types:
# gradient boosted tree
# support vector regression
"""

now = datetime.now()
print(now)

# tunes models using gridsearchCV
from sklearn.model_selection import GridSearchCV

# parameters = {'n_estimators': range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto', 'sqrt', 'log2' )}
parameters = {'n_estimators': range(119,120,1), 'criterion':('mse','mae'), 'max_features':('auto', 'sqrt', 'log2' )}


gs = GridSearchCV(rf, parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(x_train,y_train)

now = datetime.now()
print(now)


print("Grid Search Random Forest Error = " + str(gs.best_score_)) # output: -11.956382809520369
print("Grid Search Random Forest Parameters = " + str(gs.best_estimator_)) # output: RandomForestRegressor(n_estimators=119, mse, auto)



# test ensembles

lm_L = Lasso(alpha = 0.02)
lm_L.fit(x_train, y_train)

rf = RandomForestRegressor(n_estimators=119, criterion='mse', max_features='auto')

tpred_lm = lm.predict(x_test)
tpred_lm_L = lm_L.predict(x_test)
tpred_rf = gs.best_estimator_.predict(x_test)

# comparing predictions to actual salary estimates
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, tpred_lm))
print(mean_absolute_error(y_test, tpred_lm_L))
print(mean_absolute_error(y_test, tpred_rf))