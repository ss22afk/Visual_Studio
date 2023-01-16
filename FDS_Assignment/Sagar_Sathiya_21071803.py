# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Name: Sagar Sathiya
StudentID: 21071803
"""
# Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Load File using numpy into numpy array
np_input3 = np.loadtxt('inputdata3.csv', delimiter=',',dtype='str', skiprows=1)
print(np_input3)

# Convert an array value from String to integer
rainfall_arr =  np.asarray(np_input3[...,0],dtype="int")
print('Rainfall Ar00ray: ', rainfall_arr)

# Convert an array from String to Float
productivity_arr = np.asarray(np_input3[...,1],dtype="float")
print('Productivity Array: ', productivity_arr)

'''
Linear regression on observation of the amount of rain per year and field productivity in dry area at Central america  
'''
# Create x and y Variable and assign a rainfall array and Productivity array 
x = rainfall_arr
y = productivity_arr

# Reshape x array
x = x.reshape(-1,1)

#train linear Regration model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print(X_train)
print(y_train)

# Instance of class LinearRegression for regression model
model = LinearRegression()

# fits the model
model.fit(X_train, y_train)

print('Cofficient of Productivity: ', model.coef_) #Cofficient of Productivity:  [0.00038018]
print('intercept  of Productivity:', model.intercept_) #intercept  of Productivity: 0.006136743441194484

# X mm Value for Productivity prediction
score = model.predict([[350.0]])
print('Score: ', score) #Score:  [0.13919972]

# Regression line
y_pred = model.predict(x)

#Procuctivity priction using train set
productiVity_pred = model.predict(X_test)

prod_arr = np.asarray(productiVity_pred)
print('Array prod:',prod_arr)
print('Productivity Prediction: ', productiVity_pred) 
''' Productivity Prediction: [0.027807   0.03503042 0.0354106  0.03997276 0.04529528 0.05175833
    0.05632049 0.05936193 0.06392409 0.06544481 0.06962679 0.07570967
    0.07989165 0.08673489 0.08635471 0.09433849 0.09699975 0.10270244
    0.10574388 0.10954568 0.11600874 0.1167691  0.12095108 0.1262736
    0.13121594 0.13501774 0.13881954 0.14566278]'''
'''
2D Scatter plot of rainfall per year and productivity in dry area at Central America with regression line.
'''
plt.Figure(figsize=(6,6))
plt.plot(x,y_pred, color='black', label='Regression line')
plt.scatter(x,y, color='blue')
plt.xlabel('Rainfall')
plt.ylabel('Productivity')
plt.title('Rainfall and Productivity plot with regression line')
plt.legend()
plt.show()
