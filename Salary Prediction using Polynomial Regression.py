#importing libraries

import pandas as pd

#loading dataset

dataset = pd.read_csv('dataset.csv')

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#segregating dataset into Input X and Output Y

X = dataset.iloc[:,:-1].values
X
Y = dataset.iloc[:,-1].values

#Converting X to Polynomial Format (X^n)

from sklearn.preprocessing import PolynomialFeatures
modelPR = PolynomialFeatures(degree = 4)
XPoly = modelPR.fit_transform(X)

#training with Linear Regression using X-polynomial insted of X

from sklearn.linear_model import LinearRegression
modelPLR = LinearRegression()
modelPLR.fit(XPoly,Y)

#Visualizing Polynomial Regression results

import matplotlib.pyplot as plt
plt.scatter(X,Y, color="red")
plt.plot(X, modelPLR.predict(modelPR.fit_transform(X)))
plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Prediction using Polynomial Regression

x = float(input("Enter the Level of the Person: "))
salaryPred = modelPLR.predict(modelPR.fit_transform([[x]]))
print('Salary of a person with Level {0} is {1}'.format(x,salaryPred))