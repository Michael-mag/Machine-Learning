#!/usr/bin/env python3
"""
    Python Implementation of regularized logistic regression from Andrew Ng's
    Course Era Machine learning course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#no headers and use column numbers as column names instead
df = pd.read_csv("ex2data2.txt", header = None) #make a pandas data frame

#use print to render the data on the terminal, since I'm not using jupiter NB
print("\n Here is the first 10 lines of the data set : \n",df.head(10)) #return the first 10 lines of the data for a quick overview
#some useful information about the data
print("\n Here is a useful summary of the data : \n ",df.describe())
print()

#set the x-axis and y-axis values
#use iloc, integer-location based indexing, in the order that they appear
X= df.iloc[:,:-1].values
y = df.iloc[:,-1].values

plt.figure(1)
pos , neg = (y==1).reshape(118,1) , (y==0).reshape(118,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend(["Accepted","Rejected"],loc=0)
plt.title("Scatter Plot of the Exams data")


########## FEATURE MAPPING  ####################################################
def mapFeature(x1,x2):
    """
        - This function maps two input features to quadratic features used in the
            regularization process
        - returns all polynomial terms upto the given degree, of the numpy arrays
            x1 and x2
        - x1 and x2 should be the same size
    """

    degree = 6
    output = np.ones(len(x1)).reshape(len(x1),1)
    #Note the +1 since Matlab indexing starts from 1 while python starts from 0
    for i in range(1,degree + 1):
        for j in range(i+1):
            terms = (pow(x1,(i-j)) * pow(x2,j)).reshape(len(x1),1)
            output = np.hstack((output,terms))
    return output

X = mapFeature(X[:,0], X[:,1])



######### REGULARIZED COST FUNCTION AND GRADIENT DESCENT #######################
class CostFunction(object):
    """
        - This class is used to compute the cost and gradient descent of the
            logistic regression with regularization.
    """

    def __init__(self):
        print("\n\nInitializing fitting parameters and Lambda :... \n")
        self.initial_theta = np.zeros((X.shape[1],1))
        self.Lambda = 1

    # SIGMOID FUNCTION
    def sigmoid(self,z):
        """
            This function computes the sigmoid of z
        """
        return (1/(1 + np.exp(-z)))

    # COMPUTE THE COST FUNCTION
    def CostFunctionReg(self,theta, X, y, Lambda):
        """
            Take the numpy array of theta, X, y and Lambda and compute the cost
            of using theta as the parameter for regularized logistic regression
            and the gredient of the cost w.r.t the parameters.
        """

        m = len(y) #the number of training exmples

        #convert y from 1 to 2 dimensional array,i.e a column vector in this case
        y=y[:,np.newaxis]
        z = X @ theta #multiplying the 2D arrays X and theta
        h = self.sigmoid(z) # where h is the hypothesis
        theta = theta[1:] #first value of theta should not be regularized
        R = (Lambda/(2*m))*sum(pow(theta,2)) #the regularization

        #the regularized cost function
        J = (-(1/m))*sum((y*np.log(h)) + ((1-y)*np.log(1-h))) + R

        #the matlab for loop above will be replaced  by pandas operations for efficiency
        XT = X.transpose()
        Jth0 = (1/m)*(XT @ (h-y))[0] #derivitive of J w.r.t theta_O
        Jth = (1/m)*(XT @ (h-y))[1:] + (Lambda/m)*theta
        grad = np.vstack((Jth0[:,np.newaxis],Jth)) #define new axis to allow for vertical stacking

        return J[0], grad

    # Compute initial cost and gradient
    def initial(self):
        cost, grad = self.CostFunctionReg(self.initial_theta, X, y, self.Lambda)
        print("\nINITIAL COST IS : \n",cost)

    ############## GRADIENT DESCENT ###########################
    def gradientDescent(self,X,y,theta,alpha,num_iters,Lambda):
        """
        Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
        with learning rate of alpha

        return theta and the list of the cost of theta during each iteration
        """

        m=len(y)
        J_history =[]

        for i in range(num_iters):
            cost, grad = self.CostFunctionReg(theta,X,y,Lambda)
            theta = theta - (alpha * grad)
            J_history.append(cost)
        return theta , J_history

    def plotCost(self):
        self.theta , J_history = self.gradientDescent(X,y,self.initial_theta,1,800,0.2)
        print("The regularized theta :\n",self.theta)

        # Cost function plot
        plt.figure(2)
        plt.plot(J_history)
        plt.xlabel("Iteration")
        plt.ylabel("$J(\Theta)$")
        plt.title("Cost function using Gradient Descent")

    # Map feature plot
    def mapFeaturePlot(self,x1,x2,degree):
        """
            take in numpy array of x1 and x2, return all polynomial terms up to the given degree
        """
        out = np.ones(1)
        for i in range(1,degree+1):
            for j in range(i+1):
                terms= (x1**(i-j) * x2**j)
                out= np.hstack((out,terms))
        return out




    # Plotting decision boundary

    def plotBoundary(self):
        u_vals = np.linspace(-1,1.5,50)
        v_vals= np.linspace(-1,1.5,50)
        z=np.zeros((len(u_vals),len(v_vals)))
        for i in range(len(u_vals)):
            for j in range(len(v_vals)):
                z[i,j] =self.mapFeaturePlot(u_vals[i],v_vals[j],6) @ self.theta

        plt.figure(3)
        plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="r",marker="+",label="Admitted")
        plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c="b",marker="x",label="Not admitted")
        plt.contour(u_vals,v_vals,z.T,0)
        plt.xlabel("Exam 1 score")
        plt.ylabel("Exam 2 score")
        plt.title("Descision boundary")
        plt.legend(["Accepted","Rejected"],loc=0)


if __name__ == '__main__':
    cost = CostFunction()
    cost.initial()
    cost.plotCost()
    cost.plotBoundary()
    #cost.accurracy()
    plt.show()
