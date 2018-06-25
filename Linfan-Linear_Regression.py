# -*- coding: utf-8 -*-
"""

 Stat 202A - Homework 1
 Author: Linfan Zhang
 Date : 10/07/2017
 Description: This script implements linear regression 
 using Gauss-Jordan elimination in both plain and
 vectorized forms

 INSTRUCTIONS: Please fill in the missing lines of code
 only where specified. Do not change function names, 
 function inputs or outputs. You can add examples at the
 end of the script (in the "Optional examples" section) to 
 double-check your work, but MAKE SURE TO COMMENT OUT ALL 
 OF YOUR EXAMPLES BEFORE SUBMITTING.

 Do not use any of Python's built in functions for matrix 
 inversion or for linear modeling (except for debugging or 
 in the optional examples section).
 
"""
import numpy as np
from sklearn import datasets, linear_model
###############################################
## Function 1: Plain version of Gauss Jordan ##
###############################################


def myGaussJordan(A, m):
  
  """
  Perform Gauss Jordan elimination on A.
  
  A: a square matrix.
  m: the pivot element is A[m, m].
  Returns a matrix with the identity matrix 
  on the left and the inverse of A on the right. 

  FILL IN THE BODY OF THIS FUNCTION BELOW 
  """
  n = A.shape[0]
  # same as cbind in R, but there are double brackets
  B = np.hstack((A, np.identity(n)))
  for k in range(m):
    a = B[k, k]
    for j in range(n * 2):
      B[k, j] = B[k, j] / a
    for i in range(n):
      if i != k:
        b = B[i, k]
        for j in range(n * 2):
          B[i, j] = B[i, j] - B[k, j] * b

  ## Function returns the np.array B
  return B
  


####################################################
## Function 2: Vectorized version of Gauss Jordan ##
####################################################

def myGaussJordanVec(A, m):
  
  """
  Perform Gauss Jordan elimination on A.
  
  A: a square matrix.
  m: the pivot element is A[m, m].
  Returns a matrix with the identity matrix 
  on the left and the inverse of A on the right.
  
  FILL IN THE BODY OF THIS FUNCTION BELOW
  """
  n = A.shape[0]
  B = np.hstack((A, np.identity(n)))
  for k in range(m):
    a = B[k, k]
    B[k, :] = B[k, :] / B[k, k]
    for i in range(n):
      if i != k:
        b = B[i, k]
        B[i, :] = B[i, :] - B[k, :] * B[i, k]
  return B

  ## Function returns the np.array B
  return B
  




######################################################
## Function 3: Linear regression using Gauss Jordan ##
######################################################

def myLinearRegression(X, Y):
  
  """
  Find the regression coefficient estimates beta_hat
  corresponding to the model Y = X * beta + epsilon
  Your code must use one of the 2 Gauss Jordan 
  functions you wrote above (either one is fine).
  Note: we do not know what beta is. We are only 
  given a matrix X and a vector Y and we must come 
  up with an estimate beta_hat.
  
  X: an 'n row' by 'p column' matrix (np.array) of input variables.
  Y: an n-dimensional vector (np.array) of responses

  FILL IN THE BODY OF THIS FUNCTION BELOW
  """
  
  ## Let me start things off for you...
  n = X.shape[0]
  p = X.shape[1]

  Z = np.hstack((np.repeat(1, n).reshape(n,1), X, Y))
  A = np.dot(np.transpose(Z),Z)
  S = myGaussJordan(A, p + 1)
  beta_hat = S[range(p + 1), (p + 1)]

  ## Function returns the (p+1)-dimensional vector (np.array) 
  ## beta_hat of regression coefficient estimates
  return beta_hat
  


########################################################
## Optional examples (comment out before submitting!) ##
########################################################

#def testing_Linear_Regression():
  
  ## This function is not graded; you can use it to
  ## test out the 'myLinearRegression' function 

  ## You can set up a similar test function as was 
  ## provided to you in the R file.
  #n = 100
  #p = 3

  #X = np.reshape(np.random.normal(0, 1, n * p), (n, p))
  #beta = np.array(range(1, p + 1))
  #Y = (np.dot(X, beta) + np.random.normal(0, 1, n)).reshape(n, 1)

  #regr = linear_model.LinearRegression()
  #regr.fit(X, Y)

  #my_coef = myLinearRegression(X, Y)
  #p_coef = np.hstack((regr.intercept_.reshape(1, 1), regr.coef_))

  #sum_square_diff = np.sum(np.multiply((p_coef - my_coef), (p_coef - my_coef)))
  #if sum_square_diff <= 0.001:
    #print('Both results are identical')
  #else:
    #print('There seems to be a problem...')

#testing_Linear_Regression()


