############################################################# 
## Stat 202A - Homework 6
## Author: Linfan Zhang
## Date : 11/14/2017
## Description: This script implements ridge regression as 
## well as piecewise linear spline regression.
#############################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
#############################################################

## Source your Rcpp file (put in the name of your 
## Rcpp file)
library(Rcpp)
sourceCpp("Sweep.cpp")

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
      n <- nrow(A)
      m <- ncol(A)
      R <- A
      Q <- diag(n)
      for(k in 1:(m-1)){
            x <- rep(0, n)
            x[k:n] = R[k:n, k]
            v <- x
            my_sign <- sign(x[k])
            v[k] <- x[k] + sign(x[k]) * norm(matrix(x), "F")
            s <- norm(matrix(v), "F")
            u <- v/s
            R <- R - 2*u %*% (t(u) %*% R)
            Q <- Q - 2*u %*% (t(u) %*% Q)
      }  
  
  
  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R
  return(list("Q" = t(Q), "R" = R))
  
}


#################################
## Function 2: Sweep operation ##
#################################

mySweep <- function(A, m){
  
  # Perform a SWEEP operation on A with the pivot element A[m,m].
  # 
  # A: a square matrix.
  # m: the pivot element is A[m, m].
  # Returns a swept matrix.
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################
      n <- dim(A)[1]
      
      for(k in 1:m){
            for(i in 1:n){
                  for(j in 1:n){
                        if (i != k & j != k){
                              A[i,j] <- A[i,j] - A[i,k]*A[k,j]/A[k,k]
                        }
                  }
            }
            
            for(j in 1:n){
                  if( j != k){
                        A[k,j] <- A[k,j]/A[k,k]
                  }
            }
            
            for(i in 1:n){
                  if(i != k){
                        A[i,k] <- A[i,k]/A[k,k]
                  }
            }
            
            A[k,k] <- -1/A[k,k]
      }
      
  
  return(A)
  
}


##################################
## Function 3: Ridge regression ##
##################################

myRidge <- function(X, Y, lambda){
  
  # Perform ridge regression of Y on X.
  # 
  # X: an n x p matrix of explanatory variables.
  # Y: an n vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # Returns beta, the ridge regression solution.

  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################
      Y <- matrix(Y)
      n <-  dim(X)[1]
      p <-  dim(X)[2]
      
      # prepare for the QR matrix
      D <-  diag(rep(sqrt(lambda), p), nrow = p)
      X1 <- rbind(X, D)
      X1 <- cbind(c(rep(1, n), rep(0, p)), X1)
      Y1 <- rbind(Y, matrix(rep(0, p), ncol = 1))
      Z <- cbind(X1, Y1)
      R <-  myQRC(Z)[[2]]
      R1 <- R[1:(p+1), 1:(p+1)]
      Y1 <- R[1:(p+1), p+2]
      beta_ridge <- solve(R1) %*% Y1
  
  ## Function should output the vector beta_ridge, the 
  ## solution to the ridge regression problem. beta_ridge
  ## should have p + 1 elements.
  return(beta_ridge)
  
}


####################################################
## Function 4: Piecewise linear spline regression ##
####################################################


mySpline <- function(x, Y, lambda, p = 100){
  
  # Perform spline regression of Y on X.
  # 
  # x: An n x 1 vector or matrix of explanatory variables.
  # Y: An n x 1 vector of dependent variables. Y can also be a 
  # matrix, as long as the function works.
  # lambda: regularization parameter (lambda >= 0)
  # p: Number of cuts to make to the x-axis.

  ##################################
  ## FILL IN THIS SECTION OF CODE ##
  ##################################
      n <- length(x)
      x.order <- order(x)
      x <- x[x.order]
      
      Y <- matrix(Y, nrow = n)
      for(i in 1:ncol(Y))
            Y[ ,i] <- Y[x.order, i]
      
      X <- matrix(x, nrow = n)
      for (k in (1:(p-1))/p*(x[n] -x[1]))
            X <- cbind(X, (x>k)*(x-k))
      # myRidge function will add a one column to X
      beta_spline <- myRidge(X, Y, lambda)
      Yhat <- cbind(rep(1, n), X)%*%beta_spline
      

  ## Function should a list containing two elements:
  ## The first element of the list is the spline regression
  ## beta vector, which should be p + 1 dimensional (here, 
  ## p is the number of cuts we made to the x-axis).
  ## The second element is y.hat, the predicted Y values
  ## using the spline regression beta vector. This 
  ## can be a numeric vector or matrix.
  output <- list(beta_spline = beta_spline, predicted_y = Yhat)
  return(output)
  
}


