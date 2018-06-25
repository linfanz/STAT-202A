#########################################################
## Stat 202A - Homework 3
## Author: Linfan Zhang 405036365
## Date : 10/25/2017
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################

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
        #if(my_sign == 0)
        #      my_sign <- 1
        v[k] <- x[k] + my_sign * norm(matrix(x), "F")
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

###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  n <- nrow(X)
  p <- ncol(X)
  Z <- cbind(rep(1, n), X, Y)
  R <- myQR(Z)[[2]]
  R1 <- R[1:(p+1), 1:(p+1)]
  Y1 <- R[1:(p+1), p+2]
  beta_ls <- solve(R1) %*% Y1
  beta_ls <- t(beta_ls)
  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(beta_ls)
  
}

##################################
## Function 3: PCA based on QR  ##
##################################

myEigen_QR <- function(A, numIter = 1000){
  
  ## Perform PCA on matrix A using your QR function, myQRC.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  
  ########################
  ## FILL IN CODE BELOW ##
  ######################## 
  r <- nrow(A)
  V <- matrix(rnorm(r*r), nrow = r)
  
  for(i in 1:numIter){
        Q_R <- myQR(V)
        Q <- Q_R[[1]]
        R <- Q_R[[2]]
        V <- A %*% Q
  }
  
  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  return(list("D" = diag(R), "V" = Q))
}


