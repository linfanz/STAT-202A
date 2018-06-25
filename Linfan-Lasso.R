############################################################# 
## Stat 202A - Homework 7
## Author: Linfan Zhang
## Date : 11/23/2017
## Description: This script implements the lasso
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

#####################################
## Function 1: Lasso solution path ##
#####################################

myLasso <- function(X, Y, lambda_all){
  
  # Find the lasso solution path for various values of 
  # the regularization parameter lambda.
  # 
  # X: n x p matrix of explanatory variables.
  # Y: n dimensional response vector
  # lambda_all: Vector of regularization parameters. Make sure 
  # to sort lambda_all in decreasing order for efficiency.
  #
  # Returns a matrix containing the lasso solution vector 
  # beta for each regularization parameter.
 
  #######################
  ## FILL IN CODE HERE ##
  #######################
  

      B <- 50
      n <- nrow(X)
      R <- Y
      X <- cbind(rep(1, n), X)
      p <- ncol(X)
      SS <- apply(X, 2, function(x) return(sum(x^2)))
      
      # Sort lambda_all
      lambda_all <- sort(lambda_all, decreasing = T)
      L <- length(lambda_all)
      
      # Initialize beta
      beta_all <- matrix(rep(0,p*L), nrow = p)
      
      # Solution path
      for(l in 1:L){
            lambda <- lambda_all[l]
            beta <- beta_all[,l]
            R <- Y - X %*% beta
            for(b in 1:B){
                  for(k in 1:p){
                        # partial residuals
                        R <- R + X[,k]*beta[k]
                        
                        # soft-threshold solution
                        xr <- sum(X[,k]*R)
                        
                        if(k == 1){
                              beta[k] <- abs(xr)/SS[k]
                              beta[k] <- sign(xr)*max(beta[k],0)
                        }else{
                              beta[k] <- (abs(xr)-lambda/2)/SS[k]
                              beta[k] <- sign(xr)*max(beta[k],0)
                        }
                        
                        
                        # residuals
                        R <- R - X[ ,k]*beta[k]
                  }
            }
            beta_all[,l] <- beta
      }
  
  ## Function should output the matrix beta_all, the 
  ## solution to the lasso regression problem for all
  ## the regularization parameters. 
  ## beta_all is (p+1) x length(lambda_all)
  return(beta_all)
  
}

# # Test
# n = 50
# p = 25
# s = 5
# 
# 
# X = matrix(rnorm(n*p), nrow = n)
# beta_true = matrix(rep(0, p), nrow = p)
# beta_true[1:s] = 1:s
# Y = X %*% beta_true + rnorm(n)
# 
# lambda_all  = (100:1)*10
# L = length(lambda_all)
# beta_all <- myLasso(X, Y, lambda_all)
# matplot(t(matrix(rep(1,p+1), nrow = 1) %*% abs(beta_all)),
#         t(beta_all), type = "l")
# text(15.2,beta_all[,100],1:(p+1),cex = 0.8, col=1:(p+1))


