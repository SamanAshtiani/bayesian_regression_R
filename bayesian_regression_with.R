setwd("~/git/bayesian_regression_R/")

#warm-up on non-bayesian regression
data(women)
force(women)
x <- women$weight
y <- women$height

lin.model <- glm(y~x, data=women)
plot(x,y)
summary(lin.model)

abline(lin.model,
       col="red")

# one drawback of linear models is they don't define a limit or threshold for the response variable, 
#eg how much loosing weight is
#healtht with respect to the explanatory variable x
#the solution is:
#Generalized additive models:
install.packages("mgcv")
library(mgcv)
#press F1 ~ ?gam
spline.formula <- as.formula(y~s(x)) #s is the spline that searches the degree of the polynomial the fits best 
gam.model <- gam(spline.formula, data = women)
gam.model$converged

plot(x,y)
# abline(gam.model, col="pink")  #doesn't work for a polynomial model
plot(gam.model,
     col="pink") #we can also see credible intervals

# if we define the distribution of the height of women with mu as a parameter instead of explicitly specifying a mu,
#and say the sigma is eg 15, we have "parametric bayesian analysis"
#here we can model mu as a*weight + b , which is a single variable model 
#our job will be to find the best inferences of parameters a and b

# Start Bayesian ----

install.packages("rstanarm")
library(rstanarm)
# glm uses maximum likelihood to estimate the parameters ie slope and intercept
# stan_glm uses MCMC or HamiltonianMC to estimate the model parameters
model_A <- stan_glm(formula = height ~ weight,
                    data = women,
                    algorithm = "sampling")
summary(model_A)
# the Bayesian inference gave us the estimates with "mean" and there is no p-value. 
# Estimates:
#   mean   sd   10%   50%   90%
# (Intercept) 25.7    1.2 24.2  25.7  27.3 
# weight       0.3    0.0  0.3   0.3   0.3 
# sigma        0.5    0.1  0.4   0.5   0.6 

# compare with non-bayesian estimates:
model_B <- glm(height ~ weight, 
    data=women)
summary(model_B)

# as evident, no significant difference bet two methods for parameter inference. 
# in statistical inference we have point estimate followed by hypothesis testing. 
##point estimate in Bayesian inference is changed into "credible intervals" 
# the estimated parameters using Bayes'  are defined with an uncertainty:
# we say a `belongs to ` [alpha1, alpha2] interval, and this interval is based on
#the posterior 
#the wider the interval, the more uncertainty about the inferred parameter,ie a.
# we say there's a certain "probability" that the true value of a falls within 
#the interval ie credible interval
# *** in credible interval, the parameter is a random variable and the interval is fixed
#, however,
#in confidence interval of frequentist statistics, the parameter is fixed but the 
#interval is random, ie we assume a probability for the upper and lower limits of the
#interval

#to compute Bayesian posterior uncertainty interval:
posterior_interval(model_A,
                   prob = 0.9 #prob that unseen parameter falls in the interval
                   # , pars= "weight")
) # if you don't specify pars, it will give all by default
# prob: Unlike for a frenquentist confidence interval, it is valid to say that, 
#conditional on the data and model, we believe that with probability 
#p the value of a parameter is in its 100p% "posterior interval". This intuitive
#interpretation of Bayesian intervals is often erroneously applied to frequentist
#confidence intervals. See Morey et al. (2015) for more details on this issue 
#and the advantages of using Bayesian posterior uncertainty intervals
#(also known as credible intervals).

# compare the frequentist CI, with bayesian credible intervals:
confint(model_B,
        parm = "weight",
        level = 0.95)


# to get the prior of the model:
prior_summary(model_A)
# If we assume height(response) has normal distribution with mu and sigma, we can 
#define mu as y=b+a1*x1+a2*x2+ ... + ak*xk and in our example ak has a normal distribution
#with location(mu)=0 and adjusted sigma (based on adjusted prior):
#specified prior defined by rstanarm * sd(y)/sd(x) = 2.5 * sd(height)/sd(weight) = 0.72 as in the output

#sigma has an exponential distribution and with one parameter ie 1, so the adjusted prior
# for sigma will be : 1/sd(height)= 0.22 
#for the intercept:
2.5* sd(women$height) # =11.18 as seen in the output intercept 

# visualizing priors
model_A <- stan_glm(height ~ weight,
                    data = women,
                    algorithm = "sampling",
                    prior = normal(4,1),
                    prior_intercept = normal(1,15),
                    prior_aux = exponential(rate = 2)
                    )
prior_summary(model_A)

posterior_vs_prior(model_A,
                   pars = "alpha") #intercept and beta means slope

# you can set non-informative priors as well
model_A <- stan_glm(height ~ weight,
                    data = women,
                    algorithm = "sampling",
                    prior = NULL,
                    prior_intercept = NULL,
                    prior_aux = exponential(rate = 2)
)

posterior_vs_prior(model_A,
                   pars = "alpha")

plot(model_A, "hist")

#posterior predictive checks
women_posterior <- posterior_predict(model_A,
                                     draws=600)
class(women_posterior) #matrix array
dim(women_posterior) #600 15
dim(women)  #15 2
women$height
#[1] 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72
women_posterior[1, ]
# 1        2        3        4        5        6        7        8        9       
# 59.15931 59.65735 60.87717 61.24191 61.89401 63.18145 63.22916 65.32678 66.18069 
# 
# 10       11       12       13       14       15 
# 67.00500 67.70706 68.79224 69.39023 72.01883 73.13302 

# Bernoulli probability distribution likelihood function:

likelihood = function(n,y,theta) {return(theta^y*(1-theta)^(n-y))}
theta_set <- seq(0.01, 0.99, by=0.01)
theta_set
plot(theta_set, likelihood(400, 74,theta_set))
#we can see likelihood is maximized around .18
abline(v=0.18, col="pink")
#compliance with frequentist:
74/400 #=0.18

#logLikelihood:
loglike= function(n, y, theta) {return(y*log(theta) + (n-y)*log(1-theta)  )}
plot(theta_set,loglike(400,72, theta_set))
qnorm(0.5, 0,1) #50% quantile ie returns x value for which P(X<x) = 0.5
pnorm(0, 0, 1) #cumulative distribution function 
dnorm(0,0,1)


# Beta binomial analysis and distribution in R: Coursera ----
#suppose two students are given 40 multiple choice questions. we dont know how much they've studie but we know they'll do
#btter than just answering randomly. 
# 1) what are the parameters of interset?
# theta1 : true prob that the first student answers a question correctly
#theta2: true prob that the second student answers a question correctly


# 2) what is our likelihood?
# 3) what prior should we use?
# 4) what is the prior probability? P(theta>0.25)  P(theta>0.5) P(theta>0.8)
# 5) suppose the first student gets 33 questions right. what is the posterior distribution for theta1?
#   what is a 95% posterior credible interval for theta1?
# 6) suppose the second student gets 24 questions right. what is the posterior distribution for theta2?
#    what is the 95% posterior credible interval for theta2?
# 7) what is the posterior probability that theta1 > theta2, ie the first student has a better chance to get
#     the questions right than the second student
#
#






