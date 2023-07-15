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

# Graphically compare the actual response values with the draws from posterior
color_scheme_set("brightblue")
library(bayesplot)
ppc_dens_overlay(y, yrep = women_posterior[1:30,])

# quadratic and cubic bayesian posterior predictive check ----
model_Q <- stan_glm(formula = height ~ weight + I(weight^2),
         data = women)

print(model_Q)
summary(model_Q) # Rhat 1 shows that model has converged
women_posterior_Q <- posterior_predict(model_Q,
                                       draws= 600)
dim(women_posterior_Q) # 600  15

ppc_dens_overlay(y, yrep = women_posterior_Q[1:30,])

#cubic model
model_cubic <- stan_glm(formula = height ~ weight + I(weight^2) + I(weight^3),
                    data = women)
women_posterior_cubic <- posterior_predict(model_cubic,
                                           draws=600)
ppc_dens_overlay(y, yrep = women_posterior_cubic[1:30,])

# Hostogram of the posterior predictive distribution ----
# model_A
ppc_hist(y, yrep = women_posterior[1:30,])
ppc_hist(y, yrep = women_posterior_Q[1:30,])

# pp checks with boxplots:
#linear model:
pp_check(model_A,
         plotfun = "boxplot",
         nreps=10,
         notch=FALSE)
#quadratic model:
pp_check(model_Q,
         plotfun = "boxplot",
         nreps=10,
         notch=FALSE)

# MCMC intervals, single variable ----
#we make mcmc plots to see that for each draw from the posterior distribution, which produces all the vectors of
#prediction of our response variables to visualize the uncertainty intervals, credible intervals and the mean of those
#draws.
#thicker line is 50% uncertainty and thinner is 90% uncertainty interval 
#(can be changed in the mcmc_intervals function arguments)
mcmc_intervals(model_A,
               pars = "weight")
mcmc_intervals(model_Q,
               pars = "weight")

# multivariable regression analysis in Bayesian framework ----
data(state)
dim(state.x77)
head(state.x77)
states <- state.x77[, c("Murder","Population", "Income", "Illiteracy", "Area")]
class(states)
states <- as.data.frame(states)
class(states)
#create multivariate regression model
model_A_multi <- stan_glm(formula = Murder~Population + Income + Illiteracy + Area, 
         data = states,
         algorithm = "sampling") 
summary(model_A_multi)
prior_summary(model_A_multi)

# MCMC intervals for multivariable bayesian regresson model ---- 
#to find the credible intervals, median and mean of the response variables 
#x axis is the slpe and three covariates have 0 slope, so can be omitted.
mcmc_intervals(model_A_multi,
               pars = c("Population", "Income", "Illiteracy", "Area")
               )
#credible interval 
posterior_interval(model_A_multi,
                   prob = 0.5,  # 50% inteval
                   pars = "Illiteracy")


# interaction terms in linear regression ----
install.packages("effects")
library(effects)
#remove.packages("effects")
data(mtcars)
model_no_inter <- glm(mpg ~ hp + wt, data=mtcars)
class(model_no_inter)
summary(model_no_inter)
model_inter <- glm(mpg ~ hp + wt + (hp:wt), 
                      data=mtcars)
class(model_inter)
summary(model_inter)

# wt:hp interactions visualization
mean(mtcars$wt) #3.12
sd(mtcars$wt) #0.97
effect("hp:wt", model_inter, wt=c(3.21,2.21,4.21), vcov.=vcov)
plot(effect("hp:wt", model_inter, list(wt=c(3.21,2.21,4.21)), vcov.=vcov),
     multiline = TRUE)

# Bernoulli probability distribution likelihood function ----

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
# Binomila(40, theta) if we assume that each question is independent and that probability a student gets a question right
#is the same for all questions for that student

# 3) what prior should we use?
# the conjugate prior is a beta prior. plot the density with dbeta
theta = seq(0,1, by=0.01)
#theta
#plot the default prior: beta(1,1) which is a uniform
plot(theta,dbeta(theta,1,1), type="l") #all thetas are equally likely but it does not reflect our initial guess of >0.25
#plot a beta distribution that has prior mean 2/3
plot(theta, dbeta(theta, 4,2), type="l") # we still have some mass below 0.25
# as we increase the parameter values, it increases the effect of sample sizes and concentrates the distribution
plot(theta, dbeta(theta, 8,4), type="l") # a reasonable prior for our problem. majority of the mass is between 0.25 and 1

# 4) what is the prior probabilities that the parameter s be > ...? P(theta>0.25)  P(theta>0.5) P(theta>0.8)
#pbeta gives us the cumulative distribution function for the beta distri equal to less than that value
#so we calculate 1-...
#prior probabilities:
1-pbeta(0.25, 8,4)
0.99
1-pbeta(0.5, 8,4)
0.88
1-pbeta(0.8, 8,4)
0.16

# 5) suppose the first student gets 33 questions right. what is the posterior distribution for theta1?
#   what is a 95% posterior credible interval for theta1?
beta(8+33, 4+40-33) # = beta(41,11)
#posterior mean
41/(41+11) #0.7
#MLE
33/40  #0.82
plot(theta, dbeta(theta,41,11), type="l") #posterior, as we get more info the distri becomes more concentrated
lines(theta,dbeta(theta, 8,4), lty=2) #prior
#so the posterior mean is between MLE and prior of 2/3=0.66
#let's add likelihood as well:
lines(theta, dbinom(33,size=40, p=theta), lty=3) # on a different scale, we need to rescale it coz it does not have
# a normalizing constant to make it a density
lines(theta, 44*dbinom(33,size=40, p=theta), lty=3)
#posterior(solid) and likelihood are close coz there's more info in the likelihood. we have 40 samples in the likelihood
#and a prior with effective sample size of eight + 4 =12 so it makes sense that posterior be closer to the likelihood than
#to the prior

#posterior probabilities that theta1 > 1/4 
1-pbeta(0.25,41,11)
1-pbeta(0.5, 41,11)
1-pbeta(0.8,41,11)  # 0.444 given that ??>> our data had a value >0.8   <<??now we are more confident that theta1 is a 
#larger value compared with prior prob. of 0.16

#get quantiles from beta distrib.there's a 95% posterior prob. that theta1 is bet 0.66 and 0.88, as evident in plot also
qbeta(.028, 41,11) #0.67
qbeta(.975, 41,11) #0.88

# 6) suppose the second student gets 24 questions right. what is the posterior distribution for theta2?
#    what is the 95% posterior credible interval for theta2?
#posterior is Beta(8+24, 4+40-24) = Beta(32,20)
32/(32+23) # 0.58 mea posterior
24/40 #0.6 maximum likelihood

plot(theta, dbeta(theta,32,20), type="l") #posterior
lines(theta,dbeta(theta, 8,4), lty=2) #prior
lines(theta, 44*dbinom(24,size=40, p=theta), lty=3)
#again posterior is between likelihood and prior 
#posterior mass is between .45 and 0.8

#posterior probabilities that theta1 > 1/4 
1-pbeta(0.25,32,20)
1-pbeta(0.5, 32,20)
1-pbeta(0.8,32,20)

#95% equal tailed credible interval for theta2
qbeta(0.025,32, 20) #.48
qbeta(0.975,32,20) #.74
# 7) what is the posterior probability that theta1 > theta2, ie the first student has a better chance to get
#     the questions right than the second student
#?? difficult to answer in closed form, let's answer it by simulation:
#draw 1000 samples and from each posterior distribution to see how often we obser theta1>theta2
theta1s <- rbeta(1000, 41,11)
theta2s <- rbeta(1000, 32,20)
mean(theta1s > theta2s) #.975


# Week4 linear regression in R ----

# 23 previous space shuttle launches before the Challenger disaster
# T is the temperature in Fahrenheit, I is the O-ring damage index

oring=read.table("./Challenger2.tsv", header = TRUE)
attach(oring)
#note: masking T=TRUE
oring

class(oring$Index)
class(oring$Temp)

plot(oring$Temp,oring$Index)

oring.lm=lm(Index ~ Temp, data = oring)
summary(oring.lm)

# add fitted line to scatterplot
lines(oring$Temp,fitted(oring.lm))            
# ??******* 95% posterior interval for the slope based on summary/Residual standard error/21 dgrees of freedom
-0.24337 - 0.06349*qt(.975,21)
-0.24337 + 0.06349*qt(.975,21)
# note that these are the same as the frequentist confidence intervals

# the Challenger launch was at 31 degrees Fahrenheit
# how much o-ring damage would we predict?
# y-hat
18.36508-0.24337*31
coef(oring.lm)
coef(oring.lm)[1] + coef(oring.lm)[2]*31  
# (Intercept)  #******** it actually means y_hat 
# 10.82052


# posterior prediction interval (same as frequentist)
new_dat <- data.frame(Temp=31)
predict(oring.lm,newdata=new_dat)
predict(oring.lm,new_dat ,interval="predict")  
10.82052-2.102*qt(.975,21)*sqrt(1+1/23+((31-mean(oring$Temp))^2/22/var(oring$Temp)))

# Do some Bayesian stuff:
# ??posterior probability that damage index is greater than zero
# ??waht's the probability of a t with the given center and scale
1-pt((0-10.82052)/(2.102*sqrt(1+1/23+((31-mean(oring$Temp))^2/22/var(oring$Temp)))), 21) #



# Galton's seminal data on predicting the height of children from the 
# heights of the parents, all in inches

heights <- read.table("./Galton.txt", header = TRUE)

head(heights)
attach(heights)
class(heights)

heights <- lapply(heights, as.numeric)
# library(dplyr)
# heights <- mutate_all(heights, function(x) as.numeric(x))

pairs(heights)
View(heights)
str(heights)
n="Family"
library(stringr)

for(n in colnames(heights)) {
  #n <- str_sub(n,2,-2 )
  if(!is.numeric(heights[[n]])) {
    #print(head(heights[[n]]))
    heights[[n]] <- as.numeric(heights[[n]])
  }
    
}

print(heights[[n]])

heights$Family

summary(lm(Height~Father+Mother+Gender+Kids))
summary(lm(Height~Father+Mother+Gender))
#omit Kids coz the stderr is close to the estimate
heights.lm=lm(Height~Father+Mother+Gender)

# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 15.34476    2.74696   5.586 3.08e-08 ***
#   Father       0.40598    0.02921  13.900  < 2e-16 ***
#   Mother       0.32150    0.03128  10.277  < 2e-16 ***
#   GenderM      5.22595    0.14401  36.289  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 2.154 on 894 degrees of freedom

# each extra inch taller a father is is correlated with 0.4 inch extra

# each extra inch taller a mother is is correlated with 0.3 inch extra

#****  a male child is on average 5.2 inches taller than a female child (look at the summary)
# 95% posterior interval for the the difference in height by gender
5.226 - 0.144*qt(.975,894)
5.226 + 0.144*qt(.975,894)
 
# posterior prediction interval (same as frequentist)
# what's the predicted height and 95% probability interval for a child's height given the covariate values. 
predict(heights.lm,data.frame(Father=68,Mother=64,Gender="M"),interval="predict")
predict(heights.lm,data.frame(Father=68,Mother=64,Gender="F"),interval="predict")

#Bayesian apporach facilitates quantification of uncertainty   




