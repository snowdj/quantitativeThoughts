---
layout: post
title: Linear State Space Linear Models, and Kalman Filters
tags: [Kalman Filters, State Space Model, Dynamical Linear Model]
categories: [R]
author: Lalas
---
{% include JB/setup %}
 

### Introduction

In this post, we will cover the topic of Linear State Space Models and the R-package, **dlm**([Petris, 2010](http://www.jstatsoft.org/v36/i12/)). The example we cover are taken from the [slides](http://www.rinfinance.com/agenda/2012/workshop/Zivot+Yollin.pdf) prepared by Eric Zivot and Guy Yollin; and the [slides](http://definetti.uark.edu/~gpetris/UseR-2011/SSMwR-useR2011handout.pdf) prepared by Giovanni Petris. The following are a list of topic covered:

  +   [State Space Models](#topic1) 
  +   [Dynamics Linear Models](#topic2) 
-   [Dynamics Linear Models in R](#topic2.1) 
  +   [Kalman Filters](#topic3)
  +   [Numerical Examples](#topic4)
-   [Regression Example](#topic4.1)
-   [Random Walk Plus noise Example](#topic4.2)
  +   [Seemingly Unrelated Time Series Equations (SUTSE)](#topic5)
-   [GDP example](#topic5.1)
  +   [Seemingly Unrelated Regression models](#topic6)
-   [CAPM example](#topic6.1)  
  +   [Dynamic Common Factors Model](#topic7)
-   [Example](#topic7.1) 


### <a name="topic1"></a> State Space Models

A State Space model, is composed of:

1.  Un-observable state: {\\(x_0, x_1, x_2,..., x_t,...\\)} which form a Markov Chain.
2.  Observable variable: {\\(y_0, y_1, y_2,..., y_t,...\\)} which are conditionally independent given the state.

It is specified by:

* \\(p(x_o)\\) which is initial distribution of states.
* \\(p(x_t|x_\{t-1\})\\) for \\(t = 1, 2, 3 ...\\) which is the transition probabilities of state from time \\(t-1\\) to \\(t\\)
* \\(p(y_t|x_t)\\) for \\(t = 1, 2, 3 ...\\) which is the conditional probability of the variable \\(y\\) at time \\(t\\); given that the state of the system, \\(X\\) at time \\(t\\).

### <a name="topic2"></a> Dynamics Linear Models

Dynamical Linear Models can be regarded as a special case of the state space model; where all the distributions are Gaussian. More specifically:

\\[
\begin{align}
x_0 \quad & \sim \quad N_p(m_0;C_0) \\\\
x_t|x_\{t-1\} \quad & \sim \quad N_p( G_t \, \cdot x\_{t-1};W_t) \\\\
y_t|x_t \quad & \sim \quad N_m(F_t \cdot x_t;V_t) \\\\
\end{align}
\\]

where:

* \\(F_t\\) is a \\(p \times m\\) matrices
* \\(G_t\\) is a \\(p \times p\\) matrices
* \\(V_t\\) is a \\(m \times m\\) variance/co-variance matrix
* \\(W_t\\) is a \\(p \times p\\) variance/co variance matrix

###### Note that:

Often in the literature of State Space models, the symbol \\(x_t\\) and \\(\theta_t\\) are used interchangeably to refer to the state variable. For the rest of this tutorial, we will be using the symbol \\(\theta\\) unless otherwise specified.

#### <a name="topic2.1"></a> Dynamics Linear Models in R

An equivalent formulation for a DLM is specified by the set of equations:

\\[ 
\begin{align}
y_t &= \, \, F_t \, \theta_t \, \, \, + \upsilon_t \qquad & \upsilon_t \sim N_m(0,V_t) \qquad & (1) \\\\ 
\theta_t &= \, G_t \, \theta_\{t-1\} + \omega_t\qquad & \omega_t \sim N_p(0,W_t) \qquad & (2) \\\\
\end{align}
\\]

for \\(t = 1,...\\) The specification of the model is completed by assigning a prior distribution for the initial (pre-sample) state \\(\theta_0\\). 

\\[
\begin{align}
\theta_0 \quad \sim \quad N(m_0 ,C_0) \qquad \qquad \qquad \qquad \qquad \quad & (3)
\end{align}
\\]
That is a normal distribution with mean \\(m_0\\) and variance \\(C_0\\). 

###### Note that:

* \\(y_t\\) and \\(\theta_t\\) are _m_ and _p_ dimensional random vectors, respectively
* \\(F_t\\), \\(G_t\\), \\(V_t\\) and \\(W_t\\) are real matrices of the appropriate dimensions. 
* The sequences \\(\upsilon_t\\) and \\(\omega_t\\) are assumed to be independent, both within and between, and independent of \\(\theta_0\\).
* Equation (1) above, is called the __measurement equation__ and it describes the vector of observations \\(y_t\\) through the signal \\(\theta_t\\) and a vector of disturbances \\(\upsilon_t\\)
* Equation (2) above, is called the __transition equation__ and it describes the evolution of the state vector over time using a first order Markov structure. 

In most applications, \\(y_t\\) is the value of an observable time series at time \\(t\\), while \\(\theta_t\\) is an unobservable state vector.


##### Time invariant models

In this model, the matrices \\(F_t\\), \\(G_t\\), \\(V_t\\), and \\(W_t\\) are constant and do NOT vary with time. Since they don't vary with time, we will drop the subscript \\(t\\).

###### Example 1: Random walk plus noise model (polynomial model of order one)

In this example the system of equations are
\\[
\begin{align}
y_t &= \theta_t \quad + \upsilon_t \qquad & \upsilon_t \sim N(0,V) \\\\ 
\theta_t &= \theta_\{ t-1 \} \, + \omega_t  \qquad & \omega_t \sim N(0,W) \\\\
\end{align}
\\]

Suppose one wants to define in R such a model, with \\(V = 3.1\\), \\(W = 1.2\\), \\(m_0 = 0\\), and \\(C_0 = 100\\). This can be done as follows:


{% highlight r %}
library(dlm)
 myModel <- dlm(FF = 1, V = 3.1, GG = 1, W = 1.2, m0 = 0, C0 = 100)
{% endhighlight %}
where \\(F_t\\) (the _FF_ parameters above), should be the identity matrix, but since \\(p\\) the number of dimension of \\(\theta\\) is equal to 1, \\(F_t\\) is reduced to 1. Similarly arguments goes for the \\(G_t\\) variable (the _GG_ parameters above).

##### Time Varying DLM

A _dlm_ object may contain, in addition to the components \\(FF\\), \\(V\\), \\(GG\\), \\(W\\), \\(m0\\), and \\(C0\\) described above, one or more of \\(JFF\\), \\(JV\\), \\(JGG\\), \\(JW\\), and \\(X\\). While \\(X\\) is a matrix used to store all the time-varying elements of the model, the components are indicator matrices whose entries signal whether an element of the corresponding model matrix is time-varying and, in case it is, where to retrieve its values in the matrix \\(X\\).

###### Example 2: Linear Regression (time varying parameters)

In the standard DLM representation of a _simple linear regression_ models, the state vector is \\(\theta_t =  \left( \alpha_t\, ;\beta_t \right)^{\prime}\\), the vector of regression coefficients, which may be constant or time-varying.  In the case of time varying, the model is:

\\[
\begin{align}
y_t  &= \alpha_t + \beta_t \, x_t + \epsilon_t   \qquad & \epsilon_t \, \sim N(0,\sigma\^2) \\\\
\alpha_t &= \quad \alpha\_{t-1} \quad + \epsilon_t\^{\alpha} \qquad & \epsilon_t\^{\alpha} \sim N(0,\sigma\_{\alpha}\^2) \\\\
\beta_t  &= \quad \beta\_{t-1}  \quad + \epsilon_t\^{\beta}   \qquad & \epsilon_t\^{\beta} \sim N(0, \sigma\_{\beta}\^2) \\\\
\end{align}
\\]
where

* \\(\epsilon_t\\), \\(\epsilon_t^\alpha\\) and \\(\epsilon_t^\beta\\) are _iid_
* The matrix \\(F_t\\) is [\\(1 \, x_t\\)] where \\(x_t\\) are value of the co-variate for observation \\(y_t\\)
* \\(V\\) is \\(\sigma\^{2}\\)

More generally, a dynamic linear regression model is described by:

\\[
\begin{align}
y_t &= \mathbf{ x_t^\prime } \theta_t  + v_t \qquad & \upsilon_t \sim N(0,V_t) \\\\
\theta_t &= \, G_t \, \theta_\{t-1\} + \omega_t  \qquad & \omega_t  \sim N(0,W_t) \\\\
\end{align}
\\]

where the coefficients:

* \\(\mathbf{ x_t\^{\prime} } := [x\_{1t}, \dots , x\_{pt}]\\) are the _p-explanatory_ variables at time \\(t\\). These are not assumed to be stochastic in this model, but rather fixed (i.e. this is a conditional model of \\(y_t|x_t\\)
* The system matrix \\(G_t\\) is the \\(2×2\\) identity matrix and 
* the observation matrix is \\(F_t\\) = [\\(1 \, x_t\\)], where \\(x_t\\) is the value of the co-variate for observation \\(y_t\\).

A popular default choice for the state equation is to take the evolution matrix \\(G_t\\) as the identity matrix and \\(W\\) diagonal, which corresponds to modeling the regression coefficients as independent random walks

Assuming the variances \\(V_t\\) and \\(W_t\\) are constant, the only time-varying element of the model is the \\((1, 2)^\{th\}\\) entry of \\(F_t\\). 

Accordingly, the component \\(X\\) in the _dlm object_ will be a one-column matrix containing the values of the co-variate \\(x_t\\), while \\(JFF\\) will be the \\(1 \times 2\\) matrix [0 1], where the ‘0’ signals that the \\((1, 1)^\{th\}\\) component of \\(F_t\\) is constant, and the ‘1’ means that the \\((1, 2)^{th}\\) component is time-varying and its values at different times can be found in the first column of \\(X\\).

###### Example 3: Local Linear Trend

In example 1, we described the _random walk_ plus noise model, also called polynomial model of order one. In this example, we examine the _local linear trend_ model, also called _polymonial model of order two_, which is described by the following set of equations:

\\[
\begin{align}
y_t &= \qquad \quad \mu_t  + \upsilon_t  \quad &\upsilon_t \sim N(0,V) \\\\ 
\mu_t &= \mu\_{t-1}  + \delta\_{t-1} + \omega_t\^{\mu} \quad & \omega_t\^{\mu} \sim N(0,W\^{\mu}) \\\\
\delta_t &= \qquad \,\, \, \delta\_{t-1} + \omega_t\^{\delta} \quad & \omega_t\^{\delta} \sim N(0,W\^{\delta}) \\\\
\end{align}
\\]

This model is used to describes dynamic of “straight line” observed with noise. Hence

* levels \\(\mu_t\\) which are locally linear function of time
* straight line if \\(W^\mu\\) and \\(W^\delta\\) are equal to 0
* When \\(W^\mu\\) is 0; \\(\mu\\) follows an integrated random walk

##### Signal-to-Noise Ratio

In the case where \\(m = p = 1\\) where _m_ and _p_ are the dimension of the variance matrices, \\(V\\) and \\(W\\) respectively; we can define the signal-to-noise ratio as being \\(W_t/V_t\\) where \\(W_t\\) is the evolution variance of the state \\(\theta\\) from \\(t\\) to \\(t+1\\) and \\(V_t\\) is the variance of observation given the state.

### Dynamic Linear Model package.

In this section we will examine some of the functions used in the `DLM` R package. Using this package, a user can define the _Random Walk plus noise_ model using the _dlmModPoly_ helper function as follows:


{% highlight r %}
myModel <- dlmModPoly(order = 1, dV = 3.1, dW = 1.2, C0 = 100)
{% endhighlight %}

Other helpers function exist for more complex models such as:

* `dlmModARMA`: for an ARMA process, potentially multivariate
* `dlmModPoly`: for an \\(n^\{th\}\\) order polynomial
* `dlmModReg` : for Linear regression
* `dlmModSeas`: for periodic – Seasonal factors
* `dlmModTrig`: for periodic – Trigonometric form

##### sum of DLM
The previous helper function can be combined to create a more complex models. For example: 


{% highlight r %}
myModel <- dlmModPoly(2) + dlmModSeas(4)
{% endhighlight %}

creates a Dynamical Linear Model representing a time series for quarterly data, in which one wants to include a local linear trend (polynomial model of order 2) and a seasonal component.

##### Outter sum of DLM
Also Two DLMs, modeling an _m1-_ and an _m2-variate_ time series respectively, can also be combined into a unique DLM for _m1 + m2-_ variate observations. For example


{% highlight r %}
 bivarMod <- myModel %+% myModel
{% endhighlight %}
creates two univariate models for a local trend plus a quarterly seasonal component as the one described above can be combined as follows (here _m1 = m2 = 1_). In this case the user has to be careful to specify meaningful values for the variances of the resulting model after model combination. 

Both sums and outer sums of DLMs can be iterated and they can be combined to allow the specification of more
complex models for multivariate data from simple standard univariate models.

##### Setting and Getting component of models

* In order to fetch or set a component of a model defined, one can use the function \\(FF\\), \\(V\\), \\(GG\\), \\(W\\), \\(m0\\), and \\(C0\\). 
* If the model have time varying parameters, these could accessed through the \\(JFF\\), \\(JV\\), \\(JGG\\), \\(JW\\), and \\(X\\).

For example:

{% highlight r %}
V(myModel)
m0(myModel) <- rnorm()
{% endhighlight %}

### <a name="topic3"></a> Kalman Filters

#### Filtering:

Let \\( y\^t = (y_1, ..., y_t) \\) be the vector of observation up to time \\(t\\). The _filtering_ distributions, \\(p(\theta_t|y\^{t})\\) can be computed recursively as follows:

1.  Start with \\( \theta_0 \sim N(m_0, C_0) \\) at _time_ 0

2.  One step forecast for the _state_ 
\\\[ \theta_t|y^\{t-1\} \sim N(a_t, R_t) \\]
where \\(a_t = G_t \cdot m\_{t-1} \\), and \\(R_t = G_t C_\{t-1\} G_t^\prime + W_t\\)

3.  One step forecast for the _observation_ 
\\[ y_t|y^{t-1} \sim N(f_t, Q_t) \\]
where \\(f_t = F_t \cdot a_t\\), and \\(Q_t = F_t R_\{t-1\} F_t^\prime + V_t\\)

4.  Compute the _posterior_ at time $t$; 
\\[ \theta_t|y\^t \sim N(m_t, C_t) \\] 
where \\(m_t = a_t + R_t \, f_t^\prime Q_t^\{-1\} (y_t - f_t)\\), and \\(C_t = R_t - R_t F_t^\prime Q_t^\{-1\} F_t R_t\\)

##### Filtering in the DLM package

The function `dlmFilter` returns:

* the series of filtering means \\(m_t = E(\theta_t|y\^{t})\\) – includes \\(t = 0\\)
* the series of filtering variances \\(C_t = Var(\theta_t|y\^{t})\\) – includes \\(t = 0\\)
* the series of one-step forecasts for the state \\(a_t = E(\theta_t|y\^{t−1})\\)
* the series of one-step forecast variances for the state \\(R_t = Var(\theta_t|y\^{t−1})\\)
* the series of one-step forecasts for the observation \\(f_t = E(y_t|y\^{t−1})\\)


#### Smoothing:

Backward recursive algorithm can be used to obtain \\(p(\theta_t|y^T)\\) for a fixed \\(T\\) and for \\(t =\\) {\\(0, 1, 2, ...T\\)}.

1.  Starting from \\(\theta_T|y_T \sim N(m_T, C_T\\) at time \\(t = T\\)
2.  For \\(t\\) in \\(\left \\{ T - 1, T - 2, ..., 0 \right \\}\\)
\\[ \theta_t|y\^{T} \sim N(s_t, S_t) \\]
where:
\\[
\begin{align}
    s_t &= m_t + C_t \, G\_{t+1}^{\, \prime} R\_{t+1}\^{-1}(s\_{t+1} - a\_{t + 1}) \qquad \textrm{and} \\\\
    S_t &= C_t - C_t \, G\_{t+1}^{\, \prime} R\_{t+1}\^{-1}(R\_{t+1} - S\_{t + 1}) \, R\_{t+1}^{-1} G\_{t+1}\^{\, \prime} C_t \\\\
\end{align}
\\]

##### Smoothing in the DLM package

The function `dlmSmooth` returns:

* the series of smoothing means \\( s_t = E(\theta_t|y\^{T})\\) – includes \\(t = 0\\)
* the series of smoothing variances \\(S_t = Var(\theta_t|y\^{T})\\) – includes \\(t = 0\\)

###### Note that:

* Updating in _Filtering_ is sequential which allows for _online_ processing; while in __Smoothing_ it is not; so processing is _offline_
* The formulas shown above are numerically unstable, but other more stable algorithms are available; such as square-root filters. Package `dlm` implements a filter algorithm based on the _singular value decomposition_ (SVD) of all the variance-co-variance matrices.

#### Forcasting:

To calculate the forecast distributions, for \\(k = 1, 2, ...\\) etc, we proceed as follows:

1.  starting with a sample from \\(\theta_T|y_T \sim N(m_T, C_T)\\)
2.  Forecast the _state_:
\\[ \theta\_{T+k}|y\^{T} \sim N(a_k\^{T}, R_k\^{T}) \\]
where: \\(a_t\^{k} = G\_{T+k} \, a\_{k-1}\^{T} \\) and \\( R_k\^{T} = G\_{T+k} \, R\_{k-1}\^{T} G\_{T+k}^{\, \prime} + W\_{T+k}\\)

3.  Forecast the _observation_:
\\[ \theta\_{T+k}|y\^T \sim N(f_k\^{T}, Q_k\^{T}) \\]
where: \\(f_k\^{T} = F\_{T+k} \, a_k\^{T} \\) and \\(Q_k\^{T} = F\_{T+k} \, R_k\^{T} F\_{T+k}^{\, \prime} + V\_{T+k}\\)

### Residual Analysis and model checking

One way of verifying model assumption is to look at the model residuals. The One-step forecast errors, or innovations can be defined by \\(e_t = y_t − E(y_t|y\_{t−1})\\). The innovations are a sequence of independent Gaussian random variables. These innovations can be used to obtain a sequence of _i.i.d_ standard normal random variables as follows:

\\[ \tilde {e_t} = \frac{e_t }{\sqrt{ Q_t } }\\] where \\(F_t = E(y_t|y^\{t-1\})\\) and \\(Q_t = \textrm{Var}(y_t|y^\{t-1\})\\) are both estimates obtained from the Kalman Filter.

### Notes about MLE estimation

Since we are estimating the parameters using the MLE method, there is not guarantee that the answer we obtain is the "optimal" solution. The common causes for this may be:

* Multiple local maxima of the log-likelihood
* Fairly flat log-likelihood surface

In order to gain confidence in the values return by the `dlmMLE` function, it's usually recommended that

1.  Repeat the optimization process several times, with different initial parameter values.
2.  Make sure the _Hessian_ of the negative log-likelihood at the maximum is positive definite.
3.  Compute standard errors of MLEs to get a feeling for the accuracy of the estimates

One way of computing the _Hessian_ matrix in R is to use the function `hessian` part of the `numDeriv` package. We will see a demonstration of this in example 5 below

### <a name="topic4"></a> Numerical Examples

##### <a name="topic4.1"></a> Example 4: Regression Model
The next example are taking from the [slides](http://www.rinfinance.com/agenda/2012/workshop/Zivot+Yollin.pdf) prepared by Eric Zivot and Guy Yollin. See slides 12-13 for more information.

First we load the data as such:




{% highlight r %}
library(PerformanceAnalytics, quietly = TRUE,  warn.conflicts = FALSE)
data(managers)
# extract HAM1 and SP500 excess returns
HAM1 = 100*(managers[,"HAM1", drop=FALSE] - managers[,"US 3m TR", drop=FALSE])
sp500 = 100*(managers[,"SP500 TR", drop=FALSE] - managers[,"US 3m TR",drop=FALSE])
colnames(sp500) = "SP500"
{% endhighlight %}
Then we can proceed by specifying a dynamical regression model and its parameters


{% highlight r %}
# Specifying a set model parameters
s2_obs = 1      # Variance of observations
s2_alpha = 0.01 # Variance of the alpha regression parameter
s2_beta = 0.01  # Variance of the beta regression parameter
# Construct a regression model
tvp.dlm = dlmModReg(X=sp500, addInt=TRUE, dV=s2_obs, dW=c(s2_alpha, s2_beta))
{% endhighlight %}

Now that we have define _a_ model, we can view its different component:

{% highlight r %}
# looking at the various component
tvp.dlm[c("FF","V","GG","W","m0","C0")]
tvp.dlm[c("JFF","JV","JGG","JW")]
{% endhighlight %}

If we were to do a simple linear regression (Ordinary Least Square fit - constant equity beta), we would do something like

{% highlight r %}
ols.fit = lm(HAM1 ~ sp500)
summary(ols.fit)
{% endhighlight %}



{% highlight text %}
## 
## Call:
## lm(formula = HAM1 ~ sp500)
## 
## Residuals:
##    Min     1Q Median     3Q    Max 
## -5.178 -1.387 -0.215  1.263  5.744 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)   0.5775     0.1697    3.40  0.00089 ***
## sp500         0.3901     0.0391    9.98  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.93 on 130 degrees of freedom
## Multiple R-squared:  0.434,	Adjusted R-squared:  0.43 
## F-statistic: 99.6 on 1 and 130 DF,  p-value: <2e-16
{% endhighlight %}
In order to define and estimate this regression using dynamical system model, we would use the Maximum Likelihood Estimation (MLE) method to estimates the 3 parameters: `s2_obs`, `s2_alpa` and `s2_beta`. We first start by defining initial values for the estimation methods

{% highlight r %}
start.vals = c(0,0,0)
# Names ln variance of: observation y, alpha and beta (corresponding intercept and slope of y (HAM1) with respect to X (sp500))
names(start.vals) = c("lns2_obs", "lns2_alpha", "lns2_beta")

# function to build Time Varying Parameter state space model
buildTVP <- function(parm, x.mat){
    parm <- exp(parm)
  return( dlmModReg(X=x.mat, dV=parm[1], dW=c(parm[2], parm[3])) )
}

# Estimate the model
TVP.mle = dlmMLE(y=HAM1, parm=start.vals, x.mat=sp500, build=buildTVP, hessian=T)

# get sd estimates
se2 <- sqrt(exp(TVP.mle$par))
names(se2) = c("s_obs", "s_alpha", "s_beta")
sqrt(se2)
{% endhighlight %}



{% highlight text %}
##   s_obs s_alpha  s_beta 
## 1.32902 0.03094 0.23528
{% endhighlight %}

Now that we have an estimates for the model, we can build the "optimal" model using the estimates we obtained in the previous step.


{% highlight r %}
# Build fitted ss model, passing to it sp500 as the matrix X in the model
TVP.dlm <- buildTVP(TVP.mle$par, sp500)
{% endhighlight %}

#### Filtering and Smooting:

* __Filtering__ Optimal estimates of \\(\theta_t\\) given information available at time \\(t\\), \\(I_t=\left \\{ y_1,...,y_t \right\\}\\)
* __Smoothing__ Optimal estimates of \\(\theta_t\\) given information available at time \\(T\\), \\(I_T =\left \\{ y_,...,y_T \right\\}\\)

Now that we have obtained model estimates, and build the optimal model, we can _filter_ the data through it, to obtain filtered values of the state vectors, together with their variance/co-variance matrices.


{% highlight r %}
TVP.f <- dlmFilter(y = HAM1, mod = TVP.dlm)
class(TVP.f)
{% endhighlight %}



{% highlight text %}
## [1] "dlmFiltered"
{% endhighlight %}



{% highlight r %}
names(TVP.f)
{% endhighlight %}



{% highlight text %}
## [1] "y"   "mod" "m"   "U.C" "D.C" "a"   "U.R" "D.R" "f"
{% endhighlight %}

Similarly, to obtained the smoothed values of the state vectors, together with their variance/co-variance matrices; using knowledge of the entire series

{% highlight r %}
# Optimal estimates of θ_t given information available at time T.
TVP.s <- dlmSmooth(TVP.f)
class(TVP.s)
{% endhighlight %}



{% highlight text %}
## [1] "list"
{% endhighlight %}



{% highlight r %}
names(TVP.s)
{% endhighlight %}



{% highlight text %}
## [1] "s"   "U.S" "D.S"
{% endhighlight %}
 
##### Plotting the results (smoothed values)

Now that we have obtained the smoothed values of the state vectors, we can draw them as:

{% highlight r %}
# extract smoothed states - intercept and slope coefs
alpha.s = xts(TVP.s$s[-1,1,drop=FALSE], as.Date(rownames(TVP.s$s[-1,])))
beta.s  = xts(TVP.s$s[-1,2,drop=FALSE], as.Date(rownames(TVP.s$s[-1,])))
colnames(alpha.s) = "alpha"
colnames(beta.s)  = "beta"
{% endhighlight %}
Extracting the std errors and constructing the confidence band

{% highlight r %}
# extract std errors - dlmSvd2var gives list of MSE matrices
mse.list = dlmSvd2var(TVP.s$U.S, TVP.s$D.S)
se.mat = t(sapply(mse.list, FUN=function(x) sqrt(diag(x))))
se.xts = xts(se.mat[-1, ], index(beta.s))
colnames(se.xts) = c("alpha", "beta")
a.u = alpha.s + 1.96*se.xts[, "alpha"]
a.l = alpha.s - 1.96*se.xts[, "alpha"]
b.u = beta.s  + 1.96*se.xts[, "beta"]
b.l = beta.s  - 1.96*se.xts[, "beta"]
{% endhighlight %}
And plotting the results with +/- 2 times the standard deviation

{% highlight r %}
# plot smoothed estimates with +/- 2*SE bands
chart.TimeSeries(cbind(alpha.s, a.l, a.u), main="Smoothed estimates of alpha", ylim=c(0,1),
                 colorset=c(1,2,2), lty=c(1,2,2),ylab=expression(alpha),xlab="")
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/esimatestWithBandsReg1.png) 

{% highlight r %}
chart.TimeSeries(cbind(beta.s, b.l, b.u), main="Smoothed estimates of beta",
                 colorset=c(1,2,2), lty=c(1,2,2),ylab=expression(beta),xlab="")
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/esimatestWithBandsReg2.png) 

or using package `ggplot2` ([Saravia, 2012](http://stackoverflow.com/questions/14033551/r-plotting-confidence-bands-with-ggplot2))
we would do

{% highlight r %}
library(ggplot2, warn.conflicts = FALSE)
alpha.df <- data.frame(dateTime = index(se.xts), alpha = alpha.s, upr = a.u, lwr = a.l)
names(alpha.df) <- c("dateTime", "alpha", "upr", "lwr")
beta.df  <- data.frame(dateTime = index(se.xts), beta = beta.s, upr = b.u, lwr = b.l)
names(beta.df) <- c("dateTime", "beta", "upr", "lwr")
{% endhighlight %}

{% highlight r %}
## Plotting alpha
ggplot(data = alpha.df, aes(dateTime, alpha) ) + geom_point () + geom_line() + geom_ribbon(data=alpha.df, aes(ymin=lwr,ymax=upr), alpha=0.3) + labs(x = "year", y = expression(alpha), title = expression(paste("State Space Values of ", alpha, " over Time")))
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/paramReg1.png) 

{% highlight r %}
## Plotting beta
ggplot(data = beta.df, aes(dateTime, beta) ) + geom_point (data = beta.df, aes(dateTime, beta) ) + geom_line() + geom_ribbon(data=beta.df , aes(ymin=lwr,ymax=upr), alpha=0.3) + labs(x = "year", y = expression(beta), title = expression(paste("State Space Values of ", beta, " over Time")))
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/paramReg2.png) 

##### Forcasting using Kalman Filter

We will be doing a 10-steps ahead forecast using the calibrated model:


{% highlight r %}
# Construct add 10 missing values to end of sample
new.xts = xts(rep(NA, 10), seq.Date(from=end(HAM1), by="months", length.out=11)[-1])
# Add this NA data to the original y (HAM1) series
HAM1.ext = merge(HAM1, new.xts)[,1]
# Filter extended y (HAM1) series
TVP.ext.f = dlmFilter(HAM1.ext, TVP.dlm)
# extract h-step ahead forecasts of state vector
TVP.ext.f$m[as.character(index(new.xts)),]
{% endhighlight %}



{% highlight text %}
##              [,1]   [,2]
## 2007-01-31 0.5334 0.6873
## 2007-03-03 0.5334 0.6873
## 2007-03-31 0.5334 0.6873
## 2007-05-01 0.5334 0.6873
## 2007-05-31 0.5334 0.6873
## 2007-07-01 0.5334 0.6873
## 2007-07-31 0.5334 0.6873
## 2007-08-31 0.5334 0.6873
## 2007-10-01 0.5334 0.6873
## 2007-10-31 0.5334 0.6873
{% endhighlight %}

We did not use the function `dlmForecast` part of the `dlm` package for future predictions, since that function works only with constant models!.

##### Residual Checkings


{% highlight r %}
TVP.res <- residuals(TVP.f, sd = FALSE)
# Q-Q plot
qqnorm(TVP.res)
qqline(TVP.res)
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/resReg1.png) 

{% highlight r %}
# Plotting Diagnostics for Time Series fits
tsdiag(TVP.f)
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/resReg2.png) 

##### <a name="topic4.1"></a> Example 5: Random Walk Plus noise - Nile River Data
We start this example by plotting the data set.


{% highlight r %}
data(Nile)
Nile.df <- data.frame(year = index(Nile), y = as.numeric(Nile))
qplot(y = y, x = year, data = Nile.df, geom = 'line', ylab = 'Nile water level', xlab = 'Year',
                 main = "Measurements of the annual flow of \n the river Nile at Ashwan 1871-1970")
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/nileLvl.png) 

The previous graph illustrates the annual discharge of flow of water, (in unit of \\(10\^8\\) \\(m\^3\\), of the Nile at Aswan. 

Next, we look at the effect of applying two different models with different Signal to Noise Ratio to the data, and see graphically, how well these two separate models fits the data ([Pacomet, 2012](http://stackoverflow.com/questions/10349206/add-legend-to-ggplot2-line-plot)).


{% highlight r %}
# Creating models -- the variance V is the same in mod1 and mod2; but 
# the signal variance is 10 time larger in mod2 than mod1
mod1 <- dlmModPoly(order = 1, dV = 15100, dW = 0.5 * 1468)
mod2 <- dlmModPoly(order = 1, dV = 15100, dW = 5 * 1468)
# Creating filter data
NileFilt_1 <- dlmFilter(Nile, mod1)
NileFilt_2 <- dlmFilter(Nile, mod2)
# Creating df to contain data to plot
Nile.df <- data.frame(year = index(Nile), Orig_data = as.numeric(Nile), 
                      Filt_mod_1 = as.numeric(NileFilt_1$m[-1]), Filt_mod_2 = as.numeric(NileFilt_2$m[-1]))

# Plotting the results
library(wesanderson)
myColor <- wes.palette(3, "GrandBudapest")
p <- ggplot(data = Nile.df) 
p <- p + geom_point(aes(x = year, y = Orig_data), size=3, colour= "black", shape = 21, fill = myColor[1])
p <- p + geom_line(aes(x = year, y = Orig_data,  colour = "Orig_data") , size = 1.0)
p <- p + geom_line(aes(x = year, y = Filt_mod_1, colour = "Filt_mod_1"), linetype="dotdash")
p <- p + geom_line(aes(x = year, y = Filt_mod_2, colour = "Filt_mod_2"), linetype="dotdash")
p <- p + labs(x = "year", y = "water Level", title = "Nile River water level for \n 2 different signal-to-noise ratios")
p <- p + scale_colour_manual("", breaks = c("Orig_data", "Filt_mod_1", "Filt_mod_2"),
                             labels= c("Org Data", "Filterd Data 1: SNR = x", "Filterd Data 2: SNR = 10x"),
                             values = myColor[c(2,3,1)])
p <- p + theme(legend.position="bottom")
print(p)
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/sig2noise.png) 

Next we fit a _local level_ model, as described in example 1 above.


{% highlight r %}
buildLocalLevel <- function(psi) dlmModPoly(order = 1, dV = psi[1], dW = psi[2])
mleOut <- dlmMLE(Nile, parm = c(0.2, 120), build = buildLocalLevel, lower = c(1e-7, 0))
# Checking that the MLE estimates has converged
mleOut$convergence
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Constructing the fitted model
LocalLevelmod <- buildLocalLevel(psi = mleOut$par)
# observation variance
drop(V(LocalLevelmod))
{% endhighlight %}



{% highlight text %}
## [1] 15100
{% endhighlight %}



{% highlight r %}
# system variance
drop(W(LocalLevelmod))
{% endhighlight %}



{% highlight text %}
## [1] 1468
{% endhighlight %}

**Note:** The lower bound \\(10^{-7}\\) for \\(V\\) reflects the fact that the functions in `dlm` require the matrix \\(V\\) to be non-singular. On the scale of the data, however, \\(10^{-7}\\) can be considered zero for all practical purposes.

Looking at the plot of the original data, we notice a negative spike around the year 1900. This coincided with construction of the Aswan Low Dam, by the British during the years of 1898 to 1902. Recall that the model we just fit the data to (local level plus noise) assumes that the variance matrices \\(W_t\\) and \\(V_T\\) are constants over time. Therefore, one way to improve the accuracy of this model and take the jump in water level (around 1899) into account, is to assume that the variance _did change_ on this year. The new model becomes:
\\[
\begin{align}
y_t &= \theta_t  + \upsilon_t \qquad & \upsilon_t \sim N(0,V)\\\\
\theta_t &= \, \theta\_{ t-1 } + \omega_t \qquad & \omega_t \sim N(0,W_t)\\\\
\end{align}
\\]

where
\\[
W_t = 
\begin{cases} 
W \quad  \,\,\, \textrm{if}\quad t \neq  1899 \\\\
W^*\quad \textrm{if}\quad t = 1899 \\\\
\end{cases}
\\]

We construct and estimate this model in R as follows:

{% highlight r %}
# Model Construction
buildDamEffect <- function(psi) {
   mod <- dlmModPoly(1, dV = psi[1], C0 = 1e8)
   # Creating the X matrix for the model -- For more info see Time Varying DLM section above
   X(mod) <- matrix(psi[2], nr = length(Nile))
   X(mod)[time(Nile) == 1899] <- psi[3]
   # Tell R that the values of W_t at any time are to be found in the first column of the matrix X
   JW(mod) <- 1
   return(mod)
 }
# Model estimation through MLE
mleDamEffect <- dlmMLE(Nile, parm = c(0.2, 120, 20), build = buildDamEffect, lower = c(1e-7, 0, 0))
# Verify convergence
mleDamEffect$conv
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Construct the final model
damEffect <- buildDamEffect(psi = mleDamEffect$par)
{% endhighlight %}

Finally, for the purpose of comparison, we build a _Linear Trend_ model for the Nile data set

{% highlight r %}
# Model Construction
buildLinearTrend <- function(psi) dlmModPoly(2, dV = psi[1], dW = psi[2:3], C0 = diag(1e8, 2))
# Model Estimation
mleLinearTrend <- dlmMLE(Nile, parm = c(0.2, 120, 20),build = buildLinearTrend, lower = c(1e-7, 0, 0))
# Checking convergence
mleLinearTrend$conv
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Construct Final Model
linearTrend <- buildLinearTrend(psi = mleLinearTrend$par)
{% endhighlight %}

###### MLE results checking

We validate the results returned by the MLE method and gain confidence in the results by examining:


{% highlight r %}
library(numDeriv)
# Local Level model
hs_localLevel <- hessian(function(x) dlmLL(Nile, buildLocalLevel(x)), mleOut$par)
all(eigen(hs_localLevel, only.values = TRUE)$values > 0) # positive definite?
{% endhighlight %}



{% highlight text %}
## [1] TRUE
{% endhighlight %}



{% highlight r %}
# Damn Effect model
hs_damnEffect <- hessian(function(x) dlmLL(Nile, buildDamEffect(x)), mleDamEffect$par)
all(eigen(hs_damnEffect, only.values = TRUE)$values > 0) # positive definite?
{% endhighlight %}



{% highlight text %}
## [1] TRUE
{% endhighlight %}



{% highlight r %}
# Linear Trend model
hs_linearTrend <- hessian(function(x) dlmLL(Nile, buildLinearTrend(x)), mleLinearTrend$par)
all(eigen(hs_linearTrend, only.values = TRUE)$values > 0) # positive definite?
{% endhighlight %}



{% highlight text %}
## [1] TRUE
{% endhighlight %}


###### Models Comparaison

Model selection for DLMs is usually based on either of the following criteria:

* Forecasting accuracy – such as Mean square error (MSE), Mean absolute deviation (MAD), Mean absolute percentage error (MAPE).
* Information criteria – such as AIC, BIC
* Bayes factors and posterior model probabilities (in a Bayesian setting)

If simulation is used, as it is the case in a Bayesian setting, then averages are calculated after discarding the _burn-in_ samples ([Petris, 2011](http://definetti.uark.edu/~gpetris/UseR-2011/SSMwR-useR2011handout.pdf)).

In this next piece of code, we compute these various statistics


{% highlight r %}
# Creating variable to hold the results
MSE <- MAD <- MAPE <- U <- logLik <- N <- AIC <- c()
# Calculating the filtered series for each model
LocalLevelmod_filtered <- dlmFilter(Nile, LocalLevelmod)
damEffect_filtered     <- dlmFilter(Nile, damEffect)
linearTrend_filtered   <- dlmFilter(Nile, linearTrend)
# Calculating the residuals
LocalLevel_resid  <- residuals(LocalLevelmod_filtered, type = "raw", sd = FALSE)
damEffect_resid   <- residuals(damEffect_filtered, type = "raw", sd = FALSE)
linearTrend_resid <- residuals(linearTrend_filtered , type = "raw", sd = FALSE)
# If sampling was obtained through simulation then we would remove the burn-in samples as in the next line
# linearTrend_resid <- tail(linearTrend_resid, -burn_in)
#
# Calculating statistics for different models:
# 1 LocalLevelmod
MSE["Local Level"] <- mean(LocalLevel_resid ^2)
MAD["Local Level"] <- mean(abs(LocalLevel_resid ))
MAPE["Local Level"] <- mean(abs(LocalLevel_resid) / as.numeric(Nile))
logLik["Local Level"] <- -mleOut$value
N["Local Level"] <- length(mleOut$par)
# 2 Dam Effect
MSE["Damn Effect"] <- mean(damEffect_resid^2)
MAD["Damn Effect"] <- mean(abs(damEffect_resid))
MAPE["Damn Effect"] <- mean(abs(damEffect_resid) / as.numeric(Nile))
logLik["Damn Effect"] <- -mleDamEffect$value
N["Damn Effect"] <- length(mleDamEffect$par)
# 3 linear trend
MSE["linear trend"] <- mean(linearTrend_resid^2)
MAD["linear trend"] <- mean(abs(linearTrend_resid))
MAPE["linear trend"] <- mean(abs(linearTrend_resid) / as.numeric(Nile))
logLik["linear trend"] <- -mleLinearTrend$value
N["linear trend"] <- length(mleLinearTrend$par)
# Calculating AIC and BIC- vectorized, for all models at once
AIC <- -2 * (logLik - N)
BIC <- -2 * logLik + N * log(length(Nile))
# Building a dataframe to store the results
results <- data.frame(MSE = MSE, MAD = MAD, MAPE = MAPE, logLik = logLik, AIC = AIC, BIC = BIC, NumParameter = N)
{% endhighlight %}

The following table summarize the results of the different model applied to the Nile data set. 


{% highlight r %}
# Producing a table
kable(results, digits=2, align = 'c')
{% endhighlight %}



{% highlight text %}
## 
## 
## |             |  MSE  |  MAD  | MAPE | logLik | AIC  | BIC  | NumParameter |
## |:------------|:-----:|:-----:|:----:|:------:|:----:|:----:|:------------:|
## |Local Level  | 33026 | 123.7 | 0.14 | -549.7 | 1103 | 1109 |      2       |
## |Damn Effect  | 30677 | 115.6 | 0.13 | -543.3 | 1093 | 1100 |      3       |
## |linear trend | 37927 | 133.6 | 0.15 | -558.2 | 1122 | 1130 |      3       |
{% endhighlight %}

As we can the _Damn Effect_ model is the _best_ model out of the 3 models considered.

### <a name="topic5"></a> Seemingly Unrelated Time Series Equations (SUTSE)

SUTSE is a __multivariate__ model used to describe univariate series that can be marginally modeled by the “same” DLM – same observation matrix \\(F_0\\) and system matrix \\(G_0\\), _p-dimensional_ state vectors have the same interpretation. 

For example, suppose there are \\(k\\) observed time series, each one might be modeled using a linear growth model, so that for each of them the state vector has a level and a slope component. Although not strictly required, it is **commonly assumed** for simplicity that the variance matrix of the system errors, \\(W\\) is diagonal. The system error of the dynamics of this common state vector will then be characterized by a block-diagonal variance matrix having a first \\(k \times k\\) block accounting for the correlation among levels and a second \\(k \times k\\) block accounting for the correlation among slopes. In order to introduce a further correlation between the \\(k\\) series, the observation error variance \\(V\\) can be taken non-diagonal.

##### Example

GDPs of different countries can all be described by a local linear trend – but obviously they are _not independent_
For \\(k\\) such series, define the observation and system matrices of the joint model as:

\\[ 
\begin{align}
F &= F_0 \, \otimes \,I_k  \\\\
G &= G_0 \, \otimes \,I_k  \\\\
\end{align}
\\]

where \\(I_k\\) is the \\(k \times k\\) Identity matrix, and \\(\otimes\\) is the _Kronecker_ product. The \\(pk \times pk\\) system variance \\(W\\) is typically given a block-diagonal form, with _p_ \\(k \times k\\) blocks, implying a correlated evolution of the \\(j^\{th\}\\) components of the state vector of each series. The \\(k \times k\\) observation variance \\(V\\) may be non-diagonal to account for additional correlation among the different series.

**Note:** Given two matrices \\(A\\) and \\(B\\), of dimensions \\(m \times n\\) and \\(p \times q\\) respectively, the _Kronecker_ product \\(A \otimes B\\) is the \\(mp × nq\\) matrix defined as:

\\[
\begin{bmatrix} 
a\_{11} \mathbf{B} & \cdots & a\_{1n}\mathbf{B} \\\\ 
\vdots & \ddots & \vdots \\\\ 
a\_{m1} \mathbf{B} & \cdots & a\_{mn} \mathbf{B} \\\\
\end{bmatrix}
\\]

##### <a name="topic5.1"></a> Example 6: GDP data

Consider GDP of Germany, UK, and USA. The series are obviously correlated and, for each of them – possibly on a log scale – a local linear trend model or possibly a more parsimonious integrated random walk plus noise seems a reasonable choice.


{% highlight r %}
# We are unloading the namespace "ggplot2" since it mask the function %+% (outer sum) from the dlm package.
# http://stackoverflow.com/questions/3241539/how-to-unmask-a-function-in-r
unloadNamespace("ggplot2")
#
# Reading in the data
gdp0 <- read.table("http://definetti.uark.edu/~gpetris/UseR-2011/gdp.txt", skip = 3, header = TRUE)
# Transforming the data to the log-scale
gdp <- ts(10 * log(gdp0[, c("GERMANY", "UK", "USA")]), start = 1950)
# the ’base’ univariate model. 
uni <- dlmModPoly() # Default paramters --> order = 2 --> stochastic linear trend.
# Creating an outter sum of models to  get the matrices of the correct dimension.
gdpSUTSE <- uni %+% uni %+% uni
# ## now redefine matrices to keep levels together and slopes together.
FF(gdpSUTSE) <- FF(uni) %x% diag(3)
GG(gdpSUTSE) <- GG(uni) %x% diag(3)
# ’CLEAN’ the system variance
W(gdpSUTSE)[] <- 0
#
# Define a “build” function to be given to MLE routine
buildSUTSE <- function(psi) {
 # Note log-Cholesky parametrization of the covariance matrix W
 U <- matrix(0, nrow = 3, ncol = 3)
 # putting random element in the upper triangle
 U[upper.tri(U)] <- psi[1 : 3]
 # making sure that the diagonal element of U are positive
 diag(U) <- exp(0.5 * psi[4 : 6])
 # Constructing the matrix W as the cross product of the U - equivalent to t(U) %*% U
 W(gdpSUTSE)[4 : 6, 4 : 6] <- crossprod(U)
 # parametrization of the covariance matrix V
 diag(V(gdpSUTSE)) <- exp(0.5 * psi[7 : 9])
 return(gdpSUTSE)
 }
#
# MLE estimate
gdpMLE <- dlmMLE(gdp, rep(-2, 9), buildSUTSE, control = list(maxit = 1000))
# Checking convergence
gdpMLE$conv
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Building the optimal model
gdpSUTSE <- buildSUTSE(gdpMLE$par)
{% endhighlight %}

Note the log-Cholesky parametrization of the covariance matrix \\(W\\) ([Pinheiro, 1994](#bib-JC)), in the _buildSUTSE_ function. Looking at the estimated variance/covariance matrices:


{% highlight r %}
W(gdpSUTSE)[4 : 6, 4 : 6]
{% endhighlight %}



{% highlight text %}
##         [,1]    [,2]    [,3]
## [1,] 0.06080 0.02975 0.03246
## [2,] 0.02975 0.01756 0.02609
## [3,] 0.03246 0.02609 0.05200
{% endhighlight %}



{% highlight r %}
# correlations (slopes)
cov2cor(W(gdpSUTSE)[4 : 6, 4 : 6])
{% endhighlight %}



{% highlight text %}
##        [,1]   [,2]   [,3]
## [1,] 1.0000 0.9104 0.5773
## [2,] 0.9104 1.0000 0.8634
## [3,] 0.5773 0.8634 1.0000
{% endhighlight %}



{% highlight r %}
# observation standard deviations
sqrt(diag(V(gdpSUTSE)))
{% endhighlight %}



{% highlight text %}
## [1] 0.04836 0.14746 0.12240
{% endhighlight %}



{% highlight r %}
#
# Looking at the smoothed series
gdpSmooth <- dlmSmooth(gdp, gdpSUTSE)
{% endhighlight %}
The first three columns of `gdpSmooth$s` contain the time series of _smoothed GDP level_, the last three the _smoothed slopes_

###### Plotting the results


{% highlight r %}
library(ggplot2)
library(reshape2)
# Preparing the data
# 1. Remove the first obs from the smoothed gdp, 
#    since the time series, s, starts one time unit before the first observation.
level.df <- data.frame(year = index(gdp), gdpSmooth$s[-1,1:3])
# 2. naming the variable before melting the data
names(level.df)[2:4] <- colnames(gdp)
# 3. Melting the dataframe
level.df.melt = reshape2::melt(level.df, id=c("year"))
# Plotting the level
p <- ggplot(level.df.melt) + facet_grid(facets = variable ~ ., scales = "free_y") 
p <- p + geom_line(aes(x=year, y=value, colour=variable)) + scale_colour_manual(values=myColor)
p <- p + labs(x = "year", y = expression(mu[t]), title = "The 'Smoothed' value of level of the state variable \n for log of the GDP per Capita")
p <- p + theme(legend.position="none")
print(p)
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/paramGDPs1.png) 

{% highlight r %}
# Plotting the rate of change of the smoothed GDP per capita
slope.df <- data.frame(year = index(gdp), gdpSmooth$s[-1,4:6])
names(slope.df)[2:4] <- colnames(gdp)
slope.df.melt = reshape2::melt(slope.df, id=c("year"))
p <- ggplot(slope.df.melt) + facet_grid(facets = variable ~ ., scales = "free_y") 
p <- p + geom_line(aes(x=year, y=value, colour=variable)) + scale_colour_manual(values=myColor)
p <- p + labs(x = "year", y = expression(delta[t]), title = "The 'Smoothed' value of Rate of Change of state variable \n for the log of the GDP per Capita")
p <- p + theme(legend.position="none")
print(p)
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/paramGDPs2.png) 

### <a name="topic6"></a> Seemingly Unrelated Regression models
Building on top of SUTSE, the next example we will consider is the multivariate version of the dynamic Capital Asset Pricing Model (CAPM) for \\(m\\) assets. 

Let \\({r_t}^{\,j} ,{r_t}^{\, M}\\) and \\({r_t}^{\, f}\\) be the returns at time \\(t\\) of asset, \\(j\\), under study, of the market, \\(M\\), and of a risk-free asset, \\(f\\), respectively. Define the excess returns of the asset, \\(j\\) as \\( y\_{j,t} = r_\{j,t\} − r_t^{\, f}\\) and the market’s excess returns as \\(x_t = r_t^\{\,M\} − r_t^\{\,f\}\\); where \\(j = 1, \dots , m\\). Then:

\\[
\begin{align}
y\_{j,t} &= \alpha\_{j,t} + \beta\_{j,t} \, x_t + \upsilon\_{j,t} \\\\
\alpha\_{j,t} &= \alpha\_{j,t-1} \qquad + \omega_\{j,t\}^\alpha \\\\
\beta\_{j,t} &= \beta\_{j,t-1} \qquad  + \omega\_{j,t}^\beta \\\\
\end{align}
\\]

where it is sensible to assume that the intercepts and the slopes are correlated across the $m$ stocks. The above model can be rewritten in a more general form such as

\\[
\begin{align}
\mathbf{y_t} &= (F_t \otimes I_m) \, \theta + \mathbf{\upsilon_t} \quad &\mathbf{\upsilon_t} \sim N(0,V)\\\\
\theta_t &= (G \otimes I_m) \, \theta_{t-1} + \mathbf{\omega_t} \quad &\mathbf{\omega_t} \sim N(0,W)\\\\
\end{align}
\\]

where

* \\(\mathbf{y_t} = [y\_{1t}, \dots , y\_{mt}] \\) are the values of the _m_ different \\(y\\) series
* \\(\theta_t = [\alpha\_{1t}, \dots, \alpha\_{mt}, \beta\_{1p}, \dots, \beta\_{mt}]\\)
* \\(\mathbf{\upsilon_t} = [\upsilon\_{1t}, \dots, \upsilon\_{mt}]\\) and \\(\mathbf{\omega_t} = [\omega\_{1t}, \dots, \omega\_{mt}]\\) are _iid_
* \\(W = \textrm{blockdiag}(W\_{\alpha}; W\_{\beta)}\\)
* \\(F_t = [1 \, x_t]\\); and \\(G = \textrm{I}_2\\)

##### <a name="topic6.1"></a> Example 7: CAPM regression model

We assume for simplicity that the \\(\alpha\_{j,t}\\) are time-invariant, which amounts to assuming that \\(W\_{\alpha} = 0\\).


{% highlight r %}
data <- ts(read.table("http://shazam.econ.ubc.ca/intro/P.txt", header = TRUE), 
           start = c(1978, 1), frequency = 12)
data <- data * 100
y <- data[, 1 : 4] - data[, "RKFREE"]
colnames(y) <- colnames(data)[1 : 4]
market <- data[, "MARKET"] - data[, "RKFREE"]
m <- NCOL(y)

# Define a “build” function to be given to MLE routine
buildSUR <- function(psi) {
  ### Set up the model
  CAPM <- dlmModReg(market, addInt = TRUE)
  CAPM$FF <- CAPM$FF %x% diag(m)
  CAPM$GG <- CAPM$GG %x% diag(m)
  CAPM$JFF <- CAPM$JFF %x% diag(m)
  # specifying the mean of the distribution of the initial state of theta (alpha; beta)
  CAPM$m0 <- c(rep(0,2*m))
  # Specifying the variance of the distribution of the initial state of theta (alpha; beta)
  CAPM$C0 <- CAPM$C0 %x% diag(m)
  # ’CLEAN’ the system and observation variance-covariance matrices
  CAPM$V <- CAPM$V %x% matrix(0, m, m)
  CAPM$W <- CAPM$W %x% matrix(0, m, m)
  #
  # parametrization of the covariance matrix W
  U <- matrix(0, nrow = m, ncol = m)
  # putting random element in the upper triangle
  U[upper.tri(U)] <- psi[1 : 6]
  # making sure that the diagonal element of U are positive
  diag(U) <- exp(0.5 * psi[7 : 10])
  # Constructing the matrix W_beta as the cross product of the U - equivalent to t(U) %*% U
  # Assuming W_alpha is zero
  W(CAPM)[5 : 8, 5 : 8] <- crossprod(U)
  # parametrization of the covariance matrix V
  U2 <- matrix(0, nrow = m, ncol = m)
  # putting random element in the upper triangle
  U2[upper.tri(U2)] <- psi[11 : 16]
  # making the diagonal element positive
  diag(U2) <- exp(0.5 * psi[17 : 20])
  V(CAPM) <- crossprod(U2)
  return(CAPM)
}

# MLE estimate
CAPM_MLE <- dlmMLE(y, rep(1, 20), buildSUR, control = list(maxit = 500))
# Checking convergence
CAPM_MLE$conv
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Build the model
CAPM_SUR <- buildSUR(CAPM_MLE$par)
# Looking at the estimated var-covariance of betas
W(CAPM_SUR)[5 : 8, 5 : 8]
{% endhighlight %}



{% highlight text %}
##           [,1]      [,2]      [,3]      [,4]
## [1,] 7.243e-06 9.768e-05 0.0001299 0.0002026
## [2,] 9.768e-05 1.326e-03 0.0017641 0.0027510
## [3,] 1.299e-04 1.764e-03 0.0023573 0.0036668
## [4,] 2.026e-04 2.751e-03 0.0036668 0.0057297
{% endhighlight %}



{% highlight r %}
# Viewing corr
cov2cor(W(CAPM_SUR)[5 : 8, 5 : 8])
{% endhighlight %}



{% highlight text %}
##        [,1]   [,2]   [,3]   [,4]
## [1,] 1.0000 0.9966 0.9943 0.9944
## [2,] 0.9966 1.0000 0.9976 0.9979
## [3,] 0.9943 0.9976 1.0000 0.9977
## [4,] 0.9944 0.9979 0.9977 1.0000
{% endhighlight %}



{% highlight r %}
# observation Variance Covariance
V(CAPM_SUR)
{% endhighlight %}



{% highlight text %}
##          [,1]     [,2]    [,3]   [,4]
## [1,] 41.03769 -0.01104 -0.9997 -2.389
## [2,] -0.01104 24.23393  5.7894  3.411
## [3,] -0.99971  5.78936 39.2327  8.161
## [4,] -2.38871  3.41080  8.1609 39.347
{% endhighlight %}



{% highlight r %}
# observation correlation
cov2cor(V(CAPM_SUR))
{% endhighlight %}



{% highlight text %}
##            [,1]       [,2]     [,3]     [,4]
## [1,]  1.0000000 -0.0003502 -0.02491 -0.05945
## [2,] -0.0003502  1.0000000  0.18776  0.11046
## [3,] -0.0249149  0.1877564  1.00000  0.20771
## [4,] -0.0594452  0.1104562  0.20771  1.00000
{% endhighlight %}



{% highlight r %}
# observation standard deviations
sqrt(diag(V(CAPM_SUR)))
{% endhighlight %}



{% highlight text %}
## [1] 6.406 4.923 6.264 6.273
{% endhighlight %}



{% highlight r %}
#
# Looking at the smoothed series
CAPM_Smooth <- dlmSmooth(y, CAPM_SUR)
#
# Plotting the result
myColor <- wes.palette(4, "GrandBudapest")
yr_seq <- seq(from = as.Date("1978-01-01"),to = as.Date("1987-12-01"), by = "months")
beta.df <- data.frame(year = yr_seq, CAPM_Smooth$s[-1, m + (1 : m)])
names(beta.df)[2:5] <- colnames(y)
beta.df.melt = reshape2::melt(beta.df, id=c("year"))
p <- ggplot(beta.df.melt) # + facet_grid(facets = variable ~ ., scales = "free_y") 
p <- p + geom_line(aes(x=year, y=value, colour=variable)) + scale_colour_manual(values=myColor, name = "Stocks:")
p <- p + labs(x = "year", y = expression(beta[t]), title = expression(paste("The 'Smoothed' value of the ", beta, " in the CAPM model")))
p <- p + theme(legend.position="bottom" , axis.title.y  = element_text(size=14))
print(p)
{% endhighlight %}

![center]({{ BASE_PATH }}/figs/dlmFigures/multiCAPM.png) 

### <a name="topic7"></a> Dynamic Common Factors Model

**Note:** In this section we will use the parameter \\(x\\) instead of \\(\theta\\) to indicate the state since it's more common to use \\(x\\) as a variable in this kind of analysis.

1.  Let \\(y_t\\) be an _m-dimensional_ observation that can be modeled, at any time \\(t\\), as  a linear combination
of the factor components, \\(z\\), subject to additive observation noise. So:
\\[
y_t  =  A  \, z_t  + \upsilon_t \qquad \upsilon_t \sim N_m(0,V)
\\]

2.  Let \\(z_t\\) be a _q-dimensional_ vector of _factors_ that evolves according to a DLM:
\\[
\begin{align}
z_t &= F_0 \, x_t   + \varepsilon_t \qquad & \varepsilon_t \sim N_q(0,\Sigma) \\\\ 
x_t &= G \, x_\{t-1\} + \omega_t \qquad & \omega_t \sim N_p(0,W) \\\\
\end{align}
\\]

3.  Setting \\(\Sigma = 0\\) implies that \\(\varepsilon_t = 0\\) for every \\(t\\) and then we have

\\[
\begin{align}
y_t &=  A \, F_0 \, x_t + \upsilon_t \qquad &\upsilon_t \sim N_m(0,V) \\\\ 
x_t &=  \, \, G \, x_\{t-1\}  + \omega_t \qquad & \omega_t\sim N_p(0,W) \\\\
\end{align}
\\]

This is above system of equation is nothing else by a _time-invariant_ DLM where the matrix \\(F = A \, F_0\\); where \\(F\\) is an \\(m \times p\\) matrix, \\(A\\) is a \\(m \times q\\) matrix and \\(F_0\\) is an \\(q \times p\\) matrix.

**Notes that:**

* Typically, \\(m\\) > \\(q\\) and \\(m \le p\\)
* The filtering/smoothing estimates of factors, can be calculated from the filtering/smoothing estimates of the states \\(x_t\\) through the following formula
\\[
\hat{z_t} = F_0 \, \hat{x_t}
\\]

where
\\[
\hat{x_t} = 
\begin{cases} 
E(x_t|y\^{t}) \qquad \, \textrm{for filtered estimates} \\\\
E(x_t|y\^{T}) \qquad    \textrm{for smoothed estimates} \\\\
\end{cases}
\\]

* The Variance/Co-variance matrix of the factors, given the observation can be calculated from the variance/covariance matrix of the state variable given the observation through the following formula:

\\[
\begin{align}
Var(z_t|y\^{t)} &= F_0 \, Var(x_t|y\^t) \, F_0^\prime =  F_0 \, C_t \, F_0\^\prime \\\\
Var(z_t|y\^{T}) &= F_0 \, Var(x_t|y\^T) \, F_0^\prime =  F_0 \, S_t \, F_0\^\prime \\\\
\end{align}
\\]

* As in standard factor analysis, in order to achieve identifiability of the unknown parameters, some constraints have to be imposed. The assumption that we will make ([Petris, Petrone, and Campagnoli, 2009](#bib-dlm2))
\\[
A_\{i \, j\} = 
\begin{cases} 
0 \quad \textrm{if} \, \, i < j \\\\
1 \quad \textrm{if} \, \, i = j \\\\
\end{cases}
\\]
where \\(i = 1, \dots, m\\); and \\(j = 1, \dots, q\\)

##### <a name="topic7.1"></a> Numerical example:

We will use the GDP data, used in example 6 above, and consider two factor models: one where the dimension, \\(q\\) is \\(1\\) and another where the dimension \\(q = 2\\). We will describe the dynamics of the GDP time series of Germany, UK, and USA; so \\(m = 3\\) in this case. The factor(s) in these two models is (are) modeled by _local linear trend_ model. Recall, that in a 1-dimensional DLM that is modeled as a local linear trend, the number of parameters, \\(p\\) is 2 (see example 3 above).

As a result of this choice of models:

* In the case of the 1-factor model, the system and observation variances \\(V\\) and \\(W\\) are assumed to be diagonal of order 3 and 2, respectively
* In the case of the 2-factor model, the system and observation variances \\(V\\) and \\(W\\) are assumed to be diagonal of order 3 and 4 (2 parameters x 2 factors), respectively.

Furthermore, we will assume that the factors are independent of each-other; therefore since they are normally distributed, their independence implies 0 linear correlation. We will also assume that, for each factor, the correlation between their level \\(\delta_t\\) and their slope \\(\mu_t\\) is 0. In Essence the matrix \\(W\\) is diagonal

###### 1-factor model

We start this example with the estimation of the loading in the case of 1-factor model.


{% highlight r %}
unloadNamespace("ggplot2")
# Getting the data
gdp0 <- read.table("http://definetti.uark.edu/~gpetris/UseR-2011/gdp.txt", skip = 3, header = TRUE)
# Transforming the data to the log-scale
gdp <- ts(10 * log(gdp0[, c("GERMANY", "UK", "USA")]), start = 1950)
# Create Generic model
uni <- dlmModPoly() # Default paramters --> stochastic linear trend.
# some parameters definition
m <- dim(gdp)[2] # number of series/country's GDP to model
q <- 1           # number of factors
p <- 2           # number of parameters in a local linear trend model

buildFactor1 <- function(psi) {
  # 1 Factor model:
  # 
  # Constructing matrix A of factors loading.
  A <- matrix(c(1, psi[1], psi[2]), ncol = q)
  F0 <- FF(uni)
  FF <-  A %*% F0
  GG <- GG(uni)
  # Diagonal element of system variance W
  W <- diag(exp(0.5 * psi[3:4]), ncol = p * q)
  # log-Cholesky parametrization of the covariance matrix V
  U <- matrix(0, nrow = m, ncol = m)
  # putting random element in the upper triangle
  U[upper.tri(U)] <- psi[5:7]
  # making sure that the diagonal element of U are positive
  diag(U) <- exp(0.5 * psi[8 : 10])
  # Constructing the matrix V as the cross product of the U - equivalent to t(U) %*% U
  V <- crossprod(U)
  gdpFactor1 <- dlm(FF = FF, V = V, GG = GG, W = W, m0 = rep(0, p * q), C0 = 1e+07 * diag(p * q))
  return(gdpFactor1)
}

# MLE estimate
gdpFactor1_MLE <- dlmMLE(gdp, rep(-2, 10), buildFactor1, control = list(maxit = 1000))
# Checking convergence
gdpFactor1_MLE$conv
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Factor loading
(A <- matrix(c(1, gdpFactor1_MLE$par[1:2]), ncol = q))
{% endhighlight %}



{% highlight text %}
##        [,1]
## [1,] 1.0000
## [2,] 0.1790
## [3,] 0.7106
{% endhighlight %}

###### 2-factors model

Next we consider the 2-factors model:


{% highlight r %}
unloadNamespace("ggplot2")
# Getting the data
gdp0 <- read.table("http://definetti.uark.edu/~gpetris/UseR-2011/gdp.txt", skip = 3, header = TRUE)
# Transforming the data to the log-scale
gdp <- ts(10 * log(gdp0[, c("GERMANY", "UK", "USA")]), start = 1950)
# Create Generic model
uni <- dlmModPoly() # Default paramters --> stochastic linear trend.
# some parameters definition
m <- dim(gdp)[2] # number of series/country's GDP to model
q <- 2           # number of factors
p <- 2           # number of parameters in a local linear trend model

buildFactor2 <- function(psi) {
  # 2 Factors model:
  # 
  # Constructing matrix A of factors loading.
  a1 <- c(1, psi[1], psi[2])
  a2 <- c(0, 1, psi[3])
  A <- matrix(cbind(a1,a2), ncol = q)
  F0 <- FF(uni %+% uni)
  FF <-  A %*% F0
  GG <- GG(uni %+% uni)
  # Diagonal element of system variance W
  W <- diag(exp(0.5 * psi[4:7]), ncol = p * q)
  # log-Cholesky parametrization of the covariance matrix V
  U <- matrix(0, nrow = m, ncol = m)
  # putting random element in the upper triangle
  U[upper.tri(U)] <- psi[8:10]
  # making sure that the diagonal element of U are positive
  diag(U) <- exp(0.5 * psi[11 : 13])
  # Constructing the matrix V as the cross product of the U - equivalent to t(U) %*% U
  V <- crossprod(U)
  gdpFactor2 <- dlm(FF = FF, V = V, GG = GG, W = W, m0 = rep(0, p * q), C0 = 1e+07 * diag(p * q))
  return(gdpFactor2)
}

# MLE estimate
gdpFactor2_MLE <- dlmMLE(gdp, rep(-2, 13), buildFactor2, control = list(maxit = 3000), method = "Nelder-Mead")
# Checking convergence
gdpFactor2_MLE$conv
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}



{% highlight r %}
# Calculating factor loading
a1 <- c(1,gdpFactor2_MLE$par[1:2])
a2 <- c(0, 1, gdpFactor2_MLE$par[3])
(A <- matrix(cbind(a1,a2), ncol = q))
{% endhighlight %}



{% highlight text %}
##         [,1]    [,2]
## [1,]  1.0000  0.0000
## [2,]  1.7495  1.0000
## [3,] -0.7584 -0.9348
{% endhighlight %}

###### Model Comparaison

Finally we  use the Akaike information Criterion to compare the fit of SUTSE, 1-factor and 2-factor models to the GDP data.


{% highlight r %}
# Calculating AIC
logLik <- N <- AIC <- c()
# Log Likehood
logLik["SUTSE"] <-   -gdpMLE$value
logLik["1Factor"] <- -gdpFactor1_MLE$value
logLik["2Factors"] <- -gdpFactor2_MLE$value
# Number of Parameters
N["SUTSE"] <- length(-gdpMLE$par)
N["1Factor"] <- length(gdpFactor1_MLE$par)
N["2Factors"] <- length(gdpFactor2_MLE$par)
# AIC calculation
AIC <- -2 * (logLik - N)
print(AIC)
{% endhighlight %}



{% highlight text %}
##    SUTSE  1Factor 2Factors 
##   -57.14    61.82    91.77
{% endhighlight %}

Based on the AIC, the _SUTSE_ model seems to fit the data best.

### Conclusion

In this post, we have covered the topics of linear state space model (and the corresponding dynamical linear model) that are governed by Gaussian innovations. We have looked at how to construct such model in R, how to extend them from the univariate case to the multivariate case and how to estimate the model parameters using the MLE method. 

In the upcoming post, we will cover the Bayesian formulation of such model, the usage of Monte Carlo Methods to estimates the model parameters.



### References
 
Citations made with `knitcitations` ([Boettiger, 2014](https://github.com/cboettig/knitcitations)).
 
 
<a name=bib-knitcitations></a>[[1]](#cite-knitcitations) C.
Boettiger. _knitcitations: Citations for knitr markdown files_. R
package version 1.0-1. 2014. URL:
[https://github.com/cboettig/knitcitations](https://github.com/cboettig/knitcitations).

<a name=bib-greycite17728></a>[[2]](#cite-greycite17728) Pacomet.
_Add-legend-to-ggplot2-line-plot_. 2012. URL:
[http://stackoverflow.com/questions/10349206/add-legend-to-ggplot2-line-plot](http://stackoverflow.com/questions/10349206/add-legend-to-ggplot2-line-plot).

<a name=bib-dlm1></a>[[3]](#cite-dlm1) G. Petris. "An R Package
for Dynamic Linear Models". In: _Journal of Statistical Software_
36.12 (2010), pp. 1-16. URL:
[http://www.jstatsoft.org/v36/i12/](http://www.jstatsoft.org/v36/i12/).

<a name=bib-greycite17729></a>[[4]](#cite-greycite17729) G.
Petris. _State Space Models in R - useR2011 tutorial_. 2011. URL:
[http://definetti.uark.edu/~gpetris/UseR-2011/SSMwR-useR2011handout.pdf](http://definetti.uark.edu/~gpetris/UseR-2011/SSMwR-useR2011handout.pdf).

<a name=bib-dlm2></a>[[5]](#cite-dlm2) G. Petris, S. Petrone and
P. Campagnoli. _Dynamic Linear Models with R_. useR!
Springer-Verlag, New York, 2009.

<a name=bib-JC></a>[[6]](#cite-JC) J. C. Pinheiro. "Topics in
Mixed Effects Models". PhD thesis. University Of Wisconsin –
Madison, 1994.

<a name=bib-greycite17709></a>[[7]](#cite-greycite17709) L.
Saravia. _R Plotting confidence bands with ggplot_. 2012. URL:
[http://stackoverflow.com/questions/14033551/r-plotting-confidence-bands-with-ggplot2](http://stackoverflow.com/questions/14033551/r-plotting-confidence-bands-with-ggplot2).
 
### Reproducibility
 

{% highlight r %}
sessionInfo()
{% endhighlight %}



{% highlight text %}
## R version 3.1.1 (2014-07-10)
## Platform: x86_64-apple-darwin10.8.0 (64-bit)
## 
## locale:
## [1] en_CA.UTF-8/en_CA.UTF-8/en_CA.UTF-8/C/en_CA.UTF-8/en_CA.UTF-8
## 
## attached base packages:
## [1] parallel  stats     utils     graphics  grDevices datasets  methods  
## [8] base     
## 
## other attached packages:
##  [1] ggplot2_1.0.0              depmixS4_1.3-2            
##  [3] Rsolnp_1.14                truncnorm_1.0-7           
##  [5] MASS_7.3-33                nnet_7.3-8                
##  [7] bibtex_0.3-6               reshape2_1.4              
##  [9] numDeriv_2012.9-1          knitr_1.6                 
## [11] knitcitations_1.0-1        wesanderson_0.3.2         
## [13] PerformanceAnalytics_1.1.4 xts_0.9-7                 
## [15] zoo_1.7-11                 dlm_1.1-3                 
## 
## loaded via a namespace (and not attached):
##  [1] blotter_0.8.19   codetools_0.2-8  colorspace_1.2-4 devtools_1.5    
##  [5] digest_0.6.4     evaluate_0.5.5   formatR_0.10     grid_3.1.1      
##  [9] gtable_0.1.2     htmltools_0.2.4  httr_0.3         labeling_0.2    
## [13] lattice_0.20-29  lubridate_1.3.3  memoise_0.2.1    munsell_0.4.2   
## [17] plyr_1.8.1       proto_0.3-10     quantstrat_0.8.2 Rcpp_0.11.2     
## [21] RCurl_1.95-4.1   RefManageR_0.8.2 RJSONIO_1.2-0.2  rmarkdown_0.2.50
## [25] scales_0.2.4     stats4_3.1.1     stringr_0.6.2    tools_3.1.1     
## [29] whisker_0.3-2    XML_3.98-1.1     yaml_2.1.13
{% endhighlight %}
