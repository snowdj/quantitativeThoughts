install.packages("FRBData")
library(FRBData)
# This package provides functions which can get financial and economical data from Federal Reserve Bank's website
# (http://www.federalreserve.gov/releases/h15/data.htm)
# See the footnote on the website to determine the day count convention and the compounding frequency of IR curves

#Swap rate
ir_swp_rates <- GetInterestRates(id = "SWAPS",from = as.Date("2014/06/11"), to = as.Date("2014/06/30"))

#Constant Maturity Treasury rate

# http://www.treasury.gov/resource-center/faqs/Interest-Rates/Pages/faq.aspx
# CMT yields are read directly from the Treasury's daily yield curve and represent "bond equivalent yields" for securities that pay semiannual interest, which are expressed on a simple annualized basis.
# As such, these yields are not effective annualized yields or Annualized Percentage Yields (APY), which include the effect of compounding.  To convert a CMT yield to an APY you need to apply the standard financial formula:
# APY = (1 + CMT/2)^2 -1
# Treasury does not publish the weekly, monthly or annual averages of these yields.  However, the Board of Governors of the Federal Reserve System also publishes these rates on our behalf in their Statistical Release H.15.  The web site for the H.15 includes links that have the weekly, monthly and annual averages for the CMT indexes


# Constant maturity yield curve
ir_const_maturities <- GetInterestRates(id = "TCMNOM",lastObs = 10)


# Plotting a term structure for a specific day
# http://blog.revolutionanalytics.com/2014/06/quantitative-finance-applications-in-r-6-constructing-a-term-structure-of-interest-rates-using-r-par.html

require(lubridate)
require(ggplot2)
i <- 10
term_str <- t(ir_const_maturities[i, ])
ad <- index(ir_const_maturities)[i]

# Plotting the term structure
df.tmp <- data.frame(term = 1:length(ir_const_maturities[i, ]), rates  = t(coredata(ir_const_maturities[i, ])))
gplot <- ggplot(data = df.tmp, aes(x = term, y = rates)) + geom_line() +geom_point()
gplot <- gplot + scale_x_discrete(breaks=df.tmp$term, labels=colnames(ir_const_maturities))
gplot <- gplot + ggtitle(paste("Market Zero Rates for:", ad)) + xlab("Term") + ylab("Rates (%)")
gplot

# Constructing a yield curve ------
# as described in the following post
# http://blog.revolutionanalytics.com/2014/06/quantitative-finance-applications-in-r-6-constructing-a-term-structure-of-interest-rates-using-r-par.html

term.str.dt <- c()
for (j in colnames(ir_const_maturities)) {
  term_num <- as.numeric(substring(j, 1, nchar(j) - 1))
  term <- switch(tolower(substring(j, nchar(j))), 
                 "y" = years(term_num),
                 "m" = months(term_num),
                 "w" = weeks(term_num),
                 stop("Unit unrecognized in the term structure file"))
  term.str.dt <- cbind(term.str.dt, ad + term)
}
term.str.dt <- as.Date(term.str.dt, origin = "1970-01-01")

# Construction of the most recent zero curve ----
require(xts)
ad <- index(last(ir_const_maturities))
rates <- t(coredata(last(ir_const_maturities))) * 0.01 # converting the decimal from percentage
# Make sure that the data is monotone - it's NOT always the case, since these data are weekly averages and not directly observed
if (all(rates == cummax(rates))) { 
  # Checks if the data is montonoe, if not fix it
  # the above condition is a fast check for increasing monotonicity. for decreasin, one will use cummin instead of cummax
  # http://stackoverflow.com/questions/13093912/how-to-check-if-a-sequence-of-numbers-is-monotonically-increasing-or-decreasing
} else {
  for (t in 2:length(rates)) {
    if (rates[t] < rates[t-1]) rates[t] = rates[t-1]
  }
}

zcurve <- as.xts(x = rates, order.by = term.str.dt)

seq.date <- seq.Date(from = index(last(ir_const_maturities)), to = last(index(zcurve)), by = 1)
term.str.dt.daily <- xts(order.by= seq.date)

zcurve <- merge(zcurve, term.str.dt.daily)
zcurve_linear <- na.approx(zcurve) # Linear interpolation
zcurve_spline <- na.spline(zcurve, method = "hyman") # Spline interpolation which guarentees monotone output
colnames(zcurve_spline) <- NULL
zcurve_spline <- na.locf(zcurve_spline, na.rm = FALSE, fromLast = TRUE) # flat extrapolation to the left

# Plotting the curve using ggplot2 ---

df.tmp <- data.frame(term = index(zcurve_spline), rates = coredata(zcurve_spline))
gplot <- ggplot(data = df.tmp, aes(x = term, y = rates)) + geom_line()
gplot <- gplot + scale_x_date(labels = date_format())
gplot <- gplot + ggtitle(paste("Market Zero Rates for:", ad)) + xlab("Term") + ylab("Rates (%)")
gplot

# Comparison to the default spline method ----
zcurve_spline_default <- na.spline(zcurve)
colnames(zcurve_spline_default) <- NULL
zcurve_spline_default <- na.locf(zcurve_spline_default, na.rm = FALSE, fromLast = TRUE) # flat extrapolation to the left


# Plotting both curves on the same graph

# Step 1: Merge the two xts
zcurve_grp <- merge(zcurve_spline, zcurve_spline_default)
# Graphing the quick way - autoplot is part of the zoo package; and is intended to be plot with  ggplot
# http://www.inside-r.org/packages/cran/zoo/docs/autoplot.zoo
autoplot(zcurve_grp)                        ## multiple without color/linetype
autoplot(zcurve_grp, facets = NULL)         ## multiple with series-dependent color/linetype
autoplot(zcurve_grp, facets = Series ~ .)   ## single with series-dependent color/linetype

# Doing it manually
# Step 2: Metlting the object

# fortify.zoo returns a data.frame either in long format (melt = TRUE) or in wide format (melt = FALSE). The long format has three columns: the time Index, a factor indicating the Series, and the corresponding Value. The wide format simply has the time Index plus all columns of coredata(model)
zcurve_grp_mlt <- fortify(model = zcurve_grp, melt = TRUE)

# Doing it the really manual way using reshape2 package.
require(reshape2)
zcurve_grp_tmp <- data.frame(term = index(zcurve_grp), coredata(zcurve_grp))
zcurve_grp_tmp_mlt <- melt(zcurve_grp_tmp, id.vars = 'term')

all.equal(zcurve_grp_tmp_mlt, zcurve_grp_mlt, check.attributes = FALSE, tolerance = .Machine$double.eps)

# Step 3: Graphing the melted object

ggplot(zcurve_grp_mlt, aes(x=Index, y=Value, colour=Series)) + geom_path()
ggplot(zcurve_grp_tmp_mlt, aes(x=term, y=value, colour=variable)) + geom_path()

# Construction of the forward curves and taking Day Count Convention into account
# Credit to the following blog post
# http://blog.revolutionanalytics.com/2014/07/quantitative-finance-applications-in-r-7-constructing-a-term-structure-of-interest-rates-using-r-par.html

# calculating the fraction of the year based on Act/365 day count convention
delta_t_Act365 <- function(from_date, to_date){
  if (from_date > to_date) 
    stop("the 2nd parameters(to_date) has to be larger/more in the future than 1st paramters(from_date)")
  yearFraction <- as.numeric((to_date - from_date)/365)
  return(yearFraction)  
}
# Computing the forward discount factor
fwdDiscountFactor <- function(t0, from_date, to_date, zcurve.xts, dayCountFunction) {
  if (from_date > to_date) 
    stop("the 2nd parameters(to_date) has to be larger/more in the future than 1st paramters(from_date)")
  rate1 <- as.numeric(zcurve.xts[from_date])      # R(0, T1)
  rate2 <- as.numeric(zcurve.xts[to_date])        # R(0, T2)
  # Computing discount factor
  discFactor1 <- exp(-rate1 * dayCountFunction(t0, from_date))
  discFactor2 <- exp(-rate1 * dayCountFunction(t0, to_date))
  
  fwdDF <- discFactor2/discFactor1
  return(fwdDF)
}
# Computing the forward rate
fwdInterestRate <- function(t0, from_date, to_date, zcurve.xts, dayCountFunction) {
  # we are passing the zero curve, because we will compute the discount factor inside this function
  if (from_date > to_date) 
    stop("the 2nd parameters(to_date) has to be larger/more in the future than 1st paramters(from_date)")
  else if (from_date == to_date)
    return(0)
  else {
    fwdDF <- fwdDiscountFactor(t0, from_date, to_date, zcurve.xts, dayCountFunction)
    fwdRate <- -log(fwdDF)/dayCountFunction(from_date, to_date)
  }
  return(fwdRate)
}

# Example ----

t0 <- index(last(ir_const_maturities))

fwDisc <- fwdDiscountFactor(t0, from_date = t0 + years(5), to_date = t0 + years(10), 
                            zcurve.xts = zcurve_spline, dayCountFunction = delta_t_Act365)

fwrate <- fwdInterestRate(t0, from_date = t0 + years(5), to_date = t0 + years(10), 
                          zcurve.xts = zcurve_spline, dayCountFunction = delta_t_Act365)

# Test 1: Trivial case:

fwdDiscountFactor(t0, from_date = t0 + years(5), to_date = t0 + years(5), 
                  zcurve.xts = zcurve_spline, dayCountFunction = delta_t_Act365)

fwdInterestRate(t0, from_date = t0 + years(5), to_date = t0 + years(5), 
                zcurve.xts = zcurve_spline, dayCountFunction = delta_t_Act365)

# Test 2: Recovering Original interest Rate if we set the start date to equal t0

tt1 <- fwdInterestRate(t0, from_date = t0, to_date = t0 + months(1), 
                       zcurve.xts = zcurve_spline, dayCountFunction = delta_t_Act365)
tt2 <- coredata(last(ir_const_maturities))[,"1M"] * 0.01 # since the orginial was in decimal format

# Default tolerance is passed. 2e-13 is accurate enough
all.equal(tt1, tt2, check.attributes = FALSE, tolerance = .Machine$double.eps)