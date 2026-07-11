# HAR Modeling

Using HAR Models to predict and model Volatility

### AR Models 
Auto Regressive models are used to forecast **Realised variance ($RV$)**. The Simplest one is $AR(1)$ model:

$$ RV_t =\beta_0 +\beta_1 RV_{t-1} $$

and in general $AR(p)$ in a autoregressive model to order $p$. (I am assuming no noise we can add a noise term as well)
$$ RV_T =\beta_0 +\sum_{i=0}^p\beta_i RV_{T-i} .$$

We estimate the $\beta$ parameters using data say for the $0, 1, \dots, T$ to predict the variance (and volatility) at time $T$.

## HAR Model

Prof Corsi idea that forecasting might actually only depend on the volatility over the last week and month. So essentially HAR is a AR(22) model but with some stronger constarins on the coeeficients. 

$$ \boxed{RV_{t} = \beta_0 + \beta_1 RV_{t-1} + \beta_2 RV_{t-1}^w +\beta_3 RV_{t-1}^m}$$
where the variables, 
$$ RV_{t-1}^w = \frac{1}{5} \sum _{i=1}^5 RV_{t-i}$$

and, 

$$ RV_{t-1}^m = \frac{1}{22} \sum _{i=1}^{22} RV_{t-i}. $$

$RV_{t-1}^w$ and $RV_{t-1}^m$ are the average daily valatility over the last week and month (5 and 22 are the average trading days in a week and month).

The issue is yfinance has granuar data like 5 min (anything less than 1 day) only for the last 60 days. The major issues with this is:
- We were targeting older timeperiods 2020-2022, 2014-2016, so its an issue getting granular data. 
- even using this as an estimate to predict variance and volatility for tommorrow, is not reliable as when fitting the Linear regression will only have like 30 or 40 data points (will have to remove the first 22), so the fit is also not good. 

One alternative is not look for other sources to get more granualar data, a more easier option is to use other indecators of variance like Parkinson, Garman-Klass Volatility analysis. Instead of using 5 min or 15 min data to predict Intraday volatility it uses Open, Close, High, and Low to predict the daily var. So need much less data and very efficient at predicting variance. 

### Garman-Klass variance
The (daily) variance using the Garman-Klass method
$$\boxed{RV (GK) = \frac{1}{2} \ln\left(\frac{H}{L}\right)^2 - (2\ln(2) -1)\ln \left(\frac{C}{O}\right)},$$
where, $H, L, C, O$ are the daily High, Low, Close, and Open prices. (The yfinance data contains all for daily data). 

Note: There are slight modifications Garman-Klass-Yang-Zhang where there is a term using the previous day closing as well. Ideally to take care of stocks that have after hours trading. 


### References
1. Fulvio Corsi, A Simple Approximate Long-Memory Model of Realized Volatility, Journal of Financial Econometrics, Volume 7, Issue 2, Spring 2009, Pages 174–196
2. Duke Lecture notes, https://public.econ.duke.edu/~get/browse/courses/672/Lectures/10_AR-HARmodels.pdf
3. Clements, Adam and Preve, Daniel P. A. and Tee, Clarence, Harvesting the HAR-X Volatility Model
