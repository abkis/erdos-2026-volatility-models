import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

class fit():
    '''
    Class to fit the HAR-GK model to the provided data and make predictions for the specified date range.   
    '''

    def __init__(self, data, split_date):
        '''
        Initialize the fit class with the provided data and prediction date range.
        data: DataFrame, historical stock data with multi-level columns 
        start_date_predict: str, start date for prediction in 'YYYY-MM-DD' format
        end_date_predict: str, end date for prediction in 'YYYY-MM-DD' format
        '''

        self.data = data[data.index <split_date]   
        self.ticker = self.data.columns.get_level_values(1)[0]
        self.start_date = data.index[0].strftime("%Y-%m-%d")
        self.start_date_predict = split_date
        self.end_date_predict = (data.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def volatility_fit(self):
        '''
        Finds the HAR-GK model for the provided data and returns the fitted model.
        Returns: A fitted LinearRegression model.
        '''

        H = self.data[("High", self.ticker)]
        L = self.data[("Low", self.ticker)]
        C = self.data[("Close", self.ticker)]
        O = self.data[("Open", self.ticker)]

        daily_RV = 0.5*(np.log(H/L))**2 - (2*np.log(2)-1)*(np.log(C/O))**2

        ##This is the HAR dataframe with three variable daily weekly and monthly RV

        self.har_df = pd.DataFrame({
            "rv_d": daily_RV,
            "rv_w": daily_RV.rolling(5).mean(),
            "rv_m": daily_RV.rolling(22).mean(),
            "target": daily_RV.shift(-1)
        }).dropna()

        self.data = self.data.dropna()

        X = self.har_df[["rv_d", "rv_w", "rv_m"]]
        Y = self.har_df["target"]

        model = LinearRegression()
        model.fit(X, Y)
        return model
    

    def volatility_nextday(self):
        '''
        Computes the volatility for the next day based on the fitted HAR-GK model and the last available data point.
        Returns: A float representing the predicted volatility for the next day.
        '''
        
        model = self.volatility_fit()
        X = self.har_df[["rv_d", "rv_w", "rv_m"]]
        day1RV = model.predict(X.iloc[[-1]])[0]
        vol= np.sqrt(day1RV*252)
        return vol

    def trading_days_test(self):
        '''
        Uses the CustomBusinessDay calendar to count the number of trading days between the start and end dates for prediction.
        Returns: An integer representing the number of trading days in the prediction period.
        '''
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

        # Count trading days
        trading_days = pd.date_range(
            start=self.start_date_predict, 
            end=self.end_date_predict, 
            freq=us_bd
        )
        return len(trading_days)


    def test(self):
        '''
        Uses the fitted HAR-GK model to predict volatility for each trading day in the specified prediction date range.
        Returns: A list of predicted volatilities for each trading day in the prediction period.
        '''

        model = self.volatility_fit()

        history = list(self.har_df["rv_d"][-22:])
        future = []
        n=self.trading_days_test()
        for _ in range(n):

            rv_d = history[-1]
            rv_w = np.mean(history[-5:])
            rv_m = np.mean(history[-22:])

            X_n = pd.DataFrame({
                "rv_d": [rv_d],
                "rv_w": [rv_w],
                "rv_m": [rv_m]
            })

            pred_rv = model.predict(X_n)[0]

            future.append(pred_rv)
            history.append(pred_rv)
        
        for i in range(len(future)):
            future[i] = np.sqrt(future[i]*252)

        return future

    def volatility_avg(self):
        '''
        Computes the average volatility over the prediction period.
        Returns: A float representing the average predicted volatility.
        '''

        return np.sqrt(np.sum(self.test())*252/self.trading_days_test())
    