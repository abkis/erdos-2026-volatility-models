# test ML model
from model import RF_model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf

stable_start = "2015-01-01" 
stable_end = "2018-12-31" 
unstable_start = "2019-01-01" 
unstable_end = "2022-12-31" 

time_dict = {"stable" : [stable_start, stable_end], "unstable" : [unstable_start, unstable_end], "full" : [stable_start, unstable_end]}

tickers = ["AAPL", "AMZN", "GOOG", "NVDA", "META", "TSLA"]
rf_params = {'max_depth': 12, 'max_features': 0.5, 'min_samples_leaf': 5, 'n_estimators': 500} # don't do grid search bc takes too long
window = 21
target_window = 0
target_name = "GK_vol"

def test_model(data : pd.DataFrame):
    # split into train/test
    split_date = data.Date.quantile(0.8)
        
    train_mask = data['Date'] < split_date
        
    X_train = data[train_mask]
    X_test  = data[~train_mask]

    try:
        model = RF_model.RF_Model(X_train, window=window, target_window=0, target_name=target_name, features=None, grid_search=False, rf_params=rf_params)
    
        model.det_features()

        model.fit_RF()
    except Exception as e:
        raise e

    metrics = model.test_results(X_test)

    for k, v in metrics.items():
        print(f"{k} = {v}")
    
    return metrics

def get_data(ticker : str, start_date, end_date):
    # get data from yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # collapse df
    stock_data = (
            stock_data.stack(level=1, future_stack=True)              
              .reset_index()
              .rename(columns={
                  'level_0': 'Date',
                  'Ticker': 'Symbol'
              })
            )
    return stock_data.sort_values(by=["Date"])

if __name__ == "__main__":
    results = pd.DataFrame(columns=["Symbol", "time", "mse", "rmse", "mape", "r2", "mae", "rel_mse"])

    for ticker in tickers:
        print(f"\n\n ----- Ticker: {ticker} ---- \n")
        for descr, time in time_dict.items():
            [start, end] = time
            print(f"\n {descr} Time Period {start} to {end}\n")
            data = get_data(ticker, start, end)
            res = test_model(data)
            res["Symbol"] = ticker
            res["time"] = descr
            results.loc[len(results)] = res

    # understand results
    metrics = ["mse", "rmse", "mae", "mape", "r2", "rel_mse"]

    overall = results[metrics].agg(["mean", "std", "median", "min", "max"])
    overall.to_csv("ML_model/data/overall_results.csv")
    
    ticker_summary = (
        results
        .groupby("Symbol")[metrics]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    ticker_summary.to_csv("ML_model/data/ticker_summary.csv")

    time_summary = (
        results
        .groupby("time")[metrics]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    time_summary.to_csv("ML_model/data/time_summary.csv")

    # ticker rankings
    ticker_rank = (
        results
        .groupby("Symbol")["r2"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n Ticker Rank\n")
    print(ticker_rank)
    
    ticker_rmse = (
        results
        .groupby("Symbol")["rmse"]
        .mean()
        .sort_values()
    )
    print("\n Ticker RMSE\n")
    print(ticker_rmse)

    stability = (
        results
        .groupby("Symbol")["r2"]
        .agg(["mean","std"])
    )
    print("\n Stability\n")
    print(stability)

    time_rank = (
        results
        .groupby("time")["r2"]
        .mean()
        .sort_values()
    )
    print("\n Time Rank\n")
    print(time_rank)

    # heatmap data
    heat_r2 = results.pivot_table(
        index="Symbol",
        columns="time",
        values="r2",
        aggfunc="mean"
    )
    
    heat_rmse = results.pivot_table(
        index="Symbol",
        columns="time",
        values="rmse",
        aggfunc="mean"
    )

    plt.figure(figsize=(12,8))
    sns.heatmap(heat_r2, cmap="RdYlGn", center=0)
    plt.title("Average R²")
    plt.show()

    # boxplots
    plt.figure(figsize=(12,5))
    sns.boxplot(data=results, x="Symbol", y="r2")
    plt.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(12,5))
    sns.boxplot(data=results, x="Symbol", y="rmse")
    plt.xticks(rotation=90)
    plt.show()

    # correlation info bw metrics
    corr = results[metrics].corr()
    
    print("\n Correlation bw metrics:\n", corr)
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()

    # scatterplot
    stability = (
        results
        .groupby("Symbol")["r2"]
        .agg(["mean","std"])
        .reset_index()
    )
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=stability,
        x="mean",
        y="std",
        s=80
    )
    
    for _, row in stability.iterrows():
        plt.text(row["mean"], row["std"], row["Symbol"])
    
    plt.xlabel("Mean R²")
    plt.ylabel("Std R²")
    plt.show()

    # which ticker is best most of the time?
    best_each_time = (
        results.loc[
            results.groupby("time")["r2"].idxmax()
        ]
    )
    
    wins = best_each_time["Symbol"].value_counts()
    
    print("\nBest Tickers\n", wins)