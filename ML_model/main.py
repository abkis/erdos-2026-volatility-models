# test ML model
from model import RF_model

if __name__ == "__main__":
    # see if model works at all
    stock_names = ["AAPL", "AMZN", "GOOG", "NVDA", "META", "TSLA"]
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    window = 21
    target_window = 5

    model = RF_model.RF_Model(stock_names, window, start_date, end_date, features=None, target_window=target_window)

    model.det_features()

    model.fit_RF()

    metrics = model.test_results()

    for k, v in metrics.items():
        print(f"{k} = {v}")
    
    model.plot_results()

    model.scatter()

    model.hist()

    model.rolling_mae_plot()