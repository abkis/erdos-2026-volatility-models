# test ML model
from model import RF_model
import joblib
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
tickers_path = BASE_DIR / "data" / "tickers.json"
model_path = BASE_DIR / "data" / "models2"

with open(tickers_path, "r") as file:
    ticker_dict = json.load(file)

stable_start = "2015-01-01" 
stable_end = "2018-12-31" 
unstable_start = "2019-01-01" 
unstable_end = "2022-12-31" 

time_dict = {"stable" : [stable_start, stable_end], "unstable" : [unstable_start, unstable_end], "full" : [stable_start, unstable_end]}

all_tickers = []
rf_params = {'max_depth': 12, 'max_features': 0.5, 'min_samples_leaf': 5, 'n_estimators': 500} # don't do grid search bc takes too long
window = 21
target_window = 5

def test_and_store_model(start_date, end_date, file_path):
    try:
        model = RF_model.RF_Model(tickers, window, start_date, end_date, features=None, target_window=target_window, grid_search=False, rf_params=rf_params)
    
        model.det_features()

        model.fit_RF()
    except Exception as e:
        print("err: ", e)
        raise e

    joblib.dump(model, file_path + ".pkl")

    metrics = model.test_results()

    for k, v in metrics.items():
        if k == "bias":
            continue
        print(f"{k} = {v}")
    
    model.plot_results()

    model.scatter()

    model.hist()

    model.rolling_mae_plot()
    

if __name__ == "__main__":

    for sector, tickers in ticker_dict.items():
        # train model per sector 
        # compare with overall model for all sectors
        # compare with different data lengths
        print(f"\n\n----- Sector: {sector} -----\n")
        for descr, time in time_dict.items():
            [start, end] = time
            print(f"\n {descr} Time Period {start} to {end}\n")
            try:
                test_and_store_model(start, end, f"{model_path}/{sector}_{descr}_model")
                all_tickers += tickers
            except:
                print("skipping...")

    print(f"\n\n----- All Tickers -----\n")
    for descr, time in time_dict.items():
        [start, end] = time
        print(f"\n {descr} Time Period {start} to {end}\n")
        test_and_store_model(start, end, f"all_tickers_{descr}_model")