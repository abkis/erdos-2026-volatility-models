"""Minimal example for calling the GARCH model from main.py."""

from garch_model import GARCHModel


def run_garch(train_df, test_df, ticker="^GSPC"):
    model = GARCHModel(
        train_df=train_df,
        ticker=ticker,
        price_col="Close",
        realized_vol_method="abs_return",  # use "parkinson" if High/Low are preferred
        annualize_output=False,             # set True if the comparison uses annualized vol
    )

    predicted_volatility = model.predict(test_df)
    metrics = model.test(test_df)
    params = model.get_fitted_params()

    return predicted_volatility, metrics, params
