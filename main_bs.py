
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BS_model    import BS_Model
from ML_BS_model import ML_BS_Model


# ── configuration — identical to main.py ──────────────────────────────────────

stable_start   = "2015-01-01"
stable_end     = "2018-12-31"
unstable_start = "2019-01-01"
unstable_end   = "2022-12-31"

time_dict = {
    "stable"  : [stable_start,   stable_end],
    "unstable": [unstable_start, unstable_end],
    "full"    : [stable_start,   unstable_end],
}

tickers = ["AAPL", "AMZN", "GOOG", "NVDA", "META", "TSLA"]

rf_params = {
    "max_depth":        12,
    "max_features":     0.5,
    "min_samples_leaf":  5,
    "n_estimators":    500,
    "random_state":     42,
    "n_jobs":          -1,
}
window        = 21    # trailing vol window (trading days)
target_window = 0     # convention carried from main.py
target_name   = "GK_vol"

METRICS = ["mse", "rmse", "mae", "mape", "r2", "rel_mse"]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)


# ── data download ─────────────────────────────────────────────────────────────

def get_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download and flatten yfinance OHLCV data — mirrors main.py."""
    stock_data = yf.download(
        ticker, start=start_date, end=end_date,
        auto_adjust=True, progress=False,
    )
    if stock_data.empty:
        raise ValueError(f"No data returned for {ticker} ({start_date} – {end_date})")

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = (
            stock_data
            .stack(level=1, future_stack=True)
            .reset_index()
            .rename(columns={"level_0": "Date", "Ticker": "Symbol"})
        )
    else:
        stock_data = stock_data.reset_index()
        stock_data["Symbol"] = ticker

    return stock_data.sort_values("Date").reset_index(drop=True)


# ── train / test split (same as main.py) ─────────────────────────────────────

def _split(data: pd.DataFrame):
    """80/20 chronological split; returns (train_df, test_df)."""
    split_date = data["Date"].quantile(0.8)
    return (
        data[data["Date"] <  split_date].copy(),
        data[data["Date"] >= split_date].copy(),
    )


# ── per-run evaluation ────────────────────────────────────────────────────────

def evaluate_bs(data: pd.DataFrame) -> dict:
    """
    Pure BS backtest.

    The model is initialised with the FULL data so rolling vol is correct
    at every row; only the test-date rows are scored.
    """
    train, test = _split(data)
    model = BS_Model(data, vol_window=window, horizon=window, r=0.04)
    return model.test_results(test)


def evaluate_ml_bs(data: pd.DataFrame) -> dict:
    """
    ML-corrected BS backtest.

    Same full-series initialisation pattern; RF is trained on train dates
    only, then evaluated on test dates.
    """
    train, test = _split(data)
    model = ML_BS_Model(data, vol_window=window, horizon=window,
                        r=0.04, rf_params=rf_params)
    model.fit(train)
    return model.test_results(test)


# ── print helpers — match output.txt format exactly ───────────────────────────

def _print_metrics(metrics: dict):
    for k in METRICS:
        print(f"{k} = {metrics.get(k, float('nan'))}")


def _print_section(label: str):
    print(f"\n {label}\n")


# ── summary builders — produce ticker/time CSV matching the RF output ─────────

def _build_summaries(results: pd.DataFrame, tag: str):
    """Save CSVs and print rankings to stdout."""

    ticker_summary = (
        results.groupby("Symbol")[METRICS]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    time_summary = (
        results.groupby("time")[METRICS]
        .agg(["mean", "std", "median"])
        .round(4)
    )
    overall = results[METRICS].agg(["mean", "std", "median", "min", "max"])

    ticker_summary.to_csv(os.path.join(OUT_DIR, f"ticker_summary_{tag}.csv"))
    time_summary.to_csv(  os.path.join(OUT_DIR, f"time_summary_{tag}.csv"))
    overall.to_csv(       os.path.join(OUT_DIR, f"overall_results_{tag}.csv"))

    # ── rankings (same blocks as main.py output) ──────────────────────────────

    ticker_rank = (
        results.groupby("Symbol")["r2"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\n Ticker Rank\n")
    print(ticker_rank.to_string())

    ticker_rmse = (
        results.groupby("Symbol")["rmse"]
        .mean()
        .sort_values()
    )
    print("\n Ticker RMSE\n")
    print(ticker_rmse.to_string())

    stability = results.groupby("Symbol")["r2"].agg(["mean", "std"])
    print("\n Stability\n")
    print(stability.to_string())

    time_rank = results.groupby("time")["r2"].mean().sort_values()
    print("\n Time Rank\n")
    print(time_rank.to_string())

    corr = results[METRICS].corr()
    print("\n Correlation bw metrics:\n", corr.to_string())

    best_each_time = results.loc[results.groupby("time")["r2"].idxmax()]
    wins = best_each_time["Symbol"].value_counts()
    print("\nBest Tickers\n", wins.to_string())

    return ticker_summary, time_summary


# ── visualisations ────────────────────────────────────────────────────────────

def _heatmap(bs_res: pd.DataFrame, ml_res: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, title in zip(
        axes,
        [bs_res, ml_res],
        ["Pure Black-Scholes — R²", "ML-Corrected BS — R²"],
    ):
        heat = res.pivot_table(index="Symbol", columns="time",
                               values="r2", aggfunc="mean")
        sns.heatmap(heat, cmap="RdYlGn", center=0, annot=True,
                    fmt=".3f", ax=ax, vmin=-1, vmax=1)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "r2_heatmap_comparison.png"), dpi=150)
    plt.close()


def _rmse_bar(bs_res: pd.DataFrame, ml_res: pd.DataFrame):
    bs_r  = bs_res.groupby("Symbol")["rmse"].mean().rename("Pure BS")
    ml_r  = ml_res.groupby("Symbol")["rmse"].mean().rename("ML-BS")
    melt  = pd.concat([bs_r, ml_r], axis=1).reset_index().melt(
                id_vars="Symbol", var_name="Model", value_name="RMSE")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melt, x="Symbol", y="RMSE", hue="Model")
    plt.title("Mean RMSE by Ticker: Pure BS vs ML-Corrected BS")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rmse_comparison.png"), dpi=150)
    plt.close()


def _stability_scatter(bs_res: pd.DataFrame, ml_res: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, title in zip(
        axes,
        [bs_res, ml_res],
        ["Pure BS — R² stability", "ML-Corrected BS — R² stability"],
    ):
        stab = res.groupby("Symbol")["r2"].agg(["mean", "std"]).reset_index()
        sns.scatterplot(data=stab, x="mean", y="std", s=80, ax=ax)
        for _, row in stab.iterrows():
            ax.text(row["mean"], row["std"], row["Symbol"], fontsize=9)
        ax.set_xlabel("Mean R²")
        ax.set_ylabel("Std R²")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stability_comparison.png"), dpi=150)
    plt.close()


def _boxplots(bs_res: pd.DataFrame, ml_res: pd.DataFrame):
    for metric in ["r2", "rmse"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, res, title in zip(
            axes, [bs_res, ml_res],
            [f"Pure BS — {metric.upper()}", f"ML-BS — {metric.upper()}"],
        ):
            sns.boxplot(data=res, x="Symbol", y=metric, hue="time", ax=ax)
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"boxplot_{metric}.png"), dpi=150)
        plt.close()


def _metric_corr(results: pd.DataFrame, tag: str):
    corr = results[METRICS].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Metric correlations — {tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"metric_corr_{tag}.png"), dpi=150)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sns.set_style("darkgrid")

    bs_results = pd.DataFrame(columns=["Symbol", "time"] + METRICS)
    ml_results = pd.DataFrame(columns=["Symbol", "time"] + METRICS)

    # ── redirect stdout to both console and file ───────────────────────────────
    import io, contextlib

    log_lines = []

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    log_buf = io.StringIO()
    tee = _Tee(sys.__stdout__, log_buf)
    sys.stdout = tee

    # ── main evaluation loop ───────────────────────────────────────────────────

    for ticker in tickers:
        print(f"\n\n ----- Ticker: {ticker} ---- \n")

        for descr, (start, end) in time_dict.items():
            print(f"\n {descr} Time Period {start} to {end}\n")

            try:
                data = get_data(ticker, start, end)
            except Exception as exc:
                print(f"[ERROR] data download failed: {exc}")
                continue

            # Pure BS ──────────────────────────────────────────────────────────
            try:
                m_bs = evaluate_bs(data)
                _print_metrics(m_bs)
                bs_results.loc[len(bs_results)] = {"Symbol": ticker, "time": descr, **m_bs}
            except Exception as exc:
                print(f"[ERROR BS] {exc}")

            # ML-corrected BS ──────────────────────────────────────────────────
            try:
                m_ml = evaluate_ml_bs(data)
                ml_results.loc[len(ml_results)] = {"Symbol": ticker, "time": descr, **m_ml}
            except Exception as exc:
                print(f"[ERROR ML-BS] {exc}")

    # ── summaries ──────────────────────────────────────────────────────────────

    print("\n\n" + "=" * 60)
    print("  SUMMARY: PURE BLACK-SCHOLES")
    print("=" * 60)
    _build_summaries(bs_results, tag="bs")

    print("\n\n" + "=" * 60)
    print("  SUMMARY: ML-CORRECTED BLACK-SCHOLES")
    print("=" * 60)
    _build_summaries(ml_results, tag="ml")

    # ── save console log ───────────────────────────────────────────────────────
    sys.stdout = sys.__stdout__
    log_path = os.path.join(OUT_DIR, "output_bs.txt")
    with open(log_path, "w") as f:
        f.write(log_buf.getvalue())
    print(f"\nConsole log saved to: {log_path}")

    # ── plots ──────────────────────────────────────────────────────────────────
    _heatmap(bs_results, ml_results)
    _rmse_bar(bs_results, ml_results)
    _stability_scatter(bs_results, ml_results)
    _boxplots(bs_results, ml_results)
    _metric_corr(bs_results, "bs")
    _metric_corr(ml_results, "ml")

    print(f"\nAll outputs saved to: {OUT_DIR}")
