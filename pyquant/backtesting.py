#Conditional Drawdown at Risk (CDaR) measures the average size of a portfolio’s most severe drawdowns, capturing both depth and duration of losses

import os
import warnings
from datetime import datetime
import riskfolio as rp

import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from vectorbt.portfolio.enums import Direction, SizeType
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb

vbt.settings.returns["year_freq"] = "252 days"

warnings.filterwarnings("ignore")

tickers = [
"JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C"
]

#pull daily closing prices for each ticker
data = yf.download(
    tickers, 
    start="2010-01-01", 
    end="2024-06-30", 
    auto_adjust=False
)["Close"].dropna()

print(data)

num_tests = 2000
ann_factor = data.vbt.returns(freq="D").ann_factor

def pre_sim_func_nb(sc, every_nth):
    sc.segment_mask[:, :] = False
    sc.segment_mask[every_nth::every_nth, :] = True
    return ()

def pre_segment_func_nb(
    sc, find_weights_nb, history_len, ann_factor, num_tests, srb_sharpe
):
    if history_len == -1:
        close = sc.close[: sc.i, sc.from_col : sc.to_col]
    else:
        if sc.i - history_len <= 0:
            return (np.full(sc.group_len, np.nan),)
        close = sc.close[sc.i - history_len : sc.i, sc.from_col : sc.to_col]

    best_sharpe_ratio, weights = find_weights_nb(sc, close, num_tests)
    srb_sharpe[sc.i] = best_sharpe_ratio

    size_type = np.full(sc.group_len, SizeType.TargetPercent)
    direction = np.full(sc.group_len, Direction.LongOnly)
    temp_float_arr = np.empty(sc.group_len, dtype=np.float_)
    for k in range(sc.group_len):
        col = sc.from_col + k
        sc.last_val_price[col] = sc.close[sc.i, col]
    sort_call_seq_nb(sc, weights, size_type, direction, temp_float_arr)

    return (weights,)

def order_func_nb(oc, weights):
    col_i = oc.call_seq_now[oc.call_idx]
    return order_nb(
        weights[col_i],
        oc.close[oc.i, oc.col],
        size_type=SizeType.TargetPercent,
    )

def opt_weights(sc, close, num_tests):
    close = pd.DataFrame(close, columns=tickers)
    returns = close.pct_change().dropna()
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")
    w = port.optimization(model="Classic", rm="CDaR", obj="Sharpe", hist=True)
    weights = np.ravel(w.to_numpy())
    shp = rp.Sharpe(
        weights=w,
        returns=returns,
        mu=port.mu,
        cov=port.cov,
        rm="CDaR",
        alpha=0.05
    )
    return shp, weights

sharpe = np.full(data.shape, np.nan)
pf = vbt.Portfolio.from_order_func(
    data,
    order_func_nb,
    pre_sim_func_nb=pre_sim_func_nb,
    pre_sim_args=(30,),
    pre_segment_func_nb=pre_segment_func_nb,
    pre_segment_args=(opt_weights, 252 * 4, ann_factor, num_tests, sharpe),
    cash_sharing=True,
    group_by=True,
    use_numba=False,
    freq="D"
)

pf.plot_cum_returns()

pf.stats()