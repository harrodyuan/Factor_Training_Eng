from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=ascending)


def build_equal_weight_long_short(signal: pd.DataFrame, future_returns: pd.DataFrame, top_n: int = 3, bottom_n: int = 3):
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for dt in signal.index:
        row = signal.loc[dt].dropna().sort_values()
        if len(row) < top_n + bottom_n:
            continue
        short_names = row.index[:bottom_n]
        long_names = row.index[-top_n:]
        weights.loc[dt, long_names] = 1.0 / top_n
        weights.loc[dt, short_names] = -1.0 / bottom_n

    aligned_future = future_returns.reindex(weights.index)
    portfolio_ret = (weights * aligned_future).sum(axis=1).dropna()
    weights = weights.loc[portfolio_ret.index]
    turnover = weights.diff().abs().sum(axis=1).div(2).fillna(0)
    return weights, portfolio_ret, turnover


def portfolio_summary(portfolio_ret: pd.Series, turnover: pd.Series | None = None, periods_per_year: int = 12) -> pd.DataFrame:
    ann_ret = periods_per_year * portfolio_ret.mean()
    ann_vol = np.sqrt(periods_per_year) * portfolio_ret.std()
    sharpe_like = ann_ret / ann_vol if ann_vol > 0 else np.nan
    out = {
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_like_ratio': sharpe_like,
        'hit_rate': (portfolio_ret > 0).mean(),
    }
    if turnover is not None:
        out['average_turnover'] = turnover.mean()
    return pd.DataFrame([out])


def fit_factor_model(portfolio_ret: pd.Series, factor_returns: pd.DataFrame):
    aligned = pd.concat([portfolio_ret.rename('portfolio'), factor_returns], axis=1).dropna()
    X = np.column_stack([np.ones(len(aligned)), aligned[factor_returns.columns].values])
    y = aligned['portfolio'].values
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    resid = y - pred
    denom = np.square(y - y.mean()).sum()
    r2 = 1.0 - np.square(resid).sum() / denom if denom > 0 else np.nan
    return {
        'alpha': coef[0],
        'betas': pd.Series(coef[1:], index=factor_returns.columns),
        'r2': r2,
    }


def rolling_factor_regression(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame, window: int = 24):
    idx = asset_returns.index.intersection(factor_returns.index)
    asset_returns = asset_returns.loc[idx].copy()
    factor_returns = factor_returns.loc[idx].copy()

    exposures = {
        'alpha': pd.DataFrame(index=idx, columns=asset_returns.columns, dtype=float)
    }
    for name in factor_returns.columns:
        exposures[name] = pd.DataFrame(index=idx, columns=asset_returns.columns, dtype=float)

    residuals = pd.DataFrame(index=idx, columns=asset_returns.columns, dtype=float)

    factor_matrix = factor_returns.to_numpy(dtype=float)

    for end in range(window, len(idx)):
        X_train = np.column_stack([np.ones(window), factor_matrix[end - window:end]])
        current_x = np.r_[1.0, factor_matrix[end]]

        if np.isnan(X_train).any() or np.isnan(current_x).any():
            continue

        for asset in asset_returns.columns:
            y_train = asset_returns[asset].iloc[end - window:end].to_numpy(dtype=float)
            if np.isnan(y_train).any():
                continue
            coef = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            exposures['alpha'].loc[idx[end], asset] = coef[0]
            for j, name in enumerate(factor_returns.columns, start=1):
                exposures[name].loc[idx[end], asset] = coef[j]
            residuals.loc[idx[end], asset] = asset_returns.loc[idx[end], asset] - current_x @ coef

    return residuals, exposures
