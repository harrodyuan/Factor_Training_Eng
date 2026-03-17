from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yfinance as yf


def download_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    px = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px.columns = [str(c).lower().replace(' ', '_') for c in px.columns]
    if 'adj_close' not in px.columns and 'adj close' in px.columns:
        px['adj_close'] = px['adj close']
    if 'adj_close' not in px.columns:
        px['adj_close'] = px['close']
    return px


def download_adjusted_close_panel(tickers, start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        level0 = {str(v).lower().replace(' ', '_'): v for v in data.columns.get_level_values(0).unique()}
        if 'adj_close' in level0:
            panel = data[level0['adj_close']]
        elif 'adj close' in level0:
            panel = data[level0['adj close']]
        elif 'close' in level0:
            panel = data[level0['close']]
        else:
            raise KeyError('Could not locate an adjusted close or close panel in the yfinance output.')
        if isinstance(panel, pd.Series):
            panel = panel.to_frame(name=tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers))
        panel.columns = [str(c) for c in panel.columns]
        return panel.sort_index()

    if isinstance(data, pd.Series):
        name = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
        return data.to_frame(name=name)

    if 'Adj Close' in data.columns:
        out = data[['Adj Close']].copy()
    else:
        out = data[['Close']].copy()
    out.columns = [tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)]
    return out.sort_index()


def build_basic_daily_features(px: pd.DataFrame, windows=(5, 21, 63, 126)) -> pd.DataFrame:
    feat = pd.DataFrame(index=px.index)
    feat['ret_1d'] = px['adj_close'].pct_change()
    feat['log_ret_1d'] = np.log(px['adj_close'] / px['adj_close'].shift(1))
    for window in windows:
        feat[f'mom_{window}d'] = px['adj_close'].pct_change(window)
        feat[f'vol_{window}d'] = feat['ret_1d'].rolling(window).std()
    feat['drawdown'] = px['adj_close'] / px['adj_close'].cummax() - 1
    return feat


def get_financial_table(ticker, attr: str) -> pd.DataFrame:
    obj = getattr(ticker, attr, None)
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()


def get_shares_series(ticker):
    shares = None
    for getter in ('get_shares_full', 'get_shares'):
        try:
            shares = getattr(ticker, getter)()
            if shares is not None:
                break
        except Exception:
            continue
    if isinstance(shares, pd.DataFrame) and shares.shape[1] > 0:
        shares = shares.iloc[:, 0]
    if shares is None or not hasattr(shares, 'index'):
        return None
    shares = shares[~shares.index.duplicated(keep='last')]
    shares.index = pd.to_datetime(shares.index, errors='coerce').tz_localize(None)
    return shares.dropna()


def build_quarterly_per_share(statement_df: pd.DataFrame, shares, name_options):
    if statement_df.empty or shares is None or len(shares) == 0:
        return None
    statement_df = statement_df.T.copy()
    statement_df.index = pd.to_datetime(statement_df.index, errors='coerce').tz_localize(None)
    statement_df = statement_df.dropna(how='all')
    selected_col = None
    for col in statement_df.columns:
        if any(name in str(col) for name in name_options):
            selected_col = col
            break
    if selected_col is None:
        return None
    series = statement_df[selected_col]
    shares_aligned = shares.reindex(statement_df.index, method='nearest')
    out = (series / shares_aligned).replace([np.inf, -np.inf], np.nan)
    if isinstance(out, pd.DataFrame) and out.shape[1] > 0:
        out = out.iloc[:, 0]
    return out


def build_fundamental_features(ticker_symbol: str, price_index, price_series) -> pd.DataFrame:
    tk = yf.Ticker(ticker_symbol)
    income_q = get_financial_table(tk, 'quarterly_income_stmt')
    if income_q.empty:
        income_q = get_financial_table(tk, 'income_stmt')
    balance_q = get_financial_table(tk, 'quarterly_balance_sheet')
    if balance_q.empty:
        balance_q = get_financial_table(tk, 'balance_sheet')

    shares = get_shares_series(tk)
    fund = pd.DataFrame(index=price_index)

    eps_q = build_quarterly_per_share(income_q, shares, ['Net Income'])
    if eps_q is not None:
        fund['eps_q'] = eps_q.reindex(fund.index, method='ffill')

    bvps_q = build_quarterly_per_share(balance_q, shares, ['Total Stockholder Equity', 'Total Equity'])
    if bvps_q is not None:
        fund['bvps_q'] = bvps_q.reindex(fund.index, method='ffill')

    price_series = pd.Series(price_series, index=price_index).astype(float)
    if 'eps_q' in fund.columns:
        fund['pe'] = price_series / pd.Series(fund['eps_q'], index=fund.index).astype(float)
    if 'bvps_q' in fund.columns:
        fund['pb'] = price_series / pd.Series(fund['bvps_q'], index=fund.index).astype(float)

    return fund.replace([np.inf, -np.inf], np.nan)


def locate_lobster_sample(candidates=None, extract_root=None) -> Path:
    if candidates is None:
        candidates = [
            Path('/Users/harold/4. RA work/FACTOR_TRAINING_CHI/lobster_samples/AAPL_2012-06-21_5'),
            Path('/Users/harold/Documents/LOBSTER_SampleFile_AAPL_2012-06-21_5.zip'),
        ]
    if extract_root is None:
        extract_root = Path('/Users/harold/4. RA work/Factor_Training_Eng/.cache')

    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.is_dir():
            return candidate
        if candidate.is_file() and candidate.suffix == '.zip':
            extract_dir = extract_root / candidate.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(extract_dir)
            return extract_dir

    raise FileNotFoundError('Could not locate the AAPL LOBSTER sample directory or zip file.')


def load_lobster_sample(sample_dir: Path, levels: int = 5) -> pd.DataFrame:
    msg_path = next(sample_dir.glob(f'*message_{levels}.csv'))
    book_path = next(sample_dir.glob(f'*orderbook_{levels}.csv'))

    msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
    book_cols = []
    for level in range(1, levels + 1):
        book_cols += [f'ask_{level}', f'ask_size_{level}', f'bid_{level}', f'bid_size_{level}']

    msg = pd.read_csv(msg_path, header=None, names=msg_cols)
    book = pd.read_csv(book_path, header=None, names=book_cols)
    return pd.concat([msg, book], axis=1)


def compute_ofi_one_level(bid_price, bid_size, ask_price, ask_size) -> pd.Series:
    bid_prev = bid_price.shift(1)
    ask_prev = ask_price.shift(1)
    bid_size_prev = bid_size.shift(1)
    ask_size_prev = ask_size.shift(1)

    bid_contrib = np.where(
        bid_price > bid_prev,
        bid_size,
        np.where(bid_price == bid_prev, bid_size - bid_size_prev, -bid_size_prev),
    )
    ask_contrib = np.where(
        ask_price < ask_prev,
        ask_size,
        np.where(ask_price == ask_prev, ask_size - ask_size_prev, -ask_size_prev),
    )
    return pd.Series(bid_contrib - ask_contrib, index=bid_price.index)


def build_lob_event_features(lob: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    feat_lob = lob.copy()
    feat_lob['mid'] = (feat_lob['bid_1'] + feat_lob['ask_1']) / 2
    feat_lob['spread'] = feat_lob['ask_1'] - feat_lob['bid_1']
    feat_lob['spread_bps'] = 10000 * feat_lob['spread'] / feat_lob['mid']
    feat_lob['imbalance_l1'] = (
        feat_lob['bid_size_1'] - feat_lob['ask_size_1']
    ) / (
        feat_lob['bid_size_1'] + feat_lob['ask_size_1']
    )
    bid_depth = feat_lob[[f'bid_size_{i}' for i in range(1, levels + 1)]].sum(axis=1)
    ask_depth = feat_lob[[f'ask_size_{i}' for i in range(1, levels + 1)]].sum(axis=1)
    feat_lob['imbalance_l5'] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    feat_lob['microprice'] = (
        feat_lob['ask_1'] * feat_lob['bid_size_1'] + feat_lob['bid_1'] * feat_lob['ask_size_1']
    ) / (feat_lob['bid_size_1'] + feat_lob['ask_size_1'])
    feat_lob['microprice_dev_bp'] = 10000 * (feat_lob['microprice'] - feat_lob['mid']) / feat_lob['mid']
    feat_lob['ofi_l1'] = compute_ofi_one_level(feat_lob['bid_1'], feat_lob['bid_size_1'], feat_lob['ask_1'], feat_lob['ask_size_1'])
    feat_lob['ofi_l5'] = sum(
        compute_ofi_one_level(
            feat_lob[f'bid_{level}'],
            feat_lob[f'bid_size_{level}'],
            feat_lob[f'ask_{level}'],
            feat_lob[f'ask_size_{level}'],
        )
        for level in range(1, levels + 1)
    )
    feat_lob['signed_size'] = feat_lob['direction'] * feat_lob['size']
    feat_lob['time_str'] = pd.to_timedelta(feat_lob['time'], unit='s')
    return feat_lob


def build_lob_bars(feat_lob: pd.DataFrame) -> pd.DataFrame:
    feat_lob = feat_lob.copy()
    feat_lob['time_floor'] = np.floor(feat_lob['time']).astype(int)
    bars = (
        feat_lob.groupby('time_floor', sort=True)
        .agg(
            time=('time', 'last'),
            bid_1=('bid_1', 'last'),
            bid_size_1=('bid_size_1', 'last'),
            ask_1=('ask_1', 'last'),
            ask_size_1=('ask_size_1', 'last'),
            mid=('mid', 'last'),
            spread=('spread', 'last'),
            spread_bps=('spread_bps', 'last'),
            imbalance_l1=('imbalance_l1', 'last'),
            imbalance_l5=('imbalance_l5', 'last'),
            microprice_dev_bp=('microprice_dev_bp', 'last'),
            ofi_l1=('ofi_l1', 'sum'),
            ofi_l5=('ofi_l5', 'sum'),
            signed_size=('signed_size', 'sum'),
            msg_count=('type', 'size'),
        )
        .reset_index()
    )
    bars['time_sec'] = bars['time_floor']
    bars['time_str'] = pd.to_timedelta(bars['time_sec'], unit='s')
    return bars
