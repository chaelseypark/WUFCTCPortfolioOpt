import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    csv_files = ["CSVs/DAG_part1.csv", "CSVs/KOP_part1.csv", "CSVs/MON_part1.csv", "CSVs/PED_part1.csv", "CSVs/PUG_part1.csv", "CSVs/TAW_part1.csv", "CSVs/TOW_part1.csv", "CSVs/YON_part1.csv"]
    if len(csv_files) < 8:
        raise ValueError("At least 8 CSV files are required.")

    price_series = []
    for file in csv_files[:8]:
        df = pd.read_csv(file)
        price_series.append(df['C'].rename(file))

    prices = pd.concat(price_series, axis=1).sort_index()

    returns = prices.pct_change().dropna()

    portfolio_returns = []
    for date, row in returns.iterrows():
        w = weights(row)
        port_return = np.dot(w, row.values)
        portfolio_returns.append(port_return)

    portfolio_returns = pd.Series(portfolio_returns, index=returns.index)

    pnl = (1 + portfolio_returns).cumprod()

    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

    print("Final PnL:", pnl.iloc[-1])
    print("Sharpe Ratio:", sharpe_ratio)

    plt.plot(pnl)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title("Portfolio PnL")
    plt.show()


def weights(row):
    csv_files = ["CSVs/DAG_part1.csv", "CSVs/KOP_part1.csv", "CSVs/MON_part1.csv", "CSVs/PED_part1.csv", "CSVs/PUG_part1.csv", "CSVs/TAW_part1.csv", "CSVs/TOW_part1.csv", "CSVs/YON_part1.csv"]
    if len(csv_files) < 8:
        raise ValueError("At least 8 CSV files are required.")

    price_series = []
    for file in csv_files[:8]:
        df = pd.read_csv(file)
        price_series.append(df['C'].rename(file))

    prices = pd.concat(price_series, axis=1).sort_index()

    returns = prices.pct_change().dropna()
    GLOBAL_RETURNS = returns
    ROLLING_WINDOW = 60

    current_date = row.name

    idx = GLOBAL_RETURNS.index.get_loc(current_date)
    
    if idx < ROLLING_WINDOW:
        number_assets = GLOBAL_RETURNS.shape[1]
        return np.ones(number_assets) / number_assets
    
    start_idx = idx - ROLLING_WINDOW
    data_window = GLOBAL_RETURNS.iloc[start_idx:idx, :]
    
    mu = data_window.mean().values
    Sigma = data_window.cov().values

    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.linalg.pinv(Sigma)

    unnorm_w = inv_Sigma @ mu

    sum_unnorm = np.sum(unnorm_w)
    if sum_unnorm <= 0:
        number_assets = GLOBAL_RETURNS.shape[1]
        return np.ones(number_assets) / number_assets

    w = unnorm_w / sum_unnorm

    return w




main()