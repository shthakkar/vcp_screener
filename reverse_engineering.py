import yfinance as yf
import pandas as pd


def get_stocks_with_15_percent_gain(ticker_list):
    results = []

    # Define periods to analyze: last 1 year and the full year 2022
    periods = [('Last 1 Year', '1y'),
               ('Year 2022', '2022-01-01', '2022-12-31')]

    for ticker in ticker_list:
        stock = yf.Ticker(ticker)

        for period_name, *period_range in periods:
            if len(period_range) == 1:
                df = stock.history(period=period_range[0])
            else:
                start_date, end_date = period_range
                df = stock.history(start=start_date, end=end_date)

            df['Change%'] = df['Close'].pct_change(
                periods=10) * 100  # 10 trading days ≈ 2 weeks
            # Filter rows with >15% change
            gain_periods = df[df['Change%'] > 15]

            if not gain_periods.empty:
                for idx in gain_periods.index:
                    # Determine the start and end dates of the 2-week period
                    start_idx = df.index.get_loc(idx) - 10
                    start_period = df.index[start_idx] if start_idx >= 0 else df.index[0]
                    end_period = idx

                    # Calculate the actual percentage change over the identified period
                    start_price = df.loc[start_period, 'Close']
                    end_price = df.loc[end_period, 'Close']
                    actual_change_percent = (
                        (end_price - start_price) / start_price) * 100

                    # Formatting dates for readability
                    start_period_str = start_period.strftime('%d %b %Y')
                    end_period_str = end_period.strftime('%d %b %Y')

                    results.append(
                        f"{ticker} ({period_name}): {start_period_str} to {end_period_str} - {actual_change_percent:.2f}% gain"
                    )

    return results


# Example usage
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]
stocks = get_stocks_with_15_percent_gain(tickers)
print("Stocks with >15% gain in any 2-week period:\n", "\n".join(stocks))
