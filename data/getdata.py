import yfinance as yf

# Download S&P 500 data (ticker: ^GSPC)
sp500 = yf.download('^GSPC', start='2000-01-01', end='2025-01-01')

# Save to CSV if needed
sp500.to_csv('GSPC.csv')

# Print head of DataFrame
print(sp500.head())
