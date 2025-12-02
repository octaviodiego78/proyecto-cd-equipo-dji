"""
Data fetching utilities for live gold and S&P 500 prices from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_recent_gold_prices(days=30):
    """
    Fetch recent gold prices from Yahoo Finance
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with columns: Date, Close, High, Low, Open, Volume
    """
    try:
        # Gold Futures ticker
        gold_ticker = yf.Ticker("GC=F")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer for market closures
        
        # Fetch historical data
        gold_data = gold_ticker.history(start=start_date, end=end_date)
        
        if gold_data.empty:
            raise ValueError("No gold price data returned from Yahoo Finance")
        
        # Reset index to get Date as a column
        gold_data = gold_data.reset_index()
        
        # Rename columns to match CSV format
        gold_data = gold_data.rename(columns={
            'Date': 'Date',
            'Close': 'Close',
            'High': 'High',
            'Low': 'Low',
            'Open': 'Open',
            'Volume': 'Volume'
        })
        
        # Select only required columns
        gold_data = gold_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
        
        # Convert Date to datetime without timezone
        gold_data['Date'] = pd.to_datetime(gold_data['Date']).dt.tz_localize(None)
        
        # Sort by date
        gold_data = gold_data.sort_values('Date').reset_index(drop=True)
        
        print(f"Fetched {len(gold_data)} days of gold price data from Yahoo Finance")
        print(f"Date range: {gold_data['Date'].min().date()} to {gold_data['Date'].max().date()}")
        
        return gold_data
        
    except Exception as e:
        raise Exception(f"Error fetching gold prices from Yahoo Finance: {str(e)}")


def fetch_recent_sp500_prices(days=30):
    """
    Fetch recent S&P 500 prices from Yahoo Finance
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with columns: Date, Close, High, Low, Open, Volume
    """
    try:
        # S&P 500 ticker
        sp500_ticker = yf.Ticker("^GSPC")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer for market closures
        
        # Fetch historical data
        sp500_data = sp500_ticker.history(start=start_date, end=end_date)
        
        if sp500_data.empty:
            raise ValueError("No S&P 500 data returned from Yahoo Finance")
        
        # Reset index to get Date as a column
        sp500_data = sp500_data.reset_index()
        
        # Rename columns to match CSV format
        sp500_data = sp500_data.rename(columns={
            'Date': 'Date',
            'Close': 'Close',
            'High': 'High',
            'Low': 'Low',
            'Open': 'Open',
            'Volume': 'Volume'
        })
        
        # Select only required columns
        sp500_data = sp500_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
        
        # Convert Date to datetime without timezone
        sp500_data['Date'] = pd.to_datetime(sp500_data['Date']).dt.tz_localize(None)
        
        # Sort by date
        sp500_data = sp500_data.sort_values('Date').reset_index(drop=True)
        
        print(f"Fetched {len(sp500_data)} days of S&P 500 data from Yahoo Finance")
        print(f"Date range: {sp500_data['Date'].min().date()} to {sp500_data['Date'].max().date()}")
        
        return sp500_data
        
    except Exception as e:
        raise Exception(f"Error fetching S&P 500 prices from Yahoo Finance: {str(e)}")


def fetch_live_data_for_prediction(days=30):
    """
    Fetch both gold and S&P 500 data for making predictions
    
    Args:
        days: Number of days of historical data to fetch
        
    Returns:
        Tuple of (gold_df, sp500_df)
    """
    gold_df = fetch_recent_gold_prices(days=days)
    sp500_df = fetch_recent_sp500_prices(days=days)
    
    return gold_df, sp500_df

