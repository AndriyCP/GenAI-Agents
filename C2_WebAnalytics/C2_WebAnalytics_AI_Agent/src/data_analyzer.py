import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import json

class DataAnalyzer:
    def __init__(self, df):
        """Initialize the DataAnalyzer with a pandas DataFrame."""
        self.df = df
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def filter_data(self, start_date=None, end_date=None, domain=None, brand=None, dma=None, action=None):
        """Filter the data based on specified criteria."""
        filtered_df = self.df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]
        if domain:
            filtered_df = filtered_df[filtered_df['assigned_domain'] == domain]
        if brand:
            filtered_df = filtered_df[filtered_df['assigned_brand'] == brand]
        if dma:
            filtered_df = filtered_df[filtered_df['dma_name'] == dma]
        if action:
            filtered_df = filtered_df[filtered_df['action'] == action]
            
        return filtered_df
    
    def analyze_time_series(self, df):
        """Perform time series analysis on the filtered data."""
        # Aggregate data by date
        daily_data = df.groupby('date')['hitCount'].sum().reset_index()
        
        # Calculate moving averages
        daily_data['7d_ma'] = daily_data['hitCount'].rolling(window=7).mean()
        daily_data['30d_ma'] = daily_data['hitCount'].rolling(window=30).mean()
        
        # Calculate growth rates
        daily_data['daily_growth'] = daily_data['hitCount'].pct_change()
        daily_data['weekly_growth'] = daily_data['hitCount'].pct_change(periods=7)
        
        return daily_data
    
    def forecast_hit_count(self, df, forecast_days):
        """Generate forecast using XGBoost model."""
        # Prepare features
        daily_data = df.groupby('date')['hitCount'].sum().reset_index()
        
        # Create time-based features
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['year'] = daily_data['date'].dt.year
        daily_data['day_of_year'] = daily_data['date'].dt.dayofyear
        
        # Create lag features
        for lag in [1, 7, 14, 30]:
            daily_data[f'lag_{lag}'] = daily_data['hitCount'].shift(lag)
        
        # Create rolling mean features
        for window in [7, 14, 30]:
            daily_data[f'rolling_mean_{window}'] = daily_data['hitCount'].rolling(window=window).mean()
        
        # Drop rows with NaN values
        daily_data = daily_data.dropna()
        
        # Prepare features and target
        features = ['day_of_week', 'month', 'year', 'day_of_year',
                   'lag_1', 'lag_7', 'lag_14', 'lag_30',
                   'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']
        X = daily_data[features]
        y = daily_data['hitCount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Generate future dates
        last_date = daily_data['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Prepare future features
        future_data = pd.DataFrame({'date': future_dates})
        future_data['day_of_week'] = future_data['date'].dt.dayofweek
        future_data['month'] = future_data['date'].dt.month
        future_data['year'] = future_data['date'].dt.year
        future_data['day_of_year'] = future_data['date'].dt.dayofyear
        
        # Use the last known values for lag features
        for lag in [1, 7, 14, 30]:
            future_data[f'lag_{lag}'] = daily_data['hitCount'].iloc[-lag]
        
        # Calculate rolling means for future dates
        for window in [7, 14, 30]:
            future_data[f'rolling_mean_{window}'] = daily_data['hitCount'].rolling(window=window).mean().iloc[-1]
        
        # Scale future features
        future_features = future_data[features]
        future_features_scaled = scaler.transform(future_features)
        
        # Generate predictions
        predictions = model.predict(future_features_scaled)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': pd.concat([
                pd.Series(daily_data['date'].values),
                pd.Series(future_dates.values)
            ]),
            'hitCount': pd.concat([
                pd.Series(daily_data['hitCount'].values),
                pd.Series(predictions)
            ]),
            'forecast': pd.concat([
                pd.Series([np.nan] * len(daily_data)),
                pd.Series(predictions)
            ])
        })
        
        return forecast_df
    
    def generate_report(self, df):
        """Generate an HTML report with analysis results."""
        # Perform analysis
        time_series = self.analyze_time_series(df)
        forecast = self.forecast_hit_count(df, 14)
        
        # Calculate key metrics
        total_hits = df['hitCount'].sum()
        avg_daily_hits = df.groupby('date')['hitCount'].sum().mean()
        unique_domains = df['assigned_domain'].nunique()
        unique_dmas = df['dma_name'].nunique()
        
        # Create HTML report
        html_report = f"""
        <html>
        <head>
            <title>Website Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Website Analytics Report</h1>
            
            <h2>Key Metrics</h2>
            <div class="metric">Total Hits: {total_hits:,.0f}</div>
            <div class="metric">Average Daily Hits: {avg_daily_hits:,.0f}</div>
            <div class="metric">Unique Domains: {unique_domains}</div>
            <div class="metric">Unique DMAs: {unique_dmas}</div>
            
            <h2>Time Series Analysis</h2>
            <div class="chart">
                {self._generate_time_series_plot(time_series)}
            </div>
            
            <h2>Forecast</h2>
            <div class="chart">
                {self._generate_forecast_plot(forecast)}
            </div>
        </body>
        </html>
        """
        
        return html_report
    
    def _generate_time_series_plot(self, time_series):
        """Generate HTML for time series plot."""
        fig = px.line(time_series, x='date', y=['hitCount', '7d_ma', '30d_ma'],
                     title='Daily Hit Count Trend with Moving Averages')
        return fig.to_html(full_html=False)
    
    def _generate_forecast_plot(self, forecast):
        """Generate HTML for forecast plot."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['hitCount'],
            name='Historical Data',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['forecast'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='Hit Count Forecast',
            xaxis_title='Date',
            yaxis_title='Hit Count',
            showlegend=True
        )
        return fig.to_html(full_html=False) 