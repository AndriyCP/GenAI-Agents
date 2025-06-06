import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, data_path):
        """Initialize the DataAnalyzer with the data path."""
        self.data_path = data_path
        self.df = None
        self.domain_data = None
        self.market_data = None
        
    def load_data(self):
        """Load the CSV data."""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            logger.info(f"Data loaded successfully with {len(self.df)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def filter_data(self, domain_name, dma=None, action_types=None, 
                   start_date=None, end_date=None):
        """Filter data based on user inputs."""
        try:
            # Filter by domain
            self.domain_data = self.df[self.df['assigned_domain'] == domain_name].copy()
            
            # Filter by DMA if specified
            if dma:
                self.domain_data = self.domain_data[self.domain_data['dma'] == dma]
            
            # Filter by action types if specified
            if action_types:
                self.domain_data = self.domain_data[self.domain_data['action'].isin(action_types)]
            
            # Filter by date range if specified
            if start_date:
                self.domain_data = self.domain_data[self.domain_data['date'] >= start_date]
            if end_date:
                self.domain_data = self.domain_data[self.domain_data['date'] <= end_date]
            
            # Get market data (all domains) for the same DMAs
            dmas = self.domain_data['dma'].unique()
            self.market_data = self.df[self.df['dma'].isin(dmas)].copy()
            
            logger.info(f"Data filtered successfully for domain: {domain_name}")
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            raise

    def perform_eda(self, frequency='daily'):
        """Perform exploratory data analysis."""
        try:
            # Aggregate data based on frequency
            freq_map = {
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M',
                'quarterly': 'Q',
                'yearly': 'Y'
            }
            
            # Domain analysis
            domain_agg = self.domain_data.groupby([
                pd.Grouper(key='date', freq=freq_map[frequency]),
                'action', 'dma'
            ])['hitCount'].sum().reset_index()
            
            # Market analysis
            market_agg = self.market_data.groupby([
                pd.Grouper(key='date', freq=freq_map[frequency]),
                'action'
            ])['hitCount'].sum().reset_index()
            
            return {
                'domain_analysis': domain_agg,
                'market_analysis': market_agg
            }
        except Exception as e:
            logger.error(f"Error in EDA: {str(e)}")
            raise

    def analyze_timeseries(self, frequency='daily'):
        """Perform time series analysis."""
        try:
            # Prepare data for time series analysis
            ts_data = self.domain_data.groupby([
                pd.Grouper(key='date', freq=frequency),
                'action'
            ])['hitCount'].sum().reset_index()
            
            # Perform seasonal decomposition for each action type
            decomposition_results = {}
            for action in ts_data['action'].unique():
                action_data = ts_data[ts_data['action'] == action].set_index('date')
                decomposition = seasonal_decompose(
                    action_data['hitCount'],
                    period=7 if frequency == 'daily' else 4,
                    extrapolate_trend='freq'
                )
                decomposition_results[action] = decomposition
            
            return {
                'time_series_data': ts_data,
                'decomposition': decomposition_results
            }
        except Exception as e:
            logger.error(f"Error in time series analysis: {str(e)}")
            raise

    def forecast_hitcount(self, forecast_days=30):
        """Generate forecasts using XGBoost."""
        try:
            # Prepare features for forecasting
            df_forecast = self.domain_data.copy()
            df_forecast['day_of_week'] = df_forecast['date'].dt.dayofweek
            df_forecast['month'] = df_forecast['date'].dt.month
            df_forecast['year'] = df_forecast['date'].dt.year
            
            # Encode categorical variables
            le_action = LabelEncoder()
            le_dma = LabelEncoder()
            df_forecast['action_encoded'] = le_action.fit_transform(df_forecast['action'])
            df_forecast['dma_encoded'] = le_dma.fit_transform(df_forecast['dma'])
            
            # Prepare training data
            X = df_forecast[['day_of_week', 'month', 'year', 'action_encoded', 'dma_encoded']]
            y = df_forecast['hitCount']
            
            # Train XGBoost model
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
            model.fit(X, y)
            
            # Generate future dates
            last_date = df_forecast['date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Prepare future features
            future_data = []
            for date in future_dates:
                for action in df_forecast['action'].unique():
                    for dma in df_forecast['dma'].unique():
                        future_data.append({
                            'date': date,
                            'day_of_week': date.dayofweek,
                            'month': date.month,
                            'year': date.year,
                            'action': action,
                            'dma': dma,
                            'action_encoded': le_action.transform([action])[0],
                            'dma_encoded': le_dma.transform([dma])[0]
                        })
            
            future_df = pd.DataFrame(future_data)
            
            # Make predictions
            X_future = future_df[['day_of_week', 'month', 'year', 'action_encoded', 'dma_encoded']]
            future_df['predicted_hitCount'] = model.predict(X_future)
            
            return future_df
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            raise

    def generate_visualizations(self, analysis_results, forecast_results=None):
        """Generate visualizations for the analysis results."""
        try:
            visualizations = {}
            
            # Time series plot
            ts_data = analysis_results['time_series_data']
            fig_ts = px.line(
                ts_data,
                x='date',
                y='hitCount',
                color='action',
                title='Time Series Analysis by Action Type'
            )
            visualizations['time_series'] = fig_ts
            
            # Seasonal decomposition plots
            for action, decomposition in analysis_results['decomposition'].items():
                fig_seasonal = go.Figure()
                fig_seasonal.add_trace(go.Scatter(
                    y=decomposition.trend,
                    name='Trend'
                ))
                fig_seasonal.add_trace(go.Scatter(
                    y=decomposition.seasonal,
                    name='Seasonal'
                ))
                fig_seasonal.add_trace(go.Scatter(
                    y=decomposition.resid,
                    name='Residual'
                ))
                fig_seasonal.update_layout(
                    title=f'Seasonal Decomposition for {action}',
                    xaxis_title='Time',
                    yaxis_title='Value'
                )
                visualizations[f'seasonal_{action}'] = fig_seasonal
            
            # Forecast plot if available
            if forecast_results is not None:
                fig_forecast = px.line(
                    forecast_results,
                    x='date',
                    y='predicted_hitCount',
                    color='action',
                    title='30-Day Forecast by Action Type'
                )
                visualizations['forecast'] = fig_forecast
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise 