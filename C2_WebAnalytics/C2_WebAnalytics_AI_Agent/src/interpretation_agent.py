import json
import pandas as pd
from datetime import datetime

class InterpretationAgent:
    def __init__(self, bedrock_client):
        """Initialize the interpretation agent with AWS Bedrock client."""
        self.bedrock = bedrock_client
        
    def generate_interpretation(self, df, time_series, forecast=None, news_context=None, top_dmas=None):
        """Generate natural language interpretation of the analysis results."""
        # Prepare data summary
        data_summary = self._prepare_data_summary(df, time_series, forecast, top_dmas)
        
        # Create prompt for Claude
        prompt = self._create_prompt(data_summary, news_context)
        
        # Call AWS Bedrock
        response = self._call_bedrock(prompt)
        
        return response
    
    def _prepare_data_summary(self, df, time_series, forecast, top_dmas=None):
        """Prepare a summary of the data for interpretation."""
        summary = {
            'overview': {
                'total_hits': int(df['hitCount'].sum()),
                'avg_daily_hits': float(df.groupby('date')['hitCount'].sum().mean()),
                'unique_domains': int(df['assigned_domain'].nunique()),
                'unique_dmas': int(df['dma_name'].nunique()),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                }
            },
            'time_series': {
                'trend': self._analyze_trend(time_series),
                'seasonality': self._analyze_seasonality(time_series),
                'volatility': self._analyze_volatility(time_series)
            }
        }
        
        if forecast is not None:
            summary['forecast'] = {
                'forecast_period': {
                    'start': forecast['date'].iloc[-len(forecast[forecast['forecast'].notna()]):].min().strftime('%Y-%m-%d'),
                    'end': forecast['date'].max().strftime('%Y-%m-%d')
                },
                'forecast_values': forecast[forecast['forecast'].notna()]['forecast'].tolist()
            }

        if top_dmas is not None and not top_dmas.empty:
            summary['top_dmas'] = top_dmas.to_dict(orient='records')
        
        return summary
    
    def _analyze_trend(self, time_series):
        """Analyze the trend in the time series data."""
        # Calculate overall trend
        first_value = time_series['hitCount'].iloc[0]
        last_value = time_series['hitCount'].iloc[-1]
        total_days = (time_series['date'].iloc[-1] - time_series['date'].iloc[0]).days
        
        trend = {
            'direction': 'increasing' if last_value > first_value else 'decreasing',
            'total_change': float(last_value - first_value),
            'daily_change': float((last_value - first_value) / total_days) if total_days > 0 else 0
        }
        
        return trend
    
    def _analyze_seasonality(self, time_series):
        """Analyze seasonality in the time series data."""
        # Calculate average hits by day of week
        time_series['day_of_week'] = time_series['date'].dt.dayofweek
        daily_avg = time_series.groupby('day_of_week')['hitCount'].mean()
        
        seasonality = {
            'weekly_pattern': daily_avg.to_dict(),
            'peak_day': int(daily_avg.idxmax()),
            'trough_day': int(daily_avg.idxmin())
        }
        
        return seasonality
    
    def _analyze_volatility(self, time_series):
        """Analyze volatility in the time series data."""
        volatility = {
            'std_dev': float(time_series['hitCount'].std()),
            'cv': float(time_series['hitCount'].std() / time_series['hitCount'].mean()),
            'range': {
                'min': float(time_series['hitCount'].min()),
                'max': float(time_series['hitCount'].max())
            }
        }
        
        return volatility
    
    def _create_prompt(self, data_summary, news_context=None):
        """Create a prompt for the Claude model."""
        prompt = f"""You are an expert data analyst. Please analyze the following website analytics data and provide insights in natural language. Focus on key trends, patterns, and actionable insights.

Data Summary:
{json.dumps(data_summary, indent=2)}"""

        if news_context:
            prompt += f"""

Industry News Context:
{news_context}

Please incorporate relevant industry news and market trends into your analysis."""

        prompt += """

Please provide a comprehensive analysis that includes:
1. Overview of the data and key metrics
2. Analysis of trends and patterns
3. Seasonal patterns and their implications
4. Volatility and stability assessment
5. Forecast interpretation (if available)
6. Industry context and market alignment (if news provided)
7. Key insights from the top DMAs.
8. Actionable recommendations

Format your response in clear, concise paragraphs with bullet points for key findings."""
        
        return prompt
    
    def _call_bedrock(self, prompt):
        """Call AWS Bedrock to generate interpretation."""
        try:
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-v2',
                body=json.dumps({
                    'prompt': f"\n\nHuman: {prompt}\n\nAssistant:",
                    'max_tokens_to_sample': 1000,
                    'temperature': 0.7,
                    'top_p': 1,
                    'stop_sequences': ["\n\nHuman:"]
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['completion']
            
        except Exception as e:
            return f"Error generating interpretation: {str(e)}" 