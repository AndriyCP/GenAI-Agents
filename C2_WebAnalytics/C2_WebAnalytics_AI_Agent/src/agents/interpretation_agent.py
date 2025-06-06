import boto3
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterpretationAgent:
    def __init__(self, aws_region='us-east-1'):
        """Initialize the interpretation agent with AWS Bedrock client."""
        try:
            self.bedrock = boto3.client(
                service_name='bedrock-runtime',
                region_name=aws_region
            )
            logger.info("AWS Bedrock client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AWS Bedrock client: {str(e)}")
            raise

    def generate_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """Generate natural language interpretation of analysis results."""
        try:
            # Prepare the prompt for Claude
            prompt = self._prepare_prompt(analysis_results)
            
            # Call Claude through Bedrock
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-v2',
                body=json.dumps({
                    "prompt": prompt,
                    "max_tokens_to_sample": 1000,
                    "temperature": 0.5,
                    "top_p": 0.9,
                })
            )
            
            # Parse and return the response
            response_body = json.loads(response['body'].read())
            interpretation = response_body['completion']
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error generating interpretation: {str(e)}")
            raise

    def _prepare_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """Prepare the prompt for Claude based on analysis results."""
        prompt = """You are an expert data analyst specializing in website analytics. 
        Please provide a clear and concise interpretation of the following analysis results.
        Focus on key insights, trends, and actionable recommendations.
        Stick strictly to the data provided and avoid making assumptions.
        
        Analysis Results:
        """
        
        # Add time series analysis
        if 'time_series_data' in analysis_results:
            prompt += "\nTime Series Analysis:\n"
            ts_data = analysis_results['time_series_data']
            for action in ts_data['action'].unique():
                action_data = ts_data[ts_data['action'] == action]
                prompt += f"\nAction: {action}\n"
                prompt += f"Total hits: {action_data['hitCount'].sum()}\n"
                prompt += f"Average daily hits: {action_data['hitCount'].mean():.2f}\n"
                prompt += f"Max daily hits: {action_data['hitCount'].max()}\n"
                prompt += f"Min daily hits: {action_data['hitCount'].min()}\n"
        
        # Add seasonal decomposition insights
        if 'decomposition' in analysis_results:
            prompt += "\nSeasonal Patterns:\n"
            for action, decomposition in analysis_results['decomposition'].items():
                prompt += f"\nAction: {action}\n"
                # Add seasonal strength analysis
                seasonal_strength = 1 - (decomposition.resid.var() / 
                                      (decomposition.trend + decomposition.seasonal).var())
                prompt += f"Seasonal strength: {seasonal_strength:.2f}\n"
        
        # Add forecast insights
        if 'forecast_results' in analysis_results:
            prompt += "\nForecast Insights:\n"
            forecast_data = analysis_results['forecast_results']
            for action in forecast_data['action'].unique():
                action_forecast = forecast_data[forecast_data['action'] == action]
                prompt += f"\nAction: {action}\n"
                prompt += f"Predicted total hits: {action_forecast['predicted_hitCount'].sum():.2f}\n"
                prompt += f"Average predicted daily hits: {action_forecast['predicted_hitCount'].mean():.2f}\n"
        
        # Add market comparison if available
        if 'market_analysis' in analysis_results:
            prompt += "\nMarket Comparison:\n"
            market_data = analysis_results['market_analysis']
            for action in market_data['action'].unique():
                action_market = market_data[market_data['action'] == action]
                prompt += f"\nAction: {action}\n"
                prompt += f"Market total hits: {action_market['hitCount'].sum()}\n"
                prompt += f"Market average daily hits: {action_market['hitCount'].mean():.2f}\n"
        
        prompt += "\nPlease provide a comprehensive interpretation of these results, focusing on:\n"
        prompt += "1. Key trends and patterns\n"
        prompt += "2. Seasonal variations\n"
        prompt += "3. Forecast implications\n"
        prompt += "4. Market position and competitive insights\n"
        prompt += "5. Actionable recommendations\n"
        
        return prompt 