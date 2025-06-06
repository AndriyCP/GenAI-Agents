import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analysis.data_analyzer import DataAnalyzer
import os
import tempfile

class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'pixid': ['test1', 'test2'] * 5,
            'unique_pixel': [True] * 10,
            'assigned_domain': ['test.com'] * 10,
            'assigned_brand': ['test_brand'] * 10,
            'dma': [501, 502] * 5,
            'dma_name': ['TEST1', 'TEST2'] * 5,
            'date': pd.date_range(start='2024-01-01', periods=10),
            'action': ['lead', 'view'] * 5,
            'hitCount': np.random.randint(1, 100, 10)
        })
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.sample_data.to_csv(self.temp_file.name, index=False)
        
        # Initialize analyzer
        self.analyzer = DataAnalyzer(self.temp_file.name)
        self.analyzer.load_data()

    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_file.name)

    def test_load_data(self):
        """Test data loading."""
        self.assertIsNotNone(self.analyzer.df)
        self.assertEqual(len(self.analyzer.df), 10)
        self.assertTrue('date' in self.analyzer.df.columns)
        self.assertTrue(self.analyzer.df['date'].dtype == 'datetime64[ns]')

    def test_filter_data(self):
        """Test data filtering."""
        self.analyzer.filter_data(
            domain_name='test.com',
            dma=501,
            action_types=['lead'],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5)
        )
        
        self.assertIsNotNone(self.analyzer.domain_data)
        self.assertTrue(len(self.analyzer.domain_data) > 0)
        self.assertTrue(all(self.analyzer.domain_data['assigned_domain'] == 'test.com'))
        self.assertTrue(all(self.analyzer.domain_data['dma'] == 501))
        self.assertTrue(all(self.analyzer.domain_data['action'] == 'lead'))

    def test_perform_eda(self):
        """Test exploratory data analysis."""
        self.analyzer.filter_data(domain_name='test.com')
        results = self.analyzer.perform_eda(frequency='daily')
        
        self.assertIn('domain_analysis', results)
        self.assertIn('market_analysis', results)
        self.assertTrue(len(results['domain_analysis']) > 0)
        self.assertTrue(len(results['market_analysis']) > 0)

    def test_analyze_timeseries(self):
        """Test time series analysis."""
        self.analyzer.filter_data(domain_name='test.com')
        results = self.analyzer.analyze_timeseries(frequency='daily')
        
        self.assertIn('time_series_data', results)
        self.assertIn('decomposition', results)
        self.assertTrue(len(results['time_series_data']) > 0)
        self.assertTrue(len(results['decomposition']) > 0)

    def test_forecast_hitcount(self):
        """Test forecasting."""
        self.analyzer.filter_data(domain_name='test.com')
        forecast_results = self.analyzer.forecast_hitcount(forecast_days=7)
        
        self.assertIsNotNone(forecast_results)
        self.assertTrue(len(forecast_results) > 0)
        self.assertTrue('predicted_hitCount' in forecast_results.columns)

    def test_generate_visualizations(self):
        """Test visualization generation."""
        self.analyzer.filter_data(domain_name='test.com')
        ts_results = self.analyzer.analyze_timeseries(frequency='daily')
        forecast_results = self.analyzer.forecast_hitcount(forecast_days=7)
        
        visualizations = self.analyzer.generate_visualizations(
            ts_results,
            forecast_results
        )
        
        self.assertIn('time_series', visualizations)
        self.assertIn('forecast', visualizations)
        for action in ts_results['time_series_data']['action'].unique():
            self.assertIn(f'seasonal_{action}', visualizations)

if __name__ == '__main__':
    unittest.main() 