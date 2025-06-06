# AI-Powered Website Analytics Dashboard

This project provides an AI-powered analytics dashboard for website data analysis, featuring automated data processing, time series analysis, forecasting, and natural language interpretation of results.

## Features

- Interactive data visualization and filtering
- Time series analysis with trend detection
- XGBoost-based forecasting
- AI-powered insights using AWS Bedrock (Claude)
- Exportable analysis reports

## Project Structure

```
project_5/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/                  # Data directory
│   └── sample_data.csv    # Sample data file
├── src/                   # Source code
│   ├── data_analyzer.py   # Data analysis module
│   └── interpretation_agent.py  # AI interpretation module
└── tests/                 # Test files
    └── test_data_analyzer.py
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up AWS credentials:
   Create a `.env` file in the project root with the following variables:
   ```
   AWS_ACCESS_KEY_ID=your_access_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_key_here
   AWS_REGION=us-east-1
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload your website data CSV file through the web interface
2. Use the sidebar filters to select:
   - Date range
   - Domain
   - Brand
   - DMA (Designated Market Area)
   - Action type

3. View the analysis results:
   - Key metrics
   - Time series trends
   - Forecasts
   - AI-generated insights

4. Export the analysis report as HTML

## Data Format

The application expects a CSV file with the following columns:
- pixid: Unique pixel identifier
- unique_pixel: Boolean indicating if the pixel is unique
- assigned_domain: Website domain
- assigned_brand: Brand name
- dma: DMA code
- dma_name: DMA name
- date: Date of the data point
- action: Type of action (e.g., 'lead', 'view')
- hitCount: Number of hits

## Testing

Run the test suite:
```bash
python -m unittest tests/test_data_analyzer.py
```

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- plotly
- boto3
- python-dotenv

## License

MIT License 