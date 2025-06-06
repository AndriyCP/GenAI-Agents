import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
from src.data_analyzer import DataAnalyzer
from src.interpretation_agent import InterpretationAgent
from langchain import hub
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_aws import ChatBedrock
from langchain_community.tools.tavily_search import TavilySearchResults
import boto3
import tempfile
import base64
import requests
import io
import re

# Load environment variables
load_dotenv()

class NewsSummaryAgent:
    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not self.tavily_api_key:
            raise ValueError("Missing TAVILY_API_KEY from environment")
        
        # Initialize Tavily search tool
        self.tavily = TavilySearchResults(tavily_api_key=self.tavily_api_key, k=3)
        self.expanded_tavily_tool = Tool.from_function(
            name="Broader Tavily Search",
            description="Searches news about the U.S. car market from multiple angles using Tavily.",
            func=self.expanded_tavily_search
        )
        
        # Initialize LLM
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_k": 250,
                "top_p": 0.9,
                "stop_sequences": ["\n\nHuman"]
            }
        )
        
        # Create agent
        prompt = hub.pull("hwchase17/structured-chat-agent")
        self.agent = create_structured_chat_agent(llm=self.llm, tools=[self.expanded_tavily_tool], prompt=prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=[self.expanded_tavily_tool],
            verbose=True,
            handle_parsing_errors=True
        )

    def expanded_tavily_search(self, user_query: str) -> str:
        today = datetime.now().strftime("%B %d, %Y")
        
        queries = [
            f"{user_query} {today}",
            f"U.S. car inventory levels {today}",
            f"U.S. vehicle pricing news {today}",
            f"automotive supply chain issues {today}",
            f"EV sales trends {today}",
            f"auto dealer news {today}"
        ]
        
        results = []
        for q in queries:
            res = self.tavily.run(q)
            if res:
                results.extend(res)
        
        # Deduplicate by URL
        seen = set()
        unique = []
        for r in results:
            if r["url"] not in seen:
                unique.append(r)
                seen.add(r["url"])
        
        if not unique:
            return f"No relevant updates found for {today}."
        
        return "\n\n".join(
            f"{r['title']}\n{r['url']}\n{r.get('content', '')[:300]}..." for r in unique[:5]
        )

    def generate_news_summary(self):
        from datetime import date
        today = date.today().strftime("%B %d, %Y")
        query = (
            f"Search for and summarize the most relevant U.S. car market news as of {today}. "
            f"Focus only on information published today or in the last 48 hours — do not include general 2025 forecasts or outdated articles. "
            f"Highlight developments that affect dealership operations, such as vehicle pricing changes, inventory levels, sales trends, and logistics or supply chain issues. "
            f"The summary should be concise, factual, and directly useful for car dealership managers. "
            f"If there are no credible updates published today or in the last 48 hours, respond with: 'No relevant updates found for {today}.'"
        )

        # Use the LangChain agent with Tavily tool
        prompt = hub.pull("hwchase17/structured-chat-agent")
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=[self.expanded_tavily_tool],
            prompt=prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[self.expanded_tavily_tool],
            verbose=False,
            handle_parsing_errors=True
        )
        response = agent_executor.invoke({"input": query})
        return response['output']

class PodcastGenerator:
    def __init__(self, bedrock_client):
        """Initialize the podcast generator with AWS services."""
        self.bedrock = bedrock_client
        self.elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
        if not self.elevenlabs_api_key:
            raise ValueError("Missing ELEVENLABS_API_KEY from environment")
        
    def generate_podcast_script(self, insights, news_summary, prizm_recommendations=None):
        """Generate a podcast script from insights, news, and PRIZM recommendations."""
        prompt = f"""Create a conversational podcast script that presents the following analytics insights and industry news in an engaging way. 
        The script should be in a natural, conversational tone, as if being presented by a data analyst and industry expert.
        Include brief musical interludes between sections using [MUSIC] markers.
        Keep the total length to about 3-4 minutes when read aloud.

        Analytics Insights:
        {insights}

        Industry News:
        {news_summary}
"""

        if prizm_recommendations:
            prompt += f"""

        PRIZM Segment Recommendations:
        {prizm_recommendations}

        Please incorporate the PRIZM segment recommendations into the marketing recommendations section of the podcast.
"""

        prompt += """

        Format the script with clear sections and natural transitions. Include brief pauses marked with [PAUSE].
        Start with a brief introduction and end with a conclusion."""

        try:
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'max_tokens': 2000,
                    'temperature': 0.4,
                    'top_p': 1,
                    'anthropic_version': 'bedrock-2023-05-31'
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            return f"Error generating podcast script: {str(e)}"

    def text_to_speech(self, script):
        """Convert the script to speech using ElevenLabs."""
        try:
            # Remove [MUSIC] and [PAUSE] markers for the speech
            clean_script = script.replace('[MUSIC]', '').replace('[PAUSE]', '')
            
            # Format numbers for better speech synthesis
            def format_number(match):
                number = match.group(0)
                # Remove commas and convert to words
                number = number.replace(',', '')
                return f"{number} "
            
            # Find and format numbers with commas
            clean_script = re.sub(r'\d{1,3}(,\d{3})*', format_number, clean_script)
            
            # ElevenLabs API endpoint
            url = "https://api.elevenlabs.io/v1/text-to-speech/UgBBYS2sOqTuMpoF3BR0"
            
            # Headers
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            # Request body
            data = {
                "text": clean_script,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.33,
                    "similarity_boost": 0.75,
                    "style": 0.05,
                    "use_speaker_boost": True
                }
            }
            
            # Make the API request
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_file.write(response.content)
                    return temp_file.name
            else:
                st.error(f"Error from ElevenLabs API: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            return None

# Initialize session state for forecast data
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'news_summary' not in st.session_state:
    st.session_state.news_summary = None
if 'podcast_audio' not in st.session_state:
    st.session_state.podcast_audio = None
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = None
if 'podcast_audio_bytes' not in st.session_state:
    st.session_state.podcast_audio_bytes = None

# Initialize AWS Bedrock client and agents
try:
    import boto3
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv('AWS_REGION', 'us-west-2'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    interpreter = InterpretationAgent(bedrock)
    news_agent = NewsSummaryAgent(bedrock)
    podcast_generator = PodcastGenerator(bedrock)
    aws_available = True
except Exception as e:
    st.warning("AWS Bedrock integration is not available. AI interpretation features will be disabled.")
    st.info("To enable AI interpretation, please set up AWS credentials in the .env file")
    aws_available = False
    interpreter = None
    news_agent = None
    podcast_generator = None

# Load PRIZM data (cache for performance)
@st.cache_data
def load_prizm_data():
    prizm_dmas = pd.read_csv(os.path.join("data", "PRIZM_info", "TopPRIZM_for_DMAs.csv"))
    prizm_desc = pd.read_csv(os.path.join("data", "PRIZM_info", "prizm_segments_descriptions.csv"))
    prizm_dmas.columns = prizm_dmas.columns.str.strip()
    prizm_desc.columns = prizm_desc.columns.str.strip()
    return prizm_dmas, prizm_desc

prizm_dmas, prizm_desc = load_prizm_data()

def get_prizm_recommendation(top_dma_code):
    dma_row = prizm_dmas[prizm_dmas['DMA_GCODE'] == top_dma_code]
    if dma_row.empty:
        return None
    segs = []
    for i in range(1, 4):
        seg_col = f'Segment_{i}'
        if seg_col in dma_row.columns:
            seg_name = dma_row.iloc[0][seg_col]
            if pd.notna(seg_name):
                # Lookup description (more robust to whitespace/case)
                desc_row = prizm_desc[prizm_desc['Segment Name'].str.strip().str.lower() == seg_name.strip().lower()]
                desc = desc_row['Description'].iloc[0] if not desc_row.empty else ""
                segs.append((seg_name, desc))
    return segs

def main():
    # Create columns for side-by-side layout
    col1, col2 = st.columns([1, 4]) # Adjust ratios as needed

    with col1:
        # Add logo image
        st.image("/Users/andriy/Desktop/GSB503-Claritas/Data_Agent/design/claritas_mbs_logo-2.jpeg", width=80, use_container_width=False)

    with col2:
        # Change text title color to orange and adjust vertical alignment
        st.markdown("<h1 style='color: #FF5F15; margin-top: 0; padding-top: 20px; font-family: \'Open Sans\', sans-serif;'>Web Analytics AI Agent</h1>", unsafe_allow_html=True)

    st.write("Upload your website data CSV file to begin analysis")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        analyzer = DataAnalyzer(df)
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Date range filter
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Domain filter
        domains = ['All'] + list(df['assigned_domain'].unique())
        selected_domain = st.sidebar.selectbox("Select Domain", domains)
        
        # Brand filter
        brands = ['All'] + list(df['assigned_brand'].unique())
        selected_brand = st.sidebar.selectbox("Select Brand", brands)
        
        # --- Reset state if brand or domain changes ---
        if 'prev_selected_brand' not in st.session_state:
            st.session_state.prev_selected_brand = None
        if 'prev_selected_domain' not in st.session_state:
            st.session_state.prev_selected_domain = None
        
        if (selected_brand != st.session_state.prev_selected_brand) or (selected_domain != st.session_state.prev_selected_domain):
            st.session_state.forecast_data = None
            st.session_state.ai_insights = None
            st.session_state.news_summary = None
            st.session_state.podcast_script = None
            st.session_state.podcast_audio_bytes = None
        
        st.session_state.prev_selected_brand = selected_brand
        st.session_state.prev_selected_domain = selected_domain
        
        # DMA filter
        dmas = ['All'] + list(df['dma_name'].unique())
        selected_dma = st.sidebar.selectbox("Select DMA", dmas)
        
        # Action type filter
        actions = ['All'] + list(df['action'].unique())
        selected_action = st.sidebar.selectbox("Select Action Type", actions)
        
        # Apply filters
        filtered_df = analyzer.filter_data(
            start_date=date_range[0],
            end_date=date_range[1],
            domain=selected_domain if selected_domain != 'All' else None,
            brand=selected_brand if selected_brand != 'All' else None,
            dma=selected_dma if selected_dma != 'All' else None,
            action=selected_action if selected_action != 'All' else None
        )
        
        # Main content
        st.header("Data Overview")
        
        # Display key metrics with wider columns for large numbers
        col1, col2, col3, col4 = st.columns([2.5, 2.5, 1, 1])
        with col1:
            st.metric("Total Hits", f"{filtered_df['hitCount'].sum():,}")
        with col2:
            st.metric("Average Daily Hits", f"{round(filtered_df.groupby('date')['hitCount'].sum().mean()):,}")
        with col3:
            st.metric("Unique Domains", filtered_df['assigned_domain'].nunique())
        with col4:
            st.metric("Unique DMAs", filtered_df['dma_name'].nunique())
        
        # --- Top 3 DMAs by % of hitCount for selected Domain or Brand ---
        show_top_dmas = (selected_domain != 'All') or (selected_brand != 'All')

        st.subheader("Top 3 DMAs by Hit Share")
        if show_top_dmas:
            # Filter for only the selected domain or only the selected brand
            if selected_domain != 'All':
                dmas_df = df[df['assigned_domain'] == selected_domain]
            elif selected_brand != 'All':
                dmas_df = df[df['assigned_brand'] == selected_brand]
            else:
                dmas_df = df  # fallback, shouldn't happen

            # Further filter for selected action type if not 'All'
            if selected_action != 'All':
                dmas_df = dmas_df[dmas_df['action'] == selected_action]

            # Group by DMA and sum hitCount
            dma_hits = dmas_df.groupby('dma_name')['hitCount'].sum().reset_index()
            total_hits = dma_hits['hitCount'].sum()
            dma_hits['percent'] = 100 * dma_hits['hitCount'] / total_hits if total_hits > 0 else 0
            top3 = dma_hits.sort_values('percent', ascending=False).head(3)

            if not top3.empty and total_hits > 0:
                # Show as text
                top3_text = ", ".join(
                    f"{row['dma_name']} ({row['percent']:.1f}%)"
                    for _, row in top3.iterrows()
                )
                st.markdown(f"**Top 3 DMAs:** {top3_text}")

                # Show as bar chart
                fig = px.bar(
                    top3,
                    x='percent',
                    y='dma_name',
                    orientation='h',
                    labels={'percent': '% of Hits', 'dma_name': 'DMA'},
                    text=top3['percent'].apply(lambda x: f"{x:.1f}%"),
                    color='percent',
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters.")
        else:
            st.info("Select Domain or Brand to see top DMAs.")
        
        # Time series analysis
        st.header("Time Series Analysis")
        time_series = analyzer.analyze_time_series(filtered_df)
        
        # Plot time series
        fig = px.line(time_series, x='date', y='hitCount', title='Daily Hit Count Trend')
        st.plotly_chart(fig, key="time_series_chart")
        
        # Forecasting
        st.header("Forecasting")
        forecast_days = st.slider("Select number of days to forecast", 7, 30, 14)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                forecast = analyzer.forecast_hit_count(filtered_df, forecast_days)
                st.session_state.forecast_data = forecast
        
        # Display forecast if it exists in session state
        if st.session_state.forecast_data is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast_data['date'],
                y=st.session_state.forecast_data['hitCount'],
                name='Historical Data',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast_data['date'],
                y=st.session_state.forecast_data['forecast'],
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title='Hit Count Forecast',
                xaxis_title='date',
                yaxis_title='hitCount',
                showlegend=True
            )
            st.plotly_chart(fig, key="forecast_chart")
        
        # Generate AI interpretation
        if aws_available:
            if st.button("Generate AI Interpretation"):
                with st.spinner("Generating insights..."):
                    # Get news summary
                    news_summary = news_agent.generate_news_summary()
                    st.session_state.news_summary = news_summary
                    
                    # Generate interpretation with news context
                    interpretation = interpreter.generate_interpretation(
                        filtered_df,
                        time_series,
                        st.session_state.forecast_data,
                        news_context=news_summary,
                        top_dmas=top3 # Pass the top3 DataFrame
                    )
                    st.session_state.ai_insights = interpretation
            
            # Display AI insights and news summary if they exist
            if st.session_state.ai_insights:
                st.header("AI-Generated Insights")
                st.write(st.session_state.ai_insights)
                
                if st.session_state.news_summary:
                    st.header("Industry News Summary")
                    st.write(st.session_state.news_summary)
                    
                    # --- PRIZM Segments Recommendation ---
                    top_dma_name = None
                    top_dma_code = None
                    prizm_summary_text = None

                    if show_top_dmas:
                        if not top3.empty and total_hits > 0:
                            top_dma_name = top3.iloc[0]['dma_name']
                            # Map DMA name to code using the uploaded data
                            dma_row = df[df['dma_name'] == top_dma_name]
                            if not dma_row.empty and 'dma' in dma_row.columns:
                                top_dma_code = dma_row.iloc[0]['dma']
                            else:
                                top_dma_code = None
                    
                    if top_dma_code:
                        prizm_recs = get_prizm_recommendation(top_dma_code)
                        if prizm_recs:
                            # Format as bullet points
                            prizm_summary_text = "Based on your Top DMA, consider tailoring your marketing efforts towards these PRIZM segments:\n\n"
                            prizm_summary_text += "\n".join([f"- **{name}**: {desc}" for name, desc in prizm_recs])
                            st.subheader("PRIZM Segments Recommendations")
                            st.markdown(prizm_summary_text)
                        else:
                            st.info("No PRIZM segment recommendations available for your top DMA.")
                    else:
                        st.info("No Top DMA identified for PRIZM segment recommendations.")
                    
                    # Add podcast generation option
                    if st.button("Generate Podcast"):
                        with st.spinner("Generating podcast..."):
                            # Generate podcast script
                            script = podcast_generator.generate_podcast_script(
                                st.session_state.ai_insights,
                                st.session_state.news_summary,
                                prizm_recommendations=prizm_summary_text # Pass PRIZM text
                            )
                            st.session_state.podcast_script = script
                            
                            # Convert to speech
                            audio_file = podcast_generator.text_to_speech(script)
                            if audio_file:
                                # Read audio bytes and store in session state
                                with open(audio_file, 'rb') as f:
                                    st.session_state.podcast_audio_bytes = f.read()
                                
                                # Clean up the temporary file
                                try:
                                    os.unlink(audio_file)
                                except:
                                    pass
                    
                    # Display podcast content if it exists
                    if st.session_state.podcast_script:
                        st.subheader("Podcast Script")
                        st.text_area("Script", st.session_state.podcast_script, height=300)
                        
                        if st.session_state.podcast_audio_bytes:
                            st.subheader("Listen to Podcast")
                            st.audio(st.session_state.podcast_audio_bytes, format='audio/mp3')
                            
                            # Add download button for the podcast
                            st.download_button(
                                label="Download Report",
                                data=st.session_state.podcast_audio_bytes,
                                file_name="analytics_podcast.mp3",
                                mime="audio/mp3"
                            )
        else:
            st.info("AI interpretation is currently disabled. Please set up AWS credentials to enable this feature.")
        
        # Export options
        st.header("Export Options")
        if st.button("Export Analysis Report"):
            # Generate and save report
            report = analyzer.generate_report(filtered_df)
            
            # Include AI insights and news summary in the report if available
            if st.session_state.ai_insights or st.session_state.news_summary:
                report += "\n\nAI-Generated Insights:\n"
                if st.session_state.ai_insights:
                    report += st.session_state.ai_insights
                if st.session_state.news_summary:
                    report += "\n\nIndustry News Summary:\n"
                    report += st.session_state.news_summary
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="analysis_report.html",
                mime="text/html"
            )

if __name__ == "__main__":
    main() 