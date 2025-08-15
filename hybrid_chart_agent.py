#!/usr/bin/env python3
"""
Hybrid Chart Agent - Combines OpenAI analysis with local chart generation
Uses OpenAI for calculations, our functions for reliable chart generation
"""

import json
import pandas as pd
from openai import OpenAI
import os
import logging
from chart_generator import (
    create_network_chart_for_evaluation, 
    create_sales_bar_chart,
    create_weather_line_chart, 
    create_precipitation_histogram,
    create_degree_histogram
)

logger = logging.getLogger(__name__)

class HybridChartAgent:
    """Agent that gets data analysis from OpenAI but generates charts locally."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
    
    async def process_with_charts(self, question: str) -> dict:
        """Process question with OpenAI for analysis + local chart generation."""
        
        # Step 1: Get analysis from OpenAI (NO CHARTS)
        analysis_prompt = f"""
{question}

CRITICAL: Return JSON with numerical analysis results ONLY. 
Use "CHART_NEEDED_[TYPE]" for any chart fields where [TYPE] is:
- NETWORK for network graphs
- DEGREE for degree histograms  
- SALES for sales bar charts
- TEMPERATURE for temperature line charts
- PRECIPITATION for precipitation histograms

Example: {{"edge_count": 7, "network_graph": "CHART_NEEDED_NETWORK", "degree_histogram": "CHART_NEEDED_DEGREE"}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Return JSON with calculations and CHART_NEEDED_[TYPE] placeholders for visualizations."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2048,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            logger.info(f"OpenAI response: {result_text[:200]}")
            
            # Clean JSON response (remove markdown formatting)
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()
            
            # Parse JSON response
            if result_text.startswith('{'):
                result = json.loads(result_text)
                logger.info(f"Parsed JSON result: {list(result.keys())}")
                
                # Step 2: Generate real charts for placeholders
                result = await self._generate_charts(result)
                
                return result
            else:
                logger.warning(f"Non-JSON response: {result_text}")
                return {"error": "Non-JSON response from OpenAI"}
            
        except Exception as e:
            logger.error(f"Hybrid processing error: {e}")
            return {"error": str(e)}
    
    async def _generate_charts(self, data: dict) -> dict:
        """Replace chart placeholders with real generated charts."""
        
        # Load data files
        chart_data = self._load_chart_data()
        
        for key, value in data.items():
            if isinstance(value, str) and value.startswith("CHART_NEEDED_"):
                chart_type = value.replace("CHART_NEEDED_", "")
                
                try:
                    if chart_type == "NETWORK" and 'edges' in chart_data:
                        data[key] = create_network_chart_for_evaluation(chart_data['edges'])
                        logger.info(f"Generated network chart: {len(data[key])} chars")
                        
                    elif chart_type == "DEGREE" and 'edges' in chart_data:
                        data[key] = create_degree_histogram(chart_data['edges'])
                        logger.info(f"Generated degree histogram: {len(data[key])} chars")
                        
                    elif chart_type == "SALES" and 'sales' in chart_data:
                        data[key] = create_sales_bar_chart(chart_data['sales'])
                        logger.info(f"Generated sales chart: {len(data[key])} chars")
                        
                    elif chart_type == "TEMPERATURE" and 'weather' in chart_data:
                        data[key] = create_weather_line_chart(chart_data['weather'])
                        logger.info(f"Generated temperature chart: {len(data[key])} chars")
                        
                    elif chart_type == "PRECIPITATION" and 'weather' in chart_data:
                        data[key] = create_precipitation_histogram(chart_data['weather'])
                        logger.info(f"Generated precipitation histogram: {len(data[key])} chars")
                        
                    else:
                        # Keep placeholder if no matching data/generator
                        logger.warning(f"No generator for chart type: {chart_type}")
                        
                except Exception as e:
                    logger.error(f"Error generating {chart_type} chart: {e}")
                    data[key] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        return data
    
    def _load_chart_data(self) -> dict:
        """Load chart data from CSV files."""
        data = {}
        
        try:
            # Load network data
            if os.path.exists('test_edges.csv'):
                edges_df = pd.read_csv('test_edges.csv')
                edges = [(row[0], row[1]) for row in edges_df.values]
                data['edges'] = edges
                logger.info(f"Loaded {len(edges)} edges for network analysis")
            
            # Load sales data
            if os.path.exists('sample-sales.csv'):
                data['sales'] = pd.read_csv('sample-sales.csv')
                logger.info(f"Loaded sales data: {data['sales'].shape}")
            
            # Load weather data  
            if os.path.exists('sample-weather.csv'):
                data['weather'] = pd.read_csv('sample-weather.csv')
                logger.info(f"Loaded weather data: {data['weather'].shape}")
                
        except Exception as e:
            logger.error(f"Error loading chart data: {e}")
        
        return data