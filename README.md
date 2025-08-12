# Data Analyst Agent

## Overview

This is an AI-powered data analysis service built for IIT Madras evaluation that processes natural language queries and performs complex data analysis tasks. The application combines OpenAI's GPT-4o with web scraping, statistical analysis, and data visualization capabilities to provide comprehensive data insights. Its purpose is to provide an advanced, dynamic data analysis solution capable of handling diverse analytical requests.

## User Preferences

- **Communication Style**: Simple, everyday language
- **Data Integrity**: CRITICAL - Never use hardcoded, fake, or simulated data
- **Error Handling**: Always throw errors if unable to access real data for analysis
- **No Fabrication**: Do not create fake data even if it matches expected format

## System Architecture

### Backend Architecture
- **Framework**: FastAPI-based REST API
- **Deployment**: Replit hosting with automatic workflows
- **AI Integration**: OpenAI GPT-4o for natural language processing and tool-calling
- **Web Scraping**: Beautiful Soup for content extraction
- **Visualization**: Matplotlib with base64 encoding for chart generation

### Design Principles
The application follows a clean, focused architecture with essential components:
- **ToolCallingAgent**: Main orchestrator using OpenAI function calling for dynamic analysis, capable of universal question processing with dynamic format recognition.
- **WebScraper**: Manages web data extraction from various sources (e.g., Wikipedia) including text content and table data.
- **DataVisualizer**: Creates charts and plots, outputting base64-encoded PNG images for web compatibility, supporting various plot types including scatterplots with regression lines.
- **API Layer**: Single FastAPI endpoint (`/api/` and `/api/analyze`) for universal question processing, adapting to various response formats (JSON objects, arrays, custom formats) and supporting file uploads.

### Data Flow
The system processes input queries, leveraging GPT-4o to analyze them and create analysis plans. It intelligently sources data (e.g., automatically detecting Titanic queries and scraping Wikipedia), performs real-time analysis, generates SVG charts that are base64-encoded, and returns results in the specified evaluation formats.

## External Dependencies

- **OpenAI API**: Used for GPT-4o natural language processing and tool-calling.
- **Web Sources**: Wikipedia and other websites for data extraction.
- **Python Libraries**: FastAPI, pandas, numpy, matplotlib, beautifulsoup4.# didactic-telegram
