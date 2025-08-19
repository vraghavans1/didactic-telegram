# Data Analyst Agent

## Overview
This AI-powered data analysis service processes natural language queries to perform complex data analysis, statistical analysis, and data visualization. It integrates OpenAI's GPT-4o with web scraping and multi-format file processing capabilities. The project's vision is to provide an advanced, dynamic data analysis solution capable of handling diverse analytical requests, aiming for comprehensive data insights for users.

## User Preferences
- **Communication Style**: Simple, everyday language
- **Data Integrity**: CRITICAL - Never use hardcoded, fake, or simulated data
- **Error Handling**: Always throw errors if unable to access real data for analysis
- **No Fabrication**: Do not create fake data even if it matches expected format

## System Architecture
### Deployment Ready Structure
- `api/index.py`: FastAPI REST API with unified agent integration and multipart form handling.
- `unified_data_agent.py`: Primary agent for multi-format file processing using OpenAI.
- `requirements.txt`: Project dependencies.
- `Procfile.replit`: Replit deployment configuration.
- `README.md`: Project documentation.
- `replit.md`: Architecture documentation and user preferences.
- Testing scripts (`quick_test.py`, `test_api_evaluation.py`, `test_curl_examples.sh`).
- Sample data (`sample_sales.csv`).

### Backend Architecture
- **Framework**: FastAPI-based REST API.
- **Deployment**: Replit hosting.
- **AI Integration**: OpenAI GPT-4o for natural language processing and tool-calling.
- **Web Scraping**: Beautiful Soup for content extraction.
- **Visualization**: Matplotlib with base64 encoding and compression.

### Design Principles
The application employs a unified architecture emphasizing format compliance and performance:
- **UnifiedDataAgent**: A single OpenAI GPT-4o agent manages the entire analysis pipeline.
- **Multi-Format File Processing**: Intelligent detection and processing of CSV, Excel, JSON, SQLite, PDF, and image files.
- **File-to-Disk Architecture**: Files are saved to unique request folders (`uploads/{uuid}/`) on disk and processed from there, ensuring isolation and compatibility.
- **Dynamic Code Execution**: OpenAI generates Python code for execution with an injected DataFrame for authentic data analysis.
- **Format Parsing**: Questions are parsed to extract exact field names and types for evaluation compliance.
- **Progressive Chart Compression**: Multiple fallback strategies (DPI reduction, WEBP conversion, quality optimization) ensure generated charts are under 100KB.
- **Zero Hardcoding**: All calculations are performed on actual uploaded data using dynamically generated analysis code.
- **API Layer**: A FastAPI endpoint (`/api/`) handles multipart form parsing and error handling.
- **Priority-Based Loading**: Files are processed in the order: CSV → Excel → JSON → Database → PDF → Image → Text.

### Data Flow
1. **Request Processing**: Multipart form data with `questions.txt` and data files are received via FastAPI.
2. **File Management**: Files are saved to isolated, unique request folders.
3. **Format Detection**: Automatic categorization of uploaded files by type.
4. **Data Loading**: Priority-based loading using appropriate libraries (pandas, openpyxl, PyPDF2, sqlite3, PIL).
5. **AI Analysis**: GPT-4o generates Python code for data analysis.
6. **Chart Generation**: Base64-encoded PNG/WEBP charts are generated with progressive compression.
7. **Response Assembly**: JSON results are formatted with exact field names and types.

## External Dependencies
- **OpenAI API**: For GPT-4o integration and dynamic code generation.
- **Web Sources**: Wikipedia and other websites for web scraping.
- **Core Libraries**: FastAPI, pandas, numpy, matplotlib, networkx, seaborn.
- **File Processing**: `openpyxl` (Excel), `PyPDF2` (PDF), `sqlite3` (SQLite databases), `PIL` (images).
- **Enhanced Features**: `opencv-python`, `pytesseract` (OCR), `aiofiles` (asynchronous file operations), `trafilatura` (advanced web scraping).