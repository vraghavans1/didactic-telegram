# Data Analyst Agent

Advanced AI-powered data analysis system that processes multiple file formats and natural language queries using OpenAI GPT-4o. Achieved **100% evaluation performance** on IIT Madras testing with comprehensive multi-format file processing capabilities.

## Key Features

- **🎯 Perfect Evaluation Score**: 100% performance on IIT Madras evaluation system
- **📁 Multi-Format Support**: CSV, Excel (.xlsx/.xls), JSON, SQLite databases, PDFs, and images
- **🧠 Real AI Analysis**: OpenAI GPT-4o generates Python code for authentic data calculations
- **💾 File-to-Disk Architecture**: Upload → Save → Process workflow for evaluation compatibility
- **📊 Chart Generation**: Base64 PNG/WEBP charts under 100KB with progressive compression
- **⚡ High Performance**: Sub-15-second response times with comprehensive error handling
- **🌐 Web Scraping**: Wikipedia and general web data extraction capabilities
- **🚫 Zero Hardcoding**: All calculations performed on real uploaded data

## Quick Start

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd data-analyst-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

4. **Run the server**
```bash
uvicorn api.index:app --host 0.0.0.0 --port 5000
```

## API Usage

### Endpoint: `/api/`

**Multi-Format File Upload (Primary Method):**
```bash
curl -X POST "http://localhost:5000/api/" \
  -F "questions=@questions.txt" \
  -F "file=@data.csv" \
  -F "file=@data.xlsx" \
  -F "file=@data.json"
```

**Supported File Formats:**
- **CSV**: Primary format with perfect evaluation performance
- **Excel**: .xlsx/.xls files with openpyxl processing
- **JSON**: Array and object formats with automatic normalization  
- **SQLite**: .db/.sqlite databases with table detection
- **PDF**: Text extraction and analysis
- **Images**: PNG/JPG with OpenAI Vision API and OCR
- **Web URLs**: Automatic Wikipedia scraping

**Response Format (JSON):**
```json
{
  "field_name": "calculated_value",
  "chart": "data:image/png;base64,compressed_chart_data"
}
```

## Project Structure

```
data-analyst-agent/
├── api/
│   └── index.py                    # FastAPI REST API with multipart form handling
├── unified_data_agent.py           # Primary AI agent with multi-format processing
├── requirements.txt                # Python dependencies (openpyxl, PyPDF2, etc.)
├── Procfile.replit                 # Replit deployment configuration
├── README.md                       # Project documentation
├── replit.md                       # Technical architecture and user preferences
├── quick_test.py                   # Fast API validation testing
├── test_api_evaluation.py          # Comprehensive testing suite
├── test_curl_examples.sh           # Command line testing examples
├── sample_sales.csv                # Sample test data
├── .gitignore                      # Git exclusion rules
└── attached_assets/
    ├── output_*.json               # Evaluation results (100% score proof)
    └── stdout_*.txt                # Evaluation logs
```

## Deployment

### Vercel
1. Connect your GitHub repository to Vercel
2. Set `OPENAI_API_KEY` in environment variables
3. Deploy automatically

### Replit
1. Import repository to Replit
2. Add `OPENAI_API_KEY` secret
3. Run with the configured workflow using any of these commands:
   - `uvicorn api.index:app --host 0.0.0.0 --port 5000 --reload` (recommended)
   - `python main.py`
   - `python start.py`
   - `./start_server.sh`

### Production Deployment

**Recommended Command:**
```bash
uvicorn api.index:app --host 0.0.0.0 --port 5000 --reload
```

**Deployment Files:**
- `Procfile.replit`: Replit-specific deployment configuration
- Automatic workflow configuration for seamless deployment

## Environment Variables

- `OPENAI_API_KEY`: Required for GPT-4o analysis

## Dependencies

**Core Libraries:**
- **FastAPI**: REST API framework with multipart form support
- **OpenAI**: GPT-4o integration for AI analysis and code generation
- **pandas**: Data processing and analysis
- **numpy**: Numerical computing
- **matplotlib**: Chart generation with base64 encoding
- **networkx**: Network analysis capabilities
- **seaborn**: Statistical data visualization

**File Processing:**
- **openpyxl**: Excel file processing (.xlsx/.xls)
- **PyPDF2**: PDF text extraction and analysis
- **sqlite3**: SQLite database processing
- **PIL (Pillow)**: Image processing and analysis

**Enhanced Features:**
- **beautifulsoup4**: Web scraping and HTML parsing
- **opencv-python**: Advanced image processing
- **pytesseract**: OCR (Optical Character Recognition)
- **aiofiles**: Asynchronous file operations
- **trafilatura**: Advanced web content extraction

## Performance Metrics

- **✅ Evaluation Score**: 100% (Perfect performance on IIT Madras evaluation)
- **⚡ Response Time**: Sub-15-second analysis with chart generation
- **📊 Chart Size**: Base64 images consistently under 100KB
- **💎 Data Integrity**: Real calculations only, no hardcoded or sample data fallbacks
- **🎯 Format Compliance**: Exact JSON structure matching evaluation requirements

## Testing

**Quick Validation:**
```bash
python quick_test.py
```

**Comprehensive Testing:**
```bash
python test_api_evaluation.py
```

**Command Line Testing:**
```bash
bash test_curl_examples.sh
```

## License

MIT License