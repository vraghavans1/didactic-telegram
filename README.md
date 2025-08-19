# Data Analyst Agent

AI-powered data analysis service that processes natural language queries and performs complex data analysis using OpenAI GPT-4o and web scraping.

## Features

- **Real OpenAI Integration**: Uses GPT-4o for authentic data analysis
- **Web Scraping**: Automatically scrapes Wikipedia for Titanic and other datasets  
- **Dynamic Analysis**: Returns real calculated values, not hardcoded responses
- **IIT Madras Compatible**: Returns 4-element array format for evaluation
- **Optimized Deployment**: Under 250MB for Vercel/Replit deployment

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

### Endpoint: `/api/analyze`

**POST Request:**
```bash
curl -X POST "http://localhost:5000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"query": "analyze titanic survival rates"}'
```

**Response Format:**
```json
[1, "Analysis text with real insights", 0.38, "data:image/png;base64,chart_data"]
```

### File Upload Support
```bash
curl -X POST "http://localhost:5000/api/analyze" \
  -F "file=@question.txt"
```

## Project Structure

```
data-analyst-agent/
├── api/
│   └── index.py          # FastAPI endpoints
├── data_agent.py         # Main orchestrator
├── query_processor.py    # OpenAI integration
├── visualization.py      # Chart generation
├── web_scraper.py       # Wikipedia scraper
├── requirements.txt     # Dependencies
├── vercel.json         # Deployment config
└── README.md           # This file
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

### Deployment Command Options

**For shell-based deployments (recommended):**
```bash
uvicorn api.index:app --host 0.0.0.0 --port 5000
```

**For Python script deployments:**
```bash
python main.py          # Main entry point
python start.py         # Alternative entry point
python run_direct.py    # Direct uvicorn execution
python deploy.py        # Production deployment script
```

**For shell script deployments:**
```bash
./run.sh              # Updated to use uvicorn directly
./start_server.sh      # Alternative shell script
```

**Files configured for deployment:**
- `Procfile`: For Heroku deployment with uvicorn
- `Procfile.replit`: For Replit-specific deployment
- `Dockerfile`: Uses uvicorn directly instead of Python scripts
- `run.sh`: Updated shell script with uvicorn
- `start_server.sh`: Alternative shell script

## Environment Variables

- `OPENAI_API_KEY`: Required for GPT-4o analysis

## Dependencies

- FastAPI: Web framework
- OpenAI: GPT-4o integration
- Beautiful Soup: Web scraping
- Matplotlib: Chart generation
- Requests: HTTP client

## License

MIT License