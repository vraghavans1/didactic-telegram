import os
import re
import json
import base64
import tempfile
import sys
import subprocess
import logging
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup

# Optional PIL for image optimization
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OpenAI integration
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDataAgent:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        
    def parse_keys_and_types(self, raw_questions: str):
        """
        Parse the questions file to extract exact field names and types required by IIT Madras.
        Returns: keys_list, type_map
        """
        # Enhanced regex to capture full type descriptions (not just single words)
        pattern = r"-\s*`([^`]+)`\s*:\s*([^\n]+)"
        matches = re.findall(pattern, raw_questions)
        
        type_map_def = {
            "number": float,
            "string": str,
            "integer": int,
            "int": int,
            "float": float
        }
        
        # Enhanced type mapping to handle complex descriptions
        type_map = {}
        keys_list = []
        
        for key, type_desc in matches:
            keys_list.append(key)
            type_desc_lower = type_desc.strip().lower()
            
            # Handle complex type descriptions
            if "number" in type_desc_lower:
                type_map[key] = float
            elif "string" in type_desc_lower or "base64" in type_desc_lower:
                type_map[key] = str
            elif "integer" in type_desc_lower or "int" in type_desc_lower:
                type_map[key] = int
            else:
                # Default to string for any unrecognized types
                type_map[key] = str
        
        return keys_list, type_map
    
    def create_plot_to_base64_helper(self):
        """Progressive compression helper for charts under 100KB"""
        return '''
def plot_to_base64(max_bytes=100000):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    
    # Progressive DPI reduction
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    
    # WEBP conversion if PIL available
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        
        # Lower quality WEBP
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    
    # Last resort: very low DPI PNG
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    def create_scraping_helper(self):
        """Web scraping helper function"""
        return '''
def scrape_url_to_dataframe(url: str):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from io import StringIO, BytesIO
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        # Try different data formats
        if url.lower().endswith('.csv') or 'text/csv' in resp.headers.get('Content-Type', ''):
            df = pd.read_csv(BytesIO(resp.content))
        elif 'application/json' in resp.headers.get('Content-Type', ''):
            data = resp.json()
            df = pd.json_normalize(data)
        else:
            # HTML tables
            try:
                tables = pd.read_html(StringIO(resp.text))
                df = tables[0] if tables else pd.DataFrame()
            except:
                # Fallback to text extraction
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = soup.get_text(separator='\\n', strip=True)
                df = pd.DataFrame({'text': [text]})
        
        # Clean column names
        df.columns = df.columns.map(str).str.replace(r'\\[.*\\]', '', regex=True).str.strip()
        return df
        
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})
'''

    def execute_python_code(self, code: str, df_pickle_path: str = None, timeout: int = 120) -> Dict[str, Any]:
        """Execute generated Python code safely with DataFrame injection"""
        
        # Create the complete Python script
        preamble = [
            "import json, sys, warnings",
            "warnings.filterwarnings('ignore')",
            "import pandas as pd, numpy as np",
            "import matplotlib",
            "matplotlib.use('Agg')",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from io import BytesIO",
            "import base64, requests",
            "from bs4 import BeautifulSoup",
            "import networkx as nx"
        ]
        
        # Add PIL if available
        if PIL_AVAILABLE:
            preamble.append("from PIL import Image")
            
        # Inject DataFrame if provided
        if df_pickle_path and os.path.exists(df_pickle_path):
            preamble.append(f"df = pd.read_pickle(r'{df_pickle_path}')")
            preamble.append("data = df.to_dict(orient='records')")
        else:
            preamble.append("data = {}")
            
        # Build complete script
        script_lines = []
        script_lines.extend(preamble)
        script_lines.append(self.create_plot_to_base64_helper())
        script_lines.append(self.create_scraping_helper())
        script_lines.append("\nresults = {}")
        script_lines.append(code)
        script_lines.append("\nprint(json.dumps({'status': 'success', 'result': results}, default=str), flush=True)")
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write("\n".join(script_lines))
            tmp_path = tmp.name
        
        try:
            # Execute with timeout
            completed = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if completed.returncode != 0:
                return {
                    "status": "error",
                    "message": completed.stderr.strip() or completed.stdout.strip()
                }
            
            # Parse JSON output
            output = completed.stdout.strip()
            try:
                return json.loads(output)
            except json.JSONDecodeError as e:
                return {
                    "status": "error", 
                    "message": f"JSON parse error: {str(e)}", 
                    "raw": output
                }
                
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Execution timed out after {timeout}s"}
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
                if df_pickle_path and os.path.exists(df_pickle_path):
                    os.unlink(df_pickle_path)
            except Exception:
                pass

    def analyze_data(self, questions: str, csv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main analysis function that processes questions and data to produce exact IIT Madras format.
        No hardcoded values - all calculations use real data.
        """
        start_time = time.time()
        
        try:
            # Parse required format from questions
            required_keys, type_map = self.parse_keys_and_types(questions)
            logger.info(f"Required fields: {required_keys}")
            
            # Save DataFrame to temp pickle for injection
            df_pickle_path = None
            if not csv_data.empty:
                df_pickle_path = tempfile.mktemp(suffix='.pkl')
                csv_data.to_pickle(df_pickle_path)
            
            # Create analysis prompt
            prompt = f"""
You are a data analyst. I have uploaded a CSV dataset and need you to analyze it according to specific requirements.

Dataset shape: {csv_data.shape}
Dataset columns: {list(csv_data.columns)}
Sample data: {csv_data.head(3).to_dict()}

Questions to answer: {questions}

Required output format (EXACT field names and types):
{json.dumps({k: str(v.__name__) for k, v in type_map.items()}, indent=2)}

Generate Python code that:
1. Uses the injected 'df' DataFrame variable containing the CSV data
2. Performs real calculations on the actual data (no hardcoded values)
3. Creates charts using plot_to_base64() function for any visualization fields
4. Stores all results in a 'results' dictionary with EXACT field names above
5. Handles any data format variations robustly

Important rules:
- Use ONLY real data calculations
- Chart fields should contain base64 encoded PNG/WEBP images
- Network analysis should use networkx if the data contains edges/relationships
- For sales data, calculate actual totals, correlations, and statistics
- For weather data, compute real temperature and precipitation metrics
- All numerical values must be calculated from the actual dataset

Generate ONLY the Python code, no explanations:
"""

            # Get code generation from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            generated_code = response.choices[0].message.content
            if generated_code:
                generated_code = generated_code.strip()
                
                # Clean code if wrapped in markdown
                if generated_code.startswith('```python'):
                    generated_code = generated_code[9:]
                if generated_code.endswith('```'):
                    generated_code = generated_code[:-3]
                generated_code = generated_code.strip()
            else:
                return {"error": "No code generated by OpenAI"}
            
            logger.info(f"Generated {len(generated_code)} chars of Python code")
            
            # Execute the generated code
            execution_result = self.execute_python_code(generated_code, df_pickle_path, timeout=150)
            
            if execution_result.get("status") == "success":
                results = execution_result.get("result", {})
                
                # Apply type casting to ensure correct format
                final_results = {}
                for key in required_keys:
                    if key in results:
                        try:
                            if type_map[key] == str:
                                final_results[key] = str(results[key])
                            elif type_map[key] in [int, float]:
                                final_results[key] = type_map[key](results[key])
                            else:
                                final_results[key] = results[key]
                        except (ValueError, TypeError):
                            final_results[key] = results[key]
                    else:
                        # Field missing - this should not happen with good code generation
                        logger.warning(f"Missing required field: {key}")
                        final_results[key] = None
                
                elapsed = time.time() - start_time
                logger.info(f"Analysis completed in {elapsed:.2f}s")
                return final_results
            else:
                logger.error(f"Code execution failed: {execution_result.get('message', 'Unknown error')}")
                return {"error": f"Analysis failed: {execution_result.get('message', 'Code execution error')}"}
                
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {"error": f"Analysis error: {str(e)}"}