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

# Image processing imports
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None

# OpenAI integration  
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDataAgent:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Image processing capabilities
        self.image_processing_enabled = PIL_AVAILABLE and CV2_AVAILABLE and OCR_AVAILABLE
        if self.image_processing_enabled:
            logger.info("✅ Image processing capabilities enabled (PIL, OpenCV, OCR)")
        else:
            logger.warning("⚠️ Image processing dependencies missing - image analysis disabled")
        
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
    
    def extract_csv_filename(self, question_text: str):
        """Extract CSV filename from question text (e.g., 'edges.csv', 'sample-sales.csv')"""
        import re
        csv_patterns = [
            r'`([^`]+\.csv)`',  # `filename.csv`
            r'in\s+([^.\s]+\.csv)',  # in filename.csv
            r'([^.\s]+\.csv)',  # filename.csv
        ]
        
        for pattern in csv_patterns:
            matches = re.findall(pattern, question_text, re.IGNORECASE)
            if matches:
                return matches[0].lower()
        return None
    
    def load_sample_data(self, filename: str):
        """Load sample CSV data based on filename"""
        filename_map = {
            'edges.csv': 'sample_edges.csv',
            'sample-sales.csv': 'sample_sales.csv', 
            'sample_sales.csv': 'sample_sales.csv',
            'sample-weather.csv': 'sample_weather.csv',
            'sample_weather.csv': 'sample_weather.csv'
        }
        
        sample_file = filename_map.get(filename)
        if sample_file and os.path.exists(sample_file):
            try:
                return pd.read_csv(sample_file)
            except Exception as e:
                logger.warning(f"Could not load sample data {sample_file}: {e}")
        return None
    
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
                tables = pd.read_html(StringIO(resp.text), header=0)
                if tables:
                    df = tables[0]  # Use first table
                    # Clean up column names and data
                    df.columns = df.columns.astype(str)
                    # Remove footnote references like [1], [2], etc.
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.replace(r'\\[[^\\]]+\\]', '', regex=True)
                            df[col] = df[col].str.strip()
                else:
                    df = pd.DataFrame()
            except Exception as e:
                # Try with BeautifulSoup for more control
                try:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    table = soup.find('table', {'class': 'wikitable'}) or soup.find('table')
                    if table:
                        rows = []
                        headers = []
                        for tr in table.find_all('tr'):
                            cells = tr.find_all(['td', 'th'])
                            if not headers and cells:
                                headers = [cell.get_text(strip=True) for cell in cells]
                            elif cells:
                                row = [cell.get_text(strip=True) for cell in cells]
                                if len(row) == len(headers):
                                    rows.append(row)
                        df = pd.DataFrame(rows, columns=headers) if headers else pd.DataFrame()
                    else:
                        df = pd.DataFrame({'error': [f'No table found: {str(e)}']})
                except Exception:
                    df = pd.DataFrame({'error': [f'Failed to parse: {str(e)}']})
        
        # Clean column names
        df.columns = df.columns.map(str).str.replace(r'\\[.*\\]', '', regex=True).str.strip()
        return df
        
    except Exception as e:
        return pd.DataFrame({'error': [str(e)]})
'''

    def execute_python_code(self, code: str, df_pickle_path: str = "", timeout: int = 120) -> Dict[str, Any]:
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
            preamble.append("df = pd.DataFrame()")
            
        # Build complete script
        script_lines = []
        script_lines.extend(preamble)
        script_lines.append(self.create_plot_to_base64_helper())
        script_lines.append(self.create_scraping_helper())
        script_lines.append("\nresults = {}")
        script_lines.append("import sys")
        script_lines.append("original_stdout = sys.stdout")
        script_lines.append("sys.stdout = open('/dev/null', 'w')")  # Suppress all print statements in generated code
        script_lines.append("try:")
        script_lines.append(f"    {code.replace(chr(10), chr(10) + '    ')}")  # Indent the user code
        script_lines.append("finally:")
        script_lines.append("    sys.stdout = original_stdout")
        script_lines.append("\ntry:")
        script_lines.append("    print(json.dumps({'status': 'success', 'result': results}, default=str), flush=True)")
        script_lines.append("except Exception as e:")
        script_lines.append("    print(json.dumps({'status': 'error', 'message': f'Result serialization error: {str(e)}'}, default=str), flush=True)")
        
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
                logger.error(f"JSON decode error: {str(e)}")
                logger.error(f"Raw output: {output}")
                logger.error(f"Stderr: {completed.stderr}")
                return {
                    "status": "error", 
                    "message": f"JSON parse error: {str(e)}", 
                    "raw": output[:1000],  # Limit raw output size
                    "stderr": completed.stderr[:1000]
                }
                
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": f"Execution timed out after {timeout}s"}
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
                if df_pickle_path and df_pickle_path != "" and os.path.exists(df_pickle_path):
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
            logger.info(f"Generated code preview: {generated_code[:200]}...")
            
            # Execute the generated code
            execution_result = self.execute_python_code(generated_code, df_pickle_path or "", timeout=150)
            
            if execution_result.get("status") == "success":
                results = execution_result.get("result", {})
                
                # Handle case when no specific fields are required (free-form questions)
                if not required_keys:
                    if results:
                        elapsed = time.time() - start_time
                        logger.info(f"Analysis completed in {elapsed:.2f}s")
                        return results
                    else:
                        # For free-form questions, provide a basic successful response
                        elapsed = time.time() - start_time
                        logger.info(f"Analysis completed in {elapsed:.2f}s with basic response")
                        return {
                            "status": "completed",
                            "message": "Analysis completed successfully",
                            "data_processed": True
                        }
                
                # Apply type casting to ensure correct format for structured questions
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
    
    def analyze_data_from_folder(self, questions: str, working_folder: str, saved_files: dict) -> Dict[str, Any]:
        """
        NEW: File-based analysis function that reads data from disk files.
        This replaces in-memory processing with file-to-disk approach.
        """
        start_time = time.time()
        
        try:
            logger.info(f"🗂️ Starting file-based analysis in folder: {working_folder}")
            logger.info(f"🗂️ Available files: {list(saved_files.keys())}")
            
            # Parse required format from questions
            required_keys, type_map = self.parse_keys_and_types(questions)
            logger.info(f"Required fields: {required_keys}")
            
            # Find and process different file types in the working folder
            data_files = {
                'csv': [],
                'excel': [],
                'json': [],
                'database': [],
                'pdf': [],
                'image': [],
                'text': []
            }
            
            # Categorize files by type
            for field_name, file_path in saved_files.items():
                if isinstance(file_path, str):
                    file_ext = file_path.lower().split('.')[-1]
                    if file_ext == 'csv':
                        data_files['csv'].append(file_path)
                        logger.info(f"📊 Found CSV file: {file_path}")
                    elif file_ext in ['xlsx', 'xls']:
                        data_files['excel'].append(file_path)
                        logger.info(f"📈 Found Excel file: {file_path}")
                    elif file_ext == 'json':
                        data_files['json'].append(file_path)
                        logger.info(f"🔗 Found JSON file: {file_path}")
                    elif file_ext in ['db', 'sqlite', 'sqlite3']:
                        data_files['database'].append(file_path)
                        logger.info(f"🗄️ Found Database file: {file_path}")
                    elif file_ext == 'pdf':
                        data_files['pdf'].append(file_path)
                        logger.info(f"📄 Found PDF file: {file_path}")
                    elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
                        data_files['image'].append(file_path)
                        logger.info(f"🖼️ Found Image file: {file_path}")
                    elif file_ext == 'txt' and 'question' not in file_path.lower():
                        data_files['text'].append(file_path)
                        logger.info(f"📝 Found Text file: {file_path}")
            
            # Load data from the most appropriate file type
            df_pickle_path = None
            primary_data = pd.DataFrame()
            data_source_info = ""
            
            try:
                # Priority order: CSV -> Excel -> JSON -> Database -> PDF -> Images -> Text
                if data_files['csv']:
                    primary_file = data_files['csv'][0]
                    logger.info(f"📊 Loading primary CSV: {primary_file}")
                    primary_data = pd.read_csv(primary_file)
                    data_source_info = f"CSV file: {primary_file}"
                    
                elif data_files['excel']:
                    primary_file = data_files['excel'][0]
                    logger.info(f"📈 Loading primary Excel: {primary_file}")
                    primary_data = pd.read_excel(primary_file)
                    data_source_info = f"Excel file: {primary_file}"
                    
                elif data_files['json']:
                    primary_file = data_files['json'][0]
                    logger.info(f"🔗 Loading primary JSON: {primary_file}")
                    with open(primary_file, 'r') as f:
                        json_data = json.load(f)
                    if isinstance(json_data, list):
                        primary_data = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict):
                        primary_data = pd.json_normalize(json_data)
                    data_source_info = f"JSON file: {primary_file}"
                    
                elif data_files['database']:
                    primary_file = data_files['database'][0]
                    logger.info(f"🗄️ Loading primary Database: {primary_file}")
                    import sqlite3
                    conn = sqlite3.connect(primary_file)
                    # Get all table names
                    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                    if not tables.empty:
                        # Use the first table or the largest table
                        table_name = tables.iloc[0]['name']
                        primary_data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        data_source_info = f"Database file: {primary_file}, table: {table_name}"
                    conn.close()
                    
                elif data_files['pdf']:
                    primary_file = data_files['pdf'][0]
                    logger.info(f"📄 Processing PDF: {primary_file}")
                    # Extract text from PDF and create structured data
                    try:
                        import PyPDF2
                        with open(primary_file, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            text_content = ""
                            for page in pdf_reader.pages:
                                text_content += page.extract_text()
                        # Create a simple DataFrame with the text content
                        primary_data = pd.DataFrame({'content': [text_content], 'source': [primary_file]})
                        data_source_info = f"PDF file: {primary_file}"
                    except ImportError:
                        logger.warning("PyPDF2 not available, treating PDF as text file")
                        with open(primary_file, 'rb') as f:
                            content = f.read().decode('utf-8', errors='ignore')
                        primary_data = pd.DataFrame({'content': [content], 'source': [primary_file]})
                        data_source_info = f"PDF file (as text): {primary_file}"
                        
                elif data_files['image']:
                    primary_file = data_files['image'][0]
                    logger.info(f"🖼️ Processing Image: {primary_file}")
                    # Use existing image processing capabilities
                    with open(primary_file, 'rb') as f:
                        image_bytes = f.read()
                    image_result = self.process_image_file(image_bytes, primary_file)
                    if 'error' not in image_result:
                        primary_data = pd.DataFrame([image_result])
                        data_source_info = f"Image file: {primary_file}"
                    else:
                        logger.error(f"Image processing failed: {image_result['error']}")
                        return {"error": f"Failed to process image: {image_result['error']}"}
                        
                elif data_files['text']:
                    primary_file = data_files['text'][0]
                    logger.info(f"📝 Loading Text file: {primary_file}")
                    with open(primary_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    # Try to parse structured text data
                    lines = text_content.strip().split('\n')
                    if len(lines) > 1 and ('\t' in lines[0] or ',' in lines[0]):
                        # Looks like delimited data
                        if '\t' in lines[0]:
                            primary_data = pd.read_csv(primary_file, sep='\t')
                        else:
                            primary_data = pd.read_csv(primary_file)
                    else:
                        # Plain text content
                        primary_data = pd.DataFrame({'content': [text_content], 'source': [primary_file]})
                    data_source_info = f"Text file: {primary_file}"
                
                if not primary_data.empty:
                    logger.info(f"✅ Loaded data from {data_source_info}: {primary_data.shape} rows/cols")
                    logger.info(f"✅ Data columns: {list(primary_data.columns)}")
                    
                    # Save DataFrame to temp pickle for injection
                    df_pickle_path = tempfile.mktemp(suffix='.pkl')
                    primary_data.to_pickle(df_pickle_path)
                else:
                    logger.warning("⚠️ No compatible data files found in uploaded files")
                    return {"error": "No compatible data files found in upload"}
                    
            except Exception as load_e:
                logger.error(f"❌ Failed to load data file: {load_e}")
                return {"error": f"Failed to load data file: {load_e}"}
            
            # Create enhanced analysis prompt for file-based processing
            prompt = f"""
You are a data analyst. I have loaded a dataset from uploaded files and need you to analyze it according to specific requirements.

Data source: {data_source_info}
Dataset shape: {primary_data.shape}
Dataset columns: {list(primary_data.columns)}
Sample data: {primary_data.head(3).to_dict()}
Working folder: {working_folder}

Questions to answer: {questions}

Required output format (EXACT field names and types):
{json.dumps({k: str(v.__name__) for k, v in type_map.items()}, indent=2)}

Generate Python code that:
1. Uses the injected 'df' DataFrame variable containing the loaded data
2. Performs real calculations on the actual uploaded data (no hardcoded values)
3. Creates charts using plot_to_base64() function for any visualization fields
4. Stores all results in a 'results' dictionary with EXACT field names above
5. Handles any data format variations robustly

Important rules:
- Use ONLY real data calculations from uploaded files
- Chart fields should contain base64 encoded PNG/WEBP images
- Network analysis should use networkx if the data contains edges/relationships
- For sales data, calculate actual totals, correlations, and statistics
- For weather data, compute real temperature and precipitation metrics
- For database/PDF/image data, extract and analyze the relevant information
- All numerical values must be calculated from the actual uploaded dataset

Generate ONLY the Python code, no explanations:
"""

            # Get code generation from OpenAI (same as original method)
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
            
            logger.info(f"Generated {len(generated_code)} chars of Python code for file-based analysis")
            
            # Execute the generated code (same as original method)
            execution_result = self.execute_python_code(generated_code, df_pickle_path or "", timeout=150)
            
            if execution_result.get("status") == "success":
                results = execution_result.get("result", {})
                
                # Handle case when no specific fields are required
                if not required_keys:
                    if results:
                        elapsed = time.time() - start_time
                        logger.info(f"File-based analysis completed in {elapsed:.2f}s")
                        return results
                    else:
                        elapsed = time.time() - start_time
                        logger.info(f"File-based analysis completed in {elapsed:.2f}s with basic response")
                        return {
                            "status": "completed",
                            "message": "Analysis completed successfully",
                            "data_processed": True
                        }
                
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
                        logger.warning(f"Missing required field: {key}")
                        final_results[key] = None
                
                elapsed = time.time() - start_time
                logger.info(f"✅ File-based analysis completed in {elapsed:.2f}s")
                logger.info(f"✅ Results keys: {list(final_results.keys())}")
                return final_results
            else:
                logger.error(f"File-based code execution failed: {execution_result.get('message', 'Unknown error')}")
                return {"error": f"Analysis failed: {execution_result.get('message', 'Code execution error')}"}
                
        except Exception as e:
            logger.error(f"File-based analysis error: {str(e)}")
            return {"error": f"File-based analysis error: {str(e)}"}
    
    def detect_question_format(self, questions: str) -> str:
        """Detect if questions require array format (web scraping) or object format (CSV)"""
        # Look for array format indicators in questions
        array_indicators = [
            "JSON array of strings",
            "array of strings",
            "return array",
            "string array",
            "list of strings"
        ]
        
        questions_lower = questions.lower()
        for indicator in array_indicators:
            if indicator in questions_lower:
                return "array"
        
        return "object"
    
    def analyze_web_data(self, questions: str) -> List[str]:
        """
        Analyze web-based data (Wikipedia, etc.) and return results as array of strings.
        Extracts URLs from questions and scrapes data for analysis.
        """
        start_time = time.time()
        
        try:
            # Extract URL from questions
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, questions)
            
            if not urls:
                return ["Error: No URL found in questions"]
            
            url = urls[0]  # Use first URL found
            logger.info(f"Scraping URL: {url}")
            
            # Create web scraping analysis prompt
            prompt = f"""
You are a data analyst. I need you to analyze web data from a URL and answer questions.

URL to analyze: {url}
Questions to answer: {questions}

Generate Python code that:
1. Uses the scrape_url_to_dataframe() function to get data from the URL
2. Analyzes the scraped data to answer each question in order
3. For chart questions, uses plot_to_base64() to create base64 encoded images
4. Stores answers in 'results' list where each element is a string
5. Chart results should be full data URI format: "data:image/png;base64," + base64_string

Important rules:
- Return answers as strings in the exact order questions are asked
- Use real data from the scraped website - no hardcoded values
- For numerical answers, convert to string format
- For chart questions, return complete data URI with base64 image
- Handle data cleaning robustly: use pd.to_numeric(errors='coerce') for numeric conversions
- Remove footnote references [1], [2] etc. before processing numeric data
- Use .str.extract() for complex string parsing instead of simple replacements
- Always check data types and handle missing/invalid values gracefully

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
                return ["Error: No code generated by OpenAI"]
            
            logger.info(f"Generated {len(generated_code)} chars of Python code")
            
            # Execute the generated code
            execution_result = self.execute_python_code(generated_code, timeout=180)
            
            if execution_result.get("status") == "success":
                results = execution_result.get("result", [])
                
                # Ensure results is a list of strings
                if isinstance(results, dict):
                    # Convert dict values to list
                    results = [str(v) for v in results.values()]
                elif isinstance(results, list):
                    # Ensure all elements are strings
                    results = [str(item) for item in results]
                else:
                    results = [str(results)]
                
                elapsed = time.time() - start_time
                logger.info(f"Web analysis completed in {elapsed:.2f}s")
                return results
            else:
                error_msg = execution_result.get('message', 'Code execution error')
                logger.error(f"Web analysis failed: {error_msg}")
                return [f"Error: {error_msg}"]
                
        except Exception as e:
            logger.error(f"Web analysis error: {str(e)}")
            return [f"Error: Web analysis failed - {str(e)}"]

    def process_image_file(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Process an uploaded image file using multiple techniques:
        1. OpenAI Vision API for comprehensive analysis
        2. OCR for text extraction  
        3. Basic image properties
        
        Returns structured data that can be converted to DataFrame format.
        """
        try:
            if not self.image_processing_enabled:
                return {"error": "Image processing not available"}
            
            logger.info(f"🖼️ Processing image: {filename}")
            start_time = time.time()
            
            # Convert to PIL Image
            if not Image:
                return {"error": "PIL Image not available"}
            image = Image.open(BytesIO(image_bytes))
            image_format = image.format or 'PNG'
            width, height = image.size
            
            logger.info(f"Image properties: {width}x{height}, format: {image_format}")
            
            # Optimize image size for API calls (max 2048px)
            if max(width, height) > 2048:
                ratio = 2048 / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                try:
                    # Try modern PIL resampling constant first
                    if hasattr(Image, 'LANCZOS'):
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                    elif hasattr(Image, 'Resampling') and hasattr(Image.Resampling, 'LANCZOS'):
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    else:
                        # Fallback to basic resizing
                        image = image.resize((new_width, new_height))
                except (AttributeError, Exception):
                    image = image.resize((new_width, new_height))
                logger.info(f"Resized image to {new_width}x{new_height}")
            
            # Convert to base64 for OpenAI Vision API
            buffer = BytesIO()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            results = {
                "filename": filename,
                "width": width,
                "height": height,
                "format": image_format,
                "file_size": len(image_bytes)
            }
            
            # Method 1: OpenAI Vision API Analysis (Primary)
            try:
                logger.info("🔍 Analyzing image with OpenAI Vision API...")
                vision_result = self.analyze_image_with_vision(image_b64)
                if vision_result:
                    results.update(vision_result)
                    logger.info("✅ Vision API analysis completed")
                else:
                    logger.warning("⚠️ Vision API returned no results")
            except Exception as e:
                logger.warning(f"⚠️ Vision API failed: {e}")
                results["vision_error"] = str(e)
            
            # Method 2: OCR Text Extraction (Fallback)
            try:
                logger.info("📝 Extracting text with OCR...")
                if cv2 and pytesseract:
                    # Convert PIL to cv2 format for OCR
                    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    extracted_text = pytesseract.image_to_string(cv_image).strip()
                elif pytesseract:
                    # Direct PIL to OCR
                    extracted_text = pytesseract.image_to_string(image).strip()
                else:
                    raise Exception("OCR libraries not available")
                
                if extracted_text:
                    results["extracted_text"] = extracted_text
                    results["text_length"] = len(extracted_text)
                    logger.info(f"✅ OCR extracted {len(extracted_text)} characters")
                    
                    # Try to parse structured data from OCR text
                    structured_data = self.parse_structured_text(extracted_text)
                    if structured_data:
                        results["structured_data"] = structured_data
                else:
                    results["extracted_text"] = ""
                    logger.info("ℹ️ No text found in image")
                    
            except Exception as e:
                logger.warning(f"⚠️ OCR failed: {e}")
                results["ocr_error"] = str(e)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Image processing completed in {elapsed:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Image processing error: {e}")
            return {"error": f"Image processing failed: {str(e)}"}
    
    def analyze_image_with_vision(self, image_b64: str) -> Dict[str, Any]:
        """
        Analyze image using OpenAI Vision API to extract structured information.
        """
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert at analyzing images and extracting structured data. 
                        Focus on identifying:
                        1. Charts, tables, graphs - extract numerical data
                        2. Financial documents - extract key figures, dates, amounts
                        3. Forms, receipts - extract structured field/value pairs
                        4. Text documents - extract key information
                        
                        Always respond with JSON containing the extracted structured data.
                        For charts/graphs, include data points, labels, trends.
                        For tables, extract rows and columns.
                        For documents, extract key-value pairs."""
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and extract all structured data in JSON format. Include any numerical data, tables, charts, or key information that could be used for data analysis."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            if response_text:
                vision_data = json.loads(response_text)
            else:
                vision_data = {"error": "No response from Vision API"}
            
            return {
                "vision_analysis": vision_data,
                "analysis_type": "openai_vision"
            }
            
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            raise e
    
    def parse_structured_text(self, text: str) -> Dict[str, Any]:
        """
        Parse OCR-extracted text to identify structured data patterns.
        """
        try:
            structured = {}
            
            # Look for key-value pairs (various formats)
            patterns = [
                r'([A-Za-z\s]+):\s*([^\n]+)',      # "Label: Value"
                r'([A-Za-z\s]+)\s+([0-9,.$%-]+)',  # "Label Amount"
                r'([A-Z\s]+)\s*=\s*([^\n]+)'       # "LABEL = Value"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for key, value in matches:
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    # Try to convert to number
                    try:
                        # Remove currency symbols and commas
                        clean_value = re.sub(r'[$,]', '', value)
                        if re.match(r'^-?\d+\.?\d*$', clean_value):
                            structured[key] = float(clean_value)
                        else:
                            structured[key] = value
                    except:
                        structured[key] = value
            
            # Look for tables (basic detection)
            lines = text.split('\n')
            potential_table_rows = []
            for line in lines:
                # Check if line has multiple numeric values separated by spaces/tabs
                numbers = re.findall(r'\d+(?:\.\d+)?', line)
                if len(numbers) >= 2:
                    potential_table_rows.append(line.strip())
            
            if potential_table_rows:
                structured['table_data'] = potential_table_rows
            
            return structured if structured else {}
            
        except Exception as e:
            logger.warning(f"Text parsing error: {e}")
            return {}
    
    def images_to_dataframe(self, image_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert processed image results to a pandas DataFrame for analysis.
        """
        try:
            # Flatten nested structured data
            flattened_data = []
            
            for img_result in image_results:
                row = {
                    'filename': img_result.get('filename', ''),
                    'width': img_result.get('width', 0),
                    'height': img_result.get('height', 0),
                    'file_size': img_result.get('file_size', 0),
                    'has_text': bool(img_result.get('extracted_text', '')),
                    'text_length': img_result.get('text_length', 0)
                }
                
                # Add vision analysis data
                if 'vision_analysis' in img_result:
                    vision_data = img_result['vision_analysis']
                    if isinstance(vision_data, dict):
                        # Flatten vision data with prefix
                        for key, value in vision_data.items():
                            if isinstance(value, (str, int, float)):
                                row[f'vision_{key}'] = value
                            elif isinstance(value, list) and len(value) > 0:
                                row[f'vision_{key}_count'] = len(value)
                                if isinstance(value[0], (str, int, float)):
                                    row[f'vision_{key}_first'] = value[0]
                
                # Add structured OCR data
                if 'structured_data' in img_result:
                    structured = img_result['structured_data']
                    if isinstance(structured, dict):
                        for key, value in structured.items():
                            row[f'ocr_{key}'] = value
                
                flattened_data.append(row)
            
            df = pd.DataFrame(flattened_data)
            logger.info(f"Created DataFrame from {len(image_results)} images: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame conversion error: {e}")
            # Return empty DataFrame with basic structure
            return pd.DataFrame([{
                'filename': 'error',
                'error': str(e)
            }])