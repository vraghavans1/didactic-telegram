"""
Enhanced Data Parser - Intelligent type casting and data processing
Inspired by colleague's approach but optimized for our OpenAI-based system
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

class EnhancedDataParser:
    """Advanced data parsing with intelligent type detection and casting."""
    
    def __init__(self):
        self.type_map_definitions = {
            "number": float,
            "string": str,
            "integer": int,
            "int": int,
            "float": float,
            "bool": bool,
            "boolean": bool
        }
    
    def parse_keys_and_types(self, raw_questions: str) -> Tuple[List[str], Dict[str, Callable]]:
        """
        Parse key/type requirements from questions text.
        Extracts patterns like: - `key_name`: type_name
        
        Returns:
            keys_list: list of keys in order
            type_map: dict key -> casting function
        """
        try:
            # Enhanced pattern to catch more variations
            patterns = [
                r"-\s*`([^`]+)`\s*:\s*(\w+)",  # - `key`: type
                r"\-\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\w+)",  # - key: type
                r"`([^`]+)`\s*:\s*(\w+)",  # `key`: type
            ]
            
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, raw_questions, re.IGNORECASE)
                matches.extend(found)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_matches = []
            for key, type_name in matches:
                if key not in seen:
                    unique_matches.append((key, type_name))
                    seen.add(key)
            
            # Build type map
            type_map = {}
            keys_list = []
            
            for key, type_name in unique_matches:
                keys_list.append(key)
                type_func = self.type_map_definitions.get(type_name.lower(), str)
                type_map[key] = type_func
            
            logger.info(f"Parsed {len(keys_list)} keys with types: {list(type_map.keys())}")
            return keys_list, type_map
            
        except Exception as e:
            logger.error(f"Error parsing keys and types: {e}")
            return [], {}
    
    def smart_cast_value(self, value: Any, target_type: Callable) -> Any:
        """
        Intelligently cast a value to target type with error handling.
        """
        try:
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return None
                
            # Handle pandas/numpy types
            if hasattr(value, 'item'):
                value = value.item()
            
            # String to boolean conversion
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', 'yes', '1', 'on')
                return bool(value)
            
            # Numeric conversions
            if target_type in (int, float):
                if isinstance(value, str):
                    # Remove common formatting
                    cleaned = re.sub(r'[,$%]', '', value.strip())
                    if cleaned.lower() in ('', 'nan', 'null', 'none'):
                        return None
                    value = cleaned
                
                result = target_type(float(value))
                
                # For int conversion, ensure it's actually an integer
                if target_type == int and isinstance(result, float):
                    result = int(result)
                    
                return result
            
            # String conversion
            if target_type == str:
                return str(value)
            
            # Default casting
            return target_type(value)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast {value} to {target_type.__name__}: {e}")
            # Return sensible defaults
            if target_type == str:
                return str(value)
            elif target_type in (int, float):
                return 0
            elif target_type == bool:
                return False
            else:
                return None
    
    def process_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize DataFrame columns for better processing.
        """
        try:
            # Clean column names
            df.columns = df.columns.astype(str)
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace(r'\[.*\]', '', regex=True)
            df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)
            df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
            
            # Ensure unique column names
            cols = df.columns.tolist()
            seen = {}
            for i, col in enumerate(cols):
                if col in seen:
                    seen[col] += 1
                    cols[i] = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
            
            df.columns = cols
            
            # Smart type inference for common patterns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        # Check if it looks like numbers with formatting
                        sample = df[col].dropna().astype(str).head(10)
                        if sample.str.match(r'^[\d,.%$-]+$').all():
                            # Clean and convert
                            cleaned = df[col].astype(str).str.replace(r'[,$%]', '', regex=True)
                            df[col] = pd.to_numeric(cleaned, errors='ignore')
                    except Exception:
                        pass
                    
                    # Try to convert to datetime
                    try:
                        if df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
                            df[col] = pd.to_datetime(df[col], errors='ignore')
                    except Exception:
                        pass
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing DataFrame columns: {e}")
            return df
    
    def extract_format_requirements(self, question: str) -> Dict[str, str]:
        """
        Extract format requirements from question text.
        """
        format_hints = {}
        
        # Look for JSON format requirements
        if re.search(r'json\s+object', question.lower()):
            format_hints['format'] = 'json_object'
        elif re.search(r'json\s+array', question.lower()):
            format_hints['format'] = 'json_array'
        elif re.search(r'single\s+word', question.lower()):
            format_hints['format'] = 'single_word'
        elif re.search(r'list\s+of', question.lower()):
            format_hints['format'] = 'list'
        
        # Look for chart requirements
        chart_types = ['bar_chart', 'line_chart', 'histogram', 'network_graph', 'degree_histogram']
        for chart_type in chart_types:
            if chart_type.replace('_', ' ') in question.lower() or chart_type in question.lower():
                format_hints['requires_chart'] = chart_type
                break
        
        return format_hints
    
    def validate_result_format(self, result: Dict[str, Any], keys_list: List[str], type_map: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Validate and cast result values according to expected types.
        """
        try:
            validated_result = {}
            
            for key in keys_list:
                if key in result:
                    target_type = type_map.get(key, str)
                    validated_result[key] = self.smart_cast_value(result[key], target_type)
                else:
                    # Provide default value based on expected type
                    target_type = type_map.get(key, str)
                    if target_type == str:
                        validated_result[key] = ""
                    elif target_type in (int, float):
                        validated_result[key] = 0
                    elif target_type == bool:
                        validated_result[key] = False
                    else:
                        validated_result[key] = None
            
            # Include any additional keys from result
            for key, value in result.items():
                if key not in validated_result:
                    validated_result[key] = value
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Error validating result format: {e}")
            return result