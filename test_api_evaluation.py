#!/usr/bin/env python3
"""
API Testing Script - Simulates IIT Madras Evaluation Format
Creates multipart form requests with question.txt + data files
"""
import requests
import json
import pandas as pd
import sqlite3
import os
from io import BytesIO

API_URL = "http://localhost:5000/api/"

def create_sample_data():
    """Create various sample data files for testing"""
    
    # Sample Sales Data (CSV)
    sales_data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Region': ['North', 'South', 'East', 'West', 'Central'],
        'Sales': [120, 180, 150, 200, 160],
        'Tax': [12, 18, 15, 20, 16],
        'Day': [1, 2, 3, 4, 5]
    }
    pd.DataFrame(sales_data).to_csv('sample_sales.csv', index=False)
    
    # Sample Network Data (CSV)
    network_data = {
        'source': ['A', 'B', 'C', 'A', 'B', 'C', 'D'],
        'target': ['B', 'C', 'D', 'C', 'D', 'A', 'A']
    }
    pd.DataFrame(network_data).to_csv('sample_edges.csv', index=False)
    
    # Sample Weather Data (Excel)
    weather_data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Temperature': [25.5, 27.2, 23.8, 26.1],
        'Humidity': [65, 70, 58, 62],
        'Precipitation': [0, 2.5, 0, 1.2]
    }
    pd.DataFrame(weather_data).to_excel('sample_weather.xlsx', index=False)
    
    # Sample Product Data (JSON)
    product_data = [
        {"id": 1, "name": "Product A", "price": 100, "category": "Electronics"},
        {"id": 2, "name": "Product B", "price": 150, "category": "Clothing"},
        {"id": 3, "name": "Product C", "price": 80, "category": "Electronics"},
        {"id": 4, "name": "Product D", "price": 200, "category": "Home"}
    ]
    with open('sample_products.json', 'w') as f:
        json.dump(product_data, f, indent=2)
    
    # Sample Employee Database (SQLite)
    conn = sqlite3.connect('sample_employees.db')
    employee_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'department': ['IT', 'HR', 'Finance', 'IT', 'Marketing'],
        'salary': [75000, 65000, 70000, 80000, 60000],
        'years_experience': [5, 3, 7, 6, 2]
    }
    pd.DataFrame(employee_data).to_sql('employees', conn, if_exists='replace', index=False)
    conn.close()
    
    print("✅ Sample data files created successfully")

def test_sales_analysis():
    """Test 1: Sales Data Analysis (like evaluation)"""
    
    question = """Analyze `sample_sales.csv`.

Return a JSON object with keys:
- `total_sales`: number
- `top_region`: string  
- `day_sales_correlation`: number
- `bar_chart`: base64 PNG string under 100kB
- `median_sales`: number
- `total_tax`: number
- `cumulative_sales_chart`: base64 PNG string under 100kB

Answer:
1. What is the total sales across all regions?
2. Which region has the highest total sales?
3. What is the correlation between day of month and sales?
4. Create a bar chart of total sales by region with blue bars
5. What is the median sales value?
6. What is the total tax collected?
7. Create a cumulative sales chart over time with red line"""

    # Create multipart form request
    files = {
        'questions': ('questions.txt', question, 'text/plain'),
        'data_file': ('sample_sales.csv', open('sample_sales.csv', 'rb'), 'text/csv')
    }
    
    print("\n🧪 Testing Sales Analysis...")
    print("📤 Sending multipart request...")
    
    try:
        response = requests.post(API_URL, files=files, timeout=60)
        files['data_file'][1].close()  # Close file handle
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response keys:", list(result.keys()))
            
            # Validate response structure
            expected_keys = ['total_sales', 'top_region', 'day_sales_correlation', 
                           'bar_chart', 'median_sales', 'total_tax', 'cumulative_sales_chart']
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            else:
                print("✅ All required keys present")
                
            # Show some results
            for key, value in result.items():
                if key.endswith('_chart'):
                    print(f"  {key}: base64 image ({len(value)} chars)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_network_analysis():
    """Test 2: Network Analysis"""
    
    question = """Analyze the network data in `sample_edges.csv`.

Return a JSON object with keys:
- `edge_count`: number
- `node_count`: number
- `density`: number
- `average_degree`: number
- `network_plot`: base64 PNG string under 100kB

Calculate:
1. Total number of edges
2. Total number of unique nodes
3. Network density
4. Average degree of nodes
5. Create a network visualization"""

    files = {
        'questions': ('questions.txt', question, 'text/plain'),
        'edges_file': ('sample_edges.csv', open('sample_edges.csv', 'rb'), 'text/csv')
    }
    
    print("\n🧪 Testing Network Analysis...")
    
    try:
        response = requests.post(API_URL, files=files, timeout=60)
        files['edges_file'][1].close()
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response:", result)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_excel_analysis():
    """Test 3: Excel File Analysis"""
    
    question = """Analyze the weather data in `sample_weather.xlsx`.

Return a JSON object with keys:
- `avg_temperature`: number
- `max_humidity`: number
- `total_precipitation`: number
- `weather_trend_chart`: base64 PNG string

Calculate:
1. Average temperature
2. Maximum humidity
3. Total precipitation
4. Create a weather trend chart"""

    files = {
        'questions': ('questions.txt', question, 'text/plain'),
        'weather_file': ('sample_weather.xlsx', open('sample_weather.xlsx', 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    }
    
    print("\n🧪 Testing Excel Analysis...")
    
    try:
        response = requests.post(API_URL, files=files, timeout=60)
        files['weather_file'][1].close()
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response:", result)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_database_analysis():
    """Test 4: SQLite Database Analysis"""
    
    question = """Analyze the employee data in `sample_employees.db`.

Return a JSON object with keys:
- `total_employees`: number
- `avg_salary`: number
- `top_department`: string
- `salary_distribution_chart`: base64 PNG string

Calculate:
1. Total number of employees
2. Average salary
3. Department with most employees
4. Create salary distribution chart"""

    files = {
        'questions': ('questions.txt', question, 'text/plain'),
        'db_file': ('sample_employees.db', open('sample_employees.db', 'rb'), 'application/x-sqlite3')
    }
    
    print("\n🧪 Testing Database Analysis...")
    
    try:
        response = requests.post(API_URL, files=files, timeout=60)
        files['db_file'][1].close()
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response:", result)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_json_analysis():
    """Test 5: JSON File Analysis"""
    
    question = """Analyze the product data in `sample_products.json`.

Return a JSON object with keys:
- `total_products`: number
- `avg_price`: number
- `top_category`: string
- `price_chart`: base64 PNG string

Calculate:
1. Total number of products
2. Average price
3. Category with most products
4. Create price distribution chart"""

    files = {
        'questions': ('questions.txt', question, 'text/plain'),
        'json_file': ('sample_products.json', open('sample_products.json', 'rb'), 'application/json')
    }
    
    print("\n🧪 Testing JSON Analysis...")
    
    try:
        response = requests.post(API_URL, files=files, timeout=60)
        files['json_file'][1].close()
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success! Response:", result)
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def cleanup_files():
    """Remove test files"""
    files_to_remove = [
        'sample_sales.csv', 'sample_edges.csv', 'sample_weather.xlsx',
        'sample_products.json', 'sample_employees.db'
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print("\n🧹 Test files cleaned up")

def main():
    """Run all tests"""
    print("🚀 Starting API Testing - IIT Madras Evaluation Format")
    print("=" * 60)
    
    # Create sample data
    create_sample_data()
    
    # Run tests
    test_sales_analysis()      # Primary test (matches evaluation)
    test_network_analysis()    # Network data test
    test_excel_analysis()      # Excel file test
    test_database_analysis()   # SQLite database test
    test_json_analysis()       # JSON file test
    
    # Cleanup
    cleanup_files()
    
    print("\n" + "=" * 60)
    print("🎯 API Testing Complete!")

if __name__ == "__main__":
    main()