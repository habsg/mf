#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template, jsonify, session
import secrets
import json
import logging # Import logging
import traceback # For detailed error logging

# --- Flask App Setup ---
# Corrected: Point to the nested template/static folders and use single quotes
app = Flask(__name__,
            template_folder=\"../templates/templates\",
            static_folder=\"../static/static\")
app.secret_key = secrets.token_hex(16) # For session management

# Set logging level to DEBUG (can be removed later)
app.logger.setLevel(logging.DEBUG)

# Define upload folder (relative to main.py location)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), \"uploads\")
app.config[\"UPLOAD_FOLDER\"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Parameter Definitions (Corrected: Use standard quotes) ---
PARAMETER_SETS = {
    \"Common\": [
        \"AUM(in Rs. cr)\", \"ExpenseRatio (%)\", \"Return (%)1 yr\", \"Return (%)3 yrs\",
        \"Return (%)5 yrs\", \"NAV\", \"RupeeVestRating\"
    ],
    \"EQ\": [
        \"Alpha\", \"Beta\", \"Standard Deviation\", \"Sharpe\", \"Sortino\",
        \"Turnover Ratio (%)\", \"Large Cap(%)\", \"Mid Cap(%)\", \"Small Cap(%)\",
        \"Avg. Market Cap(in Rs. cr)\", \"No. ofStocks\", \"Return (%)10 yrs\"
    ],
    \"DT\": [
        \"Avg. Maturity(in yrs)\", \"Mod. Duration(in yrs)\", \"Yield To Maturity (%)\",
        \"Standard Deviation\", \"Sharpe\"
    ],
    \"HY\": [
        \"Alpha\", \"Beta\", \"Standard Deviation\", \"Sharpe\", \"Sortino\",
        \"Avg. Maturity(in yrs)\", \"Mod. Duration(in yrs)\", \"Yield To Maturity (%)\",
        \"Large Cap(%)\", \"Mid Cap(%)\", \"Small Cap(%)\"
    ],
    \"Other\": [
        \"Standard Deviation\", \"Sharpe\"
    ]
}

HIGHER_IS_BETTER = [
    \"AUM(in Rs. cr)\", \"Return (%)1 yr\", \"Return (%)3 yrs\", \"Return (%)5 yrs\", \"Return (%)10 yrs\",
    \"NAV\", \"RupeeVestRating\", \"Alpha\", \"Sharpe\", \"Sortino\", \"Yield To Maturity (%)\",
    \"Large Cap(%)\", \"Mid Cap(%)\", \"Small Cap(%)\", \"Avg. Market Cap(in Rs. cr)\", \"No. ofStocks\"
]

LOWER_IS_BETTER = [
    \"ExpenseRatio (%)\", \"Beta\", \"Standard Deviation\", \"Turnover Ratio (%)\",
    \"Avg. Maturity(in yrs)\", \"Mod. Duration(in yrs)\"
]

# --- Helper Functions ---
def clean_numeric_column(series):
    # Convert to string, remove %, commas, handle NaNs/empty strings
    series_str = series.astype(str).str.replace(\"%
