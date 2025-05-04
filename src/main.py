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
            template_folder="../templates",
            static_folder="../static")
app.secret_key = secrets.token_hex(16) # For session management

# Set logging level to DEBUG (can be removed later)
app.logger.setLevel(logging.DEBUG)

# Define upload folder (relative to main.py location)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Parameter Definitions (Corrected: Use standard quotes) ---
PARAMETER_SETS = {
    "Common": [
        "AUM(in Rs. cr)", "ExpenseRatio (%)", "Return (%)1 yr", "Return (%)3 yrs",
        "Return (%)5 yrs", "NAV", "RupeeVestRating"
    ],
    "EQ": [
        "Alpha", "Beta", "Standard Deviation", "Sharpe", "Sortino",
        "Turnover Ratio (%)", "Large Cap(%)", "Mid Cap(%)", "Small Cap(%)",
        "Avg. Market Cap(in Rs. cr)", "No. ofStocks", "Return (%)10 yrs"
    ],
    "DT": [
        "Avg. Maturity(in yrs)", "Mod. Duration(in yrs)", "Yield To Maturity (%)",
        "Standard Deviation", "Sharpe"
    ],
    "HY": [
        "Alpha", "Beta", "Standard Deviation", "Sharpe", "Sortino",
        "Avg. Maturity(in yrs)", "Mod. Duration(in yrs)", "Yield To Maturity (%)",
        "Large Cap(%)", "Mid Cap(%)", "Small Cap(%)"
    ],
    "Other": [
        "Standard Deviation", "Sharpe"
    ]
}

HIGHER_IS_BETTER = [
    "AUM(in Rs. cr)", "Return (%)1 yr", "Return (%)3 yrs", "Return (%)5 yrs", "Return (%)10 yrs",
    "NAV", "RupeeVestRating", "Alpha", "Sharpe", "Sortino", "Yield To Maturity (%)",
    "Large Cap(%)", "Mid Cap(%)", "Small Cap(%)", "Avg. Market Cap(in Rs. cr)", "No. ofStocks"
]

LOWER_IS_BETTER = [
    "ExpenseRatio (%)", "Beta", "Standard Deviation", "Turnover Ratio (%)",
    "Avg. Maturity(in yrs)", "Mod. Duration(in yrs)"
]

# --- Helper Functions ---
def clean_numeric_column(series):
    # Convert to string, remove %, commas, handle NaNs/empty strings
    series_str = series.astype(str).str.replace("%", "", regex=False)\
                                  .str.replace(",", "", regex=False)\
                                  .str.strip()
    # Handle potential non-numeric placeholders like '-' or empty strings
    series_numeric = pd.to_numeric(series_str, errors='coerce')
    return series_numeric

def rank_funds(df, identifier_col, params_weights, params_direction):
    df_ranked = df.copy()
    df_ranked["Composite Score"] = 0.0
    total_weight = sum(params_weights.values())

    if total_weight <= 0:
        app.logger.warning("Total weight is zero or negative. Cannot calculate scores.")
        df_ranked["Rank"] = 1
        return df_ranked, {}

    normalized_params_cols = {}
    # Normalize selected parameter columns
    for param, weight in params_weights.items():
        if param not in df_ranked.columns or weight == 0:
            continue

        # Clean the column first
        df_ranked[param] = clean_numeric_column(df_ranked[param])

        # Check if column is numeric after cleaning
        if not pd.api.types.is_numeric_dtype(df_ranked[param]):
            app.logger.warning(f"Parameter '{param}' is not numeric after cleaning and cannot be used for ranking.")
            continue

        # Handle missing values after cleaning (impute with median or mean? Using 0 for now)
        if df_ranked[param].isnull().any():
             median_val = df_ranked[param].median()
             df_ranked[param].fillna(median_val if pd.notna(median_val) else 0, inplace=True)
             app.logger.info(f"Filled missing values in '{param}' with median ({median_val}) or 0.")

        min_val = df_ranked[param].min()
        max_val = df_ranked[param].max()
        direction = params_direction.get(param, "higher") # Default to higher is better
        norm_col_name = f"{param}_norm"
        normalized_params_cols[param] = norm_col_name

        if max_val == min_val:
            # Avoid division by zero if all values are the same
            df_ranked[norm_col_name] = 0.5 # Assign a neutral score
        else:
            if direction == "higher":
                df_ranked[norm_col_name] = (df_ranked[param] - min_val) / (max_val - min_val)
            elif direction == "lower":
                df_ranked[norm_col_name] = (max_val - df_ranked[param]) / (max_val - min_val)
            else: # Neutral (should not happen with current setup but good practice)
                 df_ranked[norm_col_name] = 0.5

        # Calculate weighted score contribution
        normalized_weight = weight / total_weight
        # Ensure the normalized column exists before calculation
        if norm_col_name in df_ranked.columns:
             df_ranked["Composite Score"] += df_ranked[norm_col_name] * normalized_weight
        else:
             app.logger.error(f"Normalized column '{norm_col_name}' not found after creation attempt.")

    # Rank based on Composite Score (higher is better)
    df_ranked["Rank"] = df_ranked["Composite Score"].rank(ascending=False, method='min').astype(int)
    df_ranked = df_ranked.sort_values(by="Rank")

    return df_ranked, normalized_params_cols

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    app.logger.debug("Accessed index route.")
    # Clear previous session data on new visit
    session.pop('uploaded_filepath', None)
    session.pop('analysis_results', None)
    return render_template('index.html', parameter_sets=PARAMETER_SETS)

@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.debug("Accessed upload route.")
    if 'file' not in request.files:
        app.logger.error("No file part in request.")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Secure filename and save
            # filename = secure_filename(file.filename) # Consider using secure_filename
            filename = file.filename # Using original for now
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f"File '{filename}' uploaded successfully to '{filepath}'.")

            # Store filepath in session
            session['uploaded_filepath'] = filepath

            # Load data to get columns
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                 return jsonify({"error": "Unsupported file type"}), 400

            # Identify potential identifier columns (case-insensitive)
            possible_identifiers = ["Name", "Fund Name", "Scheme Name", "Stock", "Company"]
            identifier_col = None
            df_cols_lower = {col.lower(): col for col in df.columns}
            for p_id in possible_identifiers:
                if p_id.lower() in df_cols_lower:
                    identifier_col = df_cols_lower[p_id.lower()]
                    break

            if not identifier_col:
                # Fallback: use the first column if no common name found
                identifier_col = df.columns[0]
                app.logger.warning(f"Could not find a standard identifier column, using first column: '{identifier_col}'")

            # Store identifier and columns in session
            session['identifier_col'] = identifier_col
            session['data_columns'] = df.columns.tolist()

            app.logger.debug(f"Identifier: {identifier_col}, Columns: {session['data_columns']}")

            return jsonify({
                "message": "File uploaded successfully",
                "columns": session['data_columns'],
                "identifier_col": identifier_col,
                "parameter_sets": PARAMETER_SETS # Send parameter sets for dynamic UI
            })
        except Exception as e:
            app.logger.error(f"Error during file upload or initial processing: {e}")
            app.logger.error(traceback.format_exc()) # Log detailed traceback
            return jsonify({"error": f"An error occurred: {e}"}), 500

    return jsonify({"error": "File upload failed"}), 400

@app.route('/analyze', methods=['POST'])
def analyze_data():
    app.logger.debug("Accessed analyze route.")
    if 'uploaded_filepath' not in session:
        app.logger.error("No file uploaded or session expired.")
        return jsonify({"error": "No file uploaded or session expired. Please upload again."}), 400

    filepath = session['uploaded_filepath']
    identifier_col = session.get('identifier_col') # Get from session
    data_columns = session.get('data_columns', [])

    if not identifier_col or not data_columns:
        app.logger.error("Identifier column or data columns not found in session.")
        return jsonify({"error": "Session data missing. Please upload again."}), 400

    try:
        # Get parameters and weights from the form
        params_weights = {}
        params_direction = {}
        form_data = request.get_json()
        selected_params = form_data.get('parameters', []) # List of selected param names
        weights_data = form_data.get('weights', {}) # Dict of param_name: weight

        app.logger.debug(f"Received for analysis - Params: {selected_params}, Weights: {weights_data}")

        # Validate and populate params_weights and params_direction
        for param in selected_params:
            if param not in data_columns:
                app.logger.warning(f"Selected parameter '{param}' not found in uploaded file columns. Skipping.")
                continue
            try:
                weight = int(weights_data.get(param, 0)) # Default weight 0 if not provided
                if weight < 0: weight = 0 # Ensure non-negative weights
                params_weights[param] = weight

                # Determine direction (higher/lower is better)
                if param in HIGHER_IS_BETTER:
                    params_direction[param] = "higher"
                elif param in LOWER_IS_BETTER:
                    params_direction[param] = "lower"
                else:
                    # Attempt to infer, default to higher if unsure
                    name_lower = param.lower()
                    if any(keyword in name_lower for keyword in ["debt", "expense", "beta", "deviation", "maturity", "duration"]):
                         params_direction[param] = "lower"
                    else:
                         params_direction[param] = "higher"
                    app.logger.info(f"Inferred direction for '{param}' as '{params_direction[param]}'")

            except ValueError:
                app.logger.warning(f"Invalid weight for parameter '{param}'. Using 0.")
                params_weights[param] = 0

        app.logger.debug(f"Processed weights: {params_weights}")
        app.logger.debug(f"Processed directions: {params_direction}")

        # Load the data again
        filename = os.path.basename(filepath)
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Ensure identifier column exists
        if identifier_col not in df.columns:
             app.logger.error(f"Identifier column '{identifier_col}' not found in the dataframe.")
             return jsonify({"error": f"Identifier column '{identifier_col}' missing. Please check the file or re-upload."}), 400

        # Ensure identifier has no NaNs
        if df[identifier_col].isnull().any():
            app.logger.warning(f"Dropping rows with missing identifier in column '{identifier_col}'.")
            df.dropna(subset=[identifier_col], inplace=True)

        if df.empty:
            app.logger.error("DataFrame is empty after handling missing identifiers.")
            return jsonify({"error": "No data remaining after handling missing identifiers."}), 400

        # --- Perform Ranking --- #
        df_ranked, normalized_cols = rank_funds(df, identifier_col, params_weights, params_direction)

        # Select columns to display (Original params, Score, Rank)
        cols_to_display = [identifier_col] + list(params_weights.keys()) + ["Composite Score", "Rank"]
        # Ensure only existing columns are selected
        cols_to_display = [col for col in cols_to_display if col in df_ranked.columns]

        results_df = df_ranked[cols_to_display]

        # Convert results to HTML table
        results_html = results_df.to_html(classes='table table-striped table-hover', index=False, border=0, float_format='%.4f')

        # Store results in session (optional, consider size limits)
        # session['analysis_results'] = results_df.to_dict('records')

        app.logger.info("Analysis complete. Sending results.")
        return jsonify({"results_html": results_html})

    except FileNotFoundError:
        app.logger.error(f"Uploaded file not found at path: {filepath}")
        session.pop('uploaded_filepath', None) # Clear invalid path from session
        return jsonify({"error": "Uploaded file not found. Please upload again."}), 404
    except KeyError as e:
        app.logger.error(f"Missing key during analysis: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Configuration error: Missing data key '{e}'. Please check parameters."}), 400
    except Exception as e:
        app.logger.error(f"Error during analysis: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"An unexpected error occurred during analysis: {e}"}), 500

# --- Main Execution --- #
if __name__ == '__main__':
    # Use Gunicorn's host/port in production (set via env vars or command line)
    # For local dev, Flask's default is 127.0.0.1:5000
    # To make accessible on local network during dev:
    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=True) # Standard local development run

