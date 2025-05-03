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
# Corrected: Use standard single quotes for paths
app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = secrets.token_hex(16) # For session management

# Set logging level to DEBUG
app.logger.setLevel(logging.DEBUG)

# Define upload folder (relative to main.py location)
# Corrected: Use standard single quotes
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Parameter Definitions (Corrected: Use standard quotes) ---
PARAMETER_SETS = {
    'Common': [
        'AUM(in Rs. cr)', 'ExpenseRatio (%)', 'Return (%)1 yr', 'Return (%)3 yrs',
        'Return (%)5 yrs', 'NAV', 'RupeeVestRating'
    ],
    'EQ': [
        'Alpha', 'Beta', 'Standard Deviation', 'Sharpe', 'Sortino',
        'Turnover Ratio (%)', 'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)',
        'Avg. Market Cap(in Rs. cr)', 'No. ofStocks', 'Return (%)10 yrs'
    ],
    'DT': [
        'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)', 'Yield To Maturity (%)',
        'Standard Deviation', 'Sharpe'
    ],
    'HY': [
        'Alpha', 'Beta', 'Standard Deviation', 'Sharpe', 'Sortino',
        'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)', 'Yield To Maturity (%)',
        'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)'
    ],
    'Other': [
        'Standard Deviation', 'Sharpe'
    ]
}

HIGHER_IS_BETTER = [
    'AUM(in Rs. cr)', 'Return (%)1 yr', 'Return (%)3 yrs', 'Return (%)5 yrs', 'Return (%)10 yrs',
    'NAV', 'RupeeVestRating', 'Alpha', 'Sharpe', 'Sortino', 'Yield To Maturity (%)',
    'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)', 'Avg. Market Cap(in Rs. cr)', 'No. ofStocks'
]

LOWER_IS_BETTER = [
    'ExpenseRatio (%)', 'Beta', 'Standard Deviation', 'Turnover Ratio (%)',
    'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)'
]

# --- Helper Functions ---
def clean_numeric_column(series):
    # Convert to string, remove %, commas, handle NaNs/empty strings
    series_str = series.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
    # Replace empty strings or specific non-numeric markers if any with NaN
    series_str = series_str.replace(['', '-'], np.nan, regex=False)
    # Convert to numeric, coercing errors
    return pd.to_numeric(series_str, errors='coerce')

def get_fund_category(fund_type_str):
    fund_type_str = str(fund_type_str).lower()
    if 'equity' in fund_type_str:
        return 'EQ'
    elif 'debt' in fund_type_str:
        return 'DT'
    elif 'hybrid' in fund_type_str:
        return 'HY'
    else:
        return 'Other'

def get_relevant_parameters(fund_category):
    params = set(PARAMETER_SETS['Common'])
    if fund_category in PARAMETER_SETS:
        params.update(PARAMETER_SETS[fund_category])
    return list(params)

def rank_funds(df, parameters, weights):
    ranked_df = df.copy()
    # Normalize parameters (0-1 scale)
    scaler = MinMaxScaler()
    numeric_params = [p for p in parameters if p in ranked_df.columns and pd.api.types.is_numeric_dtype(ranked_df[p])]

    # Handle potential division by zero or constant columns in scaler
    valid_params_for_scaling = []
    for p in numeric_params:
        if ranked_df[p].nunique() > 1: # Check if column is not constant
             valid_params_for_scaling.append(p)
        elif ranked_df[p].nunique() == 1:
             # Handle constant columns - assign a neutral score (e.g., 0.5) or skip
             ranked_df[p + '_norm'] = 0.5

    if valid_params_for_scaling:
        try:
            ranked_df[valid_params_for_scaling] = scaler.fit_transform(ranked_df[valid_params_for_scaling])
            # Rename normalized columns
            for p in valid_params_for_scaling:
                ranked_df.rename(columns={p: p + '_norm'}, inplace=True)
        except ValueError as e:
            app.logger.error(f"Error during scaling: {e}")
            # Handle error, maybe return original df or raise exception
            return df # Or raise a custom exception

    # Adjust direction (higher is better / lower is better)
    for p in valid_params_for_scaling:
        norm_col = p + '_norm'
        if p in LOWER_IS_BETTER:
            ranked_df[norm_col] = 1 - ranked_df[norm_col]

    # Calculate weighted score
    ranked_df['Score'] = 0
    total_weight = sum(weights.values())
    if total_weight == 0: total_weight = 1 # Avoid division by zero

    for p, weight in weights.items():
        norm_col = p + '_norm'
        if norm_col in ranked_df.columns:
            # Ensure weight is applied correctly, handle NaNs by treating them as 0 score contribution
            ranked_df['Score'] += ranked_df[norm_col].fillna(0) * (weight / total_weight)

    # Rank based on score
    ranked_df['Rank'] = ranked_df['Score'].rank(method='dense', ascending=False).astype(int)
    return ranked_df.sort_values('Rank')

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    # Diagnostic route - return environment info as JSON
    try:
        app.logger.debug("Accessed diagnostic route /")
        debug_info = {
            'cwd': os.getcwd(),
            'flask_root_path': app.root_path,
            'template_folder_config': app.template_folder,
            'static_folder_config': app.static_folder,
            'template_folder_abs': os.path.abspath(os.path.join(app.root_path, app.template_folder)),
            'static_folder_abs': os.path.abspath(os.path.join(app.root_path, app.static_folder)),
            'jinja_searchpath': list(app.jinja_env.loader.searchpath) if hasattr(app.jinja_env.loader, 'searchpath') else 'N/A',
            'project_root_listing': [],
            'src_listing': [],
            'templates_listing': [],
            'static_listing': []
        }

        project_root = '/opt/render/project/src' # Common Render root
        src_dir = os.path.join(project_root, 'src')
        templates_dir = os.path.join(project_root, 'templates')
        static_dir = os.path.join(project_root, 'static')

        try:
            debug_info['project_root_listing'] = os.listdir(project_root)
        except Exception as e:
            debug_info['project_root_listing'] = f"Error listing {project_root}: {str(e)}"
        try:
            debug_info['src_listing'] = os.listdir(src_dir)
        except Exception as e:
            debug_info['src_listing'] = f"Error listing {src_dir}: {str(e)}"
        try:
            debug_info['templates_listing'] = os.listdir(templates_dir)
        except Exception as e:
            debug_info['templates_listing'] = f"Error listing {templates_dir}: {str(e)}"
        try:
            debug_info['static_listing'] = os.listdir(static_dir)
        except Exception as e:
            debug_info['static_listing'] = f"Error listing {static_dir}: {str(e)}"

        app.logger.debug(f"Diagnostic Info: {json.dumps(debug_info, indent=2)}")
        # Temporarily return JSON instead of rendering template
        # return render_template('index.html')
        return jsonify(debug_info)

    except Exception as e:
        app.logger.error(f"Error in diagnostic route: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to generate diagnostic info', 'details': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            # Basic validation
            if 'SchemeName' not in df.columns or 'Category' not in df.columns:
                 raise ValueError("CSV must contain 'SchemeName' and 'Category' columns.")

            # Clean numeric columns needed for summary/analysis
            cols_to_clean = ['AUM(in Rs. cr)', 'ExpenseRatio (%)', 'Return (%)1 yr', 'Return (%)3 yrs', 'Return (%)5 yrs', 'Return (%)10 yrs', 'NAV', 'RupeeVestRating', 'Alpha', 'Beta', 'Standard Deviation', 'Sharpe', 'Sortino', 'Turnover Ratio (%)', 'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)', 'Avg. Market Cap(in Rs. cr)', 'No. ofStocks', 'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)', 'Yield To Maturity (%)']
            for col in cols_to_clean:
                if col in df.columns:
                    df[col] = clean_numeric_column(df[col])

            # Store dataframe in session (consider alternatives for large files)
            session['df_dict'] = df.to_dict('records')
            session['filename'] = file.filename

            # Generate summary
            fund_counts = df['Category'].value_counts().to_dict()
            summary = {'filename': file.filename, 'fund_counts': fund_counts}

            return jsonify(summary)
        except Exception as e:
            app.logger.error(f"Error processing file {file.filename}: {traceback.format_exc()}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_funds_route():
    if 'df_dict' not in session:
        return jsonify({'error': 'No data uploaded or session expired'}), 400

    df = pd.DataFrame(session['df_dict'])
    weights = request.json.get('weights', {})
    aum_filter = request.json.get('aum_filter', None) # e.g., {'min': 100, 'max': 5000}

    try:
        # Apply AUM filter if provided
        if aum_filter and 'AUM(in Rs. cr)' in df.columns:
            min_aum = aum_filter.get('min')
            max_aum = aum_filter.get('max')
            aum_col = df['AUM(in Rs. cr)']
            if min_aum is not None:
                df = df[aum_col >= min_aum]
            if max_aum is not None:
                df = df[aum_col <= max_aum]

        # Analyze funds by category
        results = {}
        unique_categories = df['Category'].unique()

        for category in unique_categories:
            cat_df = df[df['Category'] == category].copy()
            if cat_df.empty:
                continue

            fund_type = get_fund_category(category)
            relevant_params = get_relevant_parameters(fund_type)

            # Filter weights for relevant params
            relevant_weights = {p: weights.get(p, 1) for p in relevant_params if p in weights}
            # Ensure all relevant params have a default weight if not provided
            for p in relevant_params:
                 if p not in relevant_weights:
                      relevant_weights[p] = 1 # Default weight

            # Check if relevant_params exist in cat_df columns and are numeric
            params_present = [p for p in relevant_params if p in cat_df.columns]
            numeric_params_present = [p for p in params_present if pd.api.types.is_numeric_dtype(cat_df[p])]

            # Drop rows where ALL numeric params needed for ranking are NaN
            cat_df.dropna(subset=numeric_params_present, how='all', inplace=True)

            if not cat_df.empty and numeric_params_present:
                ranked_cat_df = rank_funds(cat_df, numeric_params_present, relevant_weights)
                # Select columns for display (Original + Score + Rank)
                display_cols = ['Rank', 'SchemeName', 'Category', 'Score'] + [p for p in params_present if p not in ['Rank', 'SchemeName', 'Category', 'Score']]
                # Ensure Score is formatted
                ranked_cat_df['Score'] = ranked_cat_df['Score'].round(4)
                results[category] = ranked_cat_df[display_cols].to_dict('records')
            else:
                results[category] = [] # No funds left after filtering or no numeric params

        # Get all unique parameters across all categories present in the data for the UI
        all_params = set(PARAMETER_SETS['Common'])
        for category in unique_categories:
            fund_type = get_fund_category(category)
            all_params.update(PARAMETER_SETS.get(fund_type, []))
        # Filter params that actually exist in the original dataframe
        all_params_in_df = [p for p in all_params if p in pd.DataFrame(session['df_dict']).columns]

        return jsonify({'results': results, 'parameters': sorted(list(all_params_in_df))})

    except Exception as e:
        app.logger.error(f"Error during analysis: {traceback.format_exc()}")
        return jsonify({'error': f'Error during analysis: {str(e)}'}), 500

if __name__ == '__main__':
    # Debugging: Print paths just before running
    app.logger.debug("--- Starting Application ---")
    app.logger.debug(f"Current Working Directory: {os.getcwd()}")
    app.logger.debug(f"Flask Root Path: {app.root_path}")
    app.logger.debug(f"Template Folder (Config): {app.template_folder}")
    app.logger.debug(f"Template Folder (Absolute): {os.path.abspath(os.path.join(app.root_path, app.template_folder))}")
    app.logger.debug(f"Static Folder (Config): {app.static_folder}")
    app.logger.debug(f"Static Folder (Absolute): {os.path.abspath(os.path.join(app.root_path, app.static_folder))}")
    if hasattr(app.jinja_env.loader, 'searchpath'):
        app.logger.debug(f"Jinja Search Paths: {app.jinja_env.loader.searchpath}")
    else:
        app.logger.debug("Jinja loader has no searchpath attribute.")

    # List directories for debugging
    try:
        project_root = '/opt/render/project/src' # Common Render root
        app.logger.debug(f"Listing {project_root}: {os.listdir(project_root)}")
    except Exception as e:
        app.logger.error(f"Error listing {project_root}: {e}")
    try:
        templates_dir = os.path.abspath(os.path.join(app.root_path, app.template_folder))
        app.logger.debug(f"Listing {templates_dir}: {os.listdir(templates_dir)}")
    except Exception as e:
        app.logger.error(f"Error listing {templates_dir}: {e}")

    app.run(host='0.0.0.0', port=5001)

