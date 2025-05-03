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

# --- Flask App Setup ---
app = Flask(__name__, template_folder=\'src/templates\', static_folder=\'src/static\')
app.secret_key = secrets.token_hex(16) # For session management

# Define upload folder (relative to main.py location)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), \'uploads\')
app.config[\'UPLOAD_FOLDER\'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Parameter Definitions (from parameter_sets.txt) ---
PARAMETER_SETS = {
    # ... (parameter sets will be loaded dynamically or defined here) ...
    # For simplicity, defining directly here based on previous analysis
    \'Common\': [
        \'AUM(in Rs. cr)\", \'ExpenseRatio (%)\", \'Return (%)1 yr\', \'Return (%)3 yrs\", 
        \'Return (%)5 yrs\', \'NAV\', \'RupeeVestRating\'
    ],
    \'EQ\': [
        \'Alpha\', \'Beta\', \'Standard Deviation\', \'Sharpe\', \'Sortino\", 
        \'Turnover Ratio (%)\", \'Large Cap(%)\", \'Mid Cap(%)\", \'Small Cap(%)\",
        \'Avg. Market Cap(in Rs. cr)\", \'No. ofStocks\', \'Return (%)10 yrs\'
    ],
    \'DT\': [
        \'Avg. Maturity(in yrs)\", \'Mod. Duration(in yrs)\", \'Yield To Maturity (%)\", 
        \'Standard Deviation\', \'Sharpe\'
    ],
    \'HY\': [
        \'Alpha\', \'Beta\', \'Standard Deviation\', \'Sharpe\', \'Sortino\",
        \'Avg. Maturity(in yrs)\", \'Mod. Duration(in yrs)\", \'Yield To Maturity (%)\",
        \'Large Cap(%)\", \'Mid Cap(%)\", \'Small Cap(%)\"
    ],
    \'Other\': [
        \'Standard Deviation\', \'Sharpe\'
    ]
}

HIGHER_IS_BETTER = [
    \'AUM(in Rs. cr)\", \'Return (%)1 yr\', \'Return (%)3 yrs\', \'Return (%)5 yrs\', \'Return (%)10 yrs\",
    \'NAV\', \'RupeeVestRating\', \'Alpha\', \'Sharpe\', \'Sortino\', \'Yield To Maturity (%)\",
    \'Large Cap(%)\", \'Mid Cap(%)\", \'Small Cap(%)\", \'Avg. Market Cap(in Rs. cr)\", \'No. ofStocks\'
]

LOWER_IS_BETTER = [
    \'ExpenseRatio (%)\", \'Beta\', \'Standard Deviation\', \'Turnover Ratio (%)\",
    \'Avg. Maturity(in yrs)\", \'Mod. Duration(in yrs)\"
]

# --- Helper Functions ---
def clean_numeric_column(series):
    series_str = series.astype(str).str.replace(\'%\', \'\', regex=False).str.replace(\,\', \'\', regex=False).str.replace(\' cr\', \'\', regex=False)
    series_cleaned = series_str.replace([\'-', \'Unrated\', \'nan\', \'None\'], np.nan, regex=False)
    return pd.to_numeric(series_cleaned, errors=\'coerce\')

def get_fund_summary(df):
    category_col = \'Category\'
    if category_col not in df.columns:
        return "Error: Column \'Category\' not found.", None
    df_cleaned = df.dropna(subset=[category_col])
    category_counts = df_cleaned[category_col].value_counts().sort_index()
    summary = category_counts.to_dict()
    total_funds = int(df_cleaned.shape[0]) # Use shape[0] for total count
    return summary, total_funds

def get_relevant_parameters(df):
    categories = df[\'Category\'].dropna().unique()
    params_by_category = {}
    all_available_params = set()
    
    for category in categories:
        prefix = category.split(\'-\' )[0]
        params = list(PARAMETER_SETS[\'Common\'])
        if prefix == \'EQ\':
            params.extend(PARAMETER_SETS[\'EQ\'])
        elif prefix == \'DT\':
            params.extend(PARAMETER_SETS[\'DT\'])
        elif prefix == \'HY\':
            params.extend(PARAMETER_SETS[\'HY\'])
        elif prefix in [\'FOF\', \'GOLD\']:
            params.extend(PARAMETER_SETS[\'Other\'])
        
        # Filter params that actually exist in the dataframe columns
        valid_params = [p for p in params if p in df.columns]
        params_by_category[category] = valid_params
        all_available_params.update(valid_params)
        
    # Filter out params that are not numeric or have no variance for weighting/ranking
    rankable_params = set()
    df_copy = df.copy()
    for p in all_available_params:
        try:
            df_copy[p] = clean_numeric_column(df_copy[p])
            if pd.api.types.is_numeric_dtype(df_copy[p]) and df_copy[p].nunique(dropna=True) > 1:
                 # Check if param is scorable (higher or lower is better)
                 if p in HIGHER_IS_BETTER or p in LOWER_IS_BETTER:
                    rankable_params.add(p)
        except Exception:
            continue # Ignore columns that cause errors during cleaning/checking
            
    return sorted(list(rankable_params))

def rank_funds_with_weights(df_original, weights):
    df = df_original.copy()
    print(f"Ranking with weights: {weights}")
    
    # --- Data Cleaning (on the copy) ---
    all_params = set(PARAMETER_SETS[\'Common\'])
    for key in [\'EQ\', \'DT\', \'HY\', \'Other\']:
        all_params.update(PARAMETER_SETS[key])
    numeric_cols_to_clean = [col for col in all_params if col in df.columns]
    for col in numeric_cols_to_clean:
        df[col] = clean_numeric_column(df[col])
        
    df = df.dropna(subset=[\'Category\'])

    # --- Ranking Logic ---
    df[\'RankScore\'] = 0.0
    df[\'Rank\'] = 0
    scaler = MinMaxScaler()
    categories = df[\'Category\'].unique()
    ranked_dfs = []

    # Normalize weights (sum to 1)
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight if total_weight > 0 else 0 for k, v in weights.items()}

    for category in categories:
        df_cat = df[df[\'Category\'] == category].copy()
        if df_cat.empty:
            continue

        prefix = category.split(\'-\' )[0]
        params = list(PARAMETER_SETS[\'Common\'])
        if prefix == \'EQ\': params.extend(PARAMETER_SETS[\'EQ\'])
        elif prefix == \'DT\': params.extend(PARAMETER_SETS[\'DT\'])
        elif prefix == \'HY\': params.extend(PARAMETER_SETS[\'HY\'])
        elif prefix in [\'FOF\', \'GOLD\']: params.extend(PARAMETER_SETS[\'Other\'])
        
        valid_params = [p for p in params if p in df_cat.columns and pd.api.types.is_numeric_dtype(df_cat[p])]
        
        # Use only parameters provided in weights for ranking
        rankable_params = [p for p in valid_params if p in weights and df_cat[p].nunique(dropna=True) > 1]
        
        if not rankable_params:
            df_cat[\'RankScore\'] = 0
            df_cat[\'Rank\'] = 1
            ranked_dfs.append(df_cat)
            continue

        # Impute missing values with median within the category
        for p in rankable_params:
            median_val = df_cat[p].median()
            if pd.notna(median_val):
                df_cat[p].fillna(median_val, inplace=True)
            else:
                df_cat[p].fillna(0, inplace=True)
        
        # Normalize rankable parameters
        try:
            df_cat[rankable_params] = scaler.fit_transform(df_cat[rankable_params])
        except ValueError:
            df_cat[\'RankScore\'] = 0
            df_cat[\'Rank\'] = 1
            ranked_dfs.append(df_cat)
            continue

        # Calculate weighted score
        cat_score = 0.0
        for p in rankable_params:
            weight = normalized_weights.get(p, 0) # Get normalized weight
            if weight == 0: continue # Skip params with zero weight
            
            if p in HIGHER_IS_BETTER:
                cat_score += df_cat[p] * weight
            elif p in LOWER_IS_BETTER:
                cat_score += (1 - df_cat[p]) * weight

        df_cat[\'RankScore\'] = cat_score
        df_cat[\'Rank\'] = df_cat[\'RankScore\'].rank(method=\'dense\', ascending=False).astype(int)
        ranked_dfs.append(df_cat.sort_values(\'Rank\'))

    if ranked_dfs:
        final_df = pd.concat(ranked_dfs)
        return final_df
    else:
        return df # Return original df if no ranking occurred

# --- Flask Routes ---
@app.route(\'/\', methods=[\'GET\', \'POST\'])
def index():
    if request.method == \'POST\':
        if \'file\' not in request.files:
            return \'No file part\', 400
        file = request.files[\'file\']
        if file.filename == \'\':
            return \'No selected file\', 400
        if file and file.filename.endswith(\' .csv\'):
            filepath = os.path.join(app.config[\'UPLOAD_FOLDER\'], \'uploaded_data.csv\')
            file.save(filepath)
            
            # Store filepath in session
            session[\'filepath\'] = filepath
            
            # Process the file immediately for summary and parameters
            try:
                df = pd.read_csv(filepath)
                session[\'original_columns\'] = df.columns.tolist() # Store original columns
                summary, total_funds = get_fund_summary(df)
                rankable_params = get_relevant_parameters(df)
                
                # Store summary and params in session
                session[\'summary\'] = summary
                session[\'total_funds\'] = total_funds
                session[\'rankable_params\'] = rankable_params
                
                # Initial ranking with equal weights for display
                initial_weights = {param: 1 for param in rankable_params}
                ranked_df = rank_funds_with_weights(df, initial_weights)
                session[\'ranked_data\'] = ranked_df.to_json(orient=\'split\')
                
                # Get AUM range
                aum_col = \'AUM(in Rs. cr)\'
                min_aum, max_aum = 0, 100000 # Default
                if aum_col in df.columns:
                    df[aum_col] = clean_numeric_column(df[aum_col])
                    min_aum = int(df[aum_col].min()) if pd.notna(df[aum_col].min()) else 0
                    max_aum = int(df[aum_col].max()) if pd.notna(df[aum_col].max()) else 100000
                session[\'min_aum\'] = min_aum
                session[\'max_aum\'] = max_aum
                session[\'current_min_aum\'] = min_aum
                session[\'current_max_aum\'] = max_aum

                return render_template(\'results.html\', 
                                       summary=summary, 
                                       total_funds=total_funds,
                                       parameters=rankable_params,
                                       weights=initial_weights, # Pass initial weights
                                       min_aum=min_aum,
                                       max_aum=max_aum,
                                       current_min_aum=min_aum,
                                       current_max_aum=max_aum,
                                       ranked_data=ranked_df.head(50).to_dict(orient=\'records\'), # Show top 50 initially
                                       columns=ranked_df.columns.tolist())
            except Exception as e:
                print(f"Error processing file: {e}")
                return f"Error processing file: {e}", 500
        else:
            return \'Invalid file type\', 400
            
    # Clear session on GET request to index
    session.clear()
    return render_template(\'index.html\')

@app.route(\'/update_ranking\', methods=[\'POST\'])
def update_ranking():
    if \'filepath\' not in session:
        return jsonify({\'error\': \'No file uploaded or session expired\'}), 400
        
    try:
        weights = {}
        data = request.get_json()
        raw_weights = data.get(\'weights\', {})
        min_aum_filter = data.get(\'min_aum\', session.get(\'min_aum\'))
        max_aum_filter = data.get(\'max_aum\', session.get(\'max_aum\'))
        
        # Convert weights to float
        for param, weight in raw_weights.items():
            try:
                weights[param] = float(weight)
            except (ValueError, TypeError):
                weights[param] = 0 # Default to 0 if conversion fails

        # Store current filter values
        session[\'current_min_aum\'] = min_aum_filter
        session[\'current_max_aum\'] = max_aum_filter
        session[\'current_weights\'] = weights # Store current weights

        # Load original data
        df_original = pd.read_csv(session[\'filepath\'])
        
        # Re-rank with new weights
        ranked_df = rank_funds_with_weights(df_original, weights)
        
        # Apply AUM filter (Step 008)
        aum_col = \'AUM(in Rs. cr)\'
        filtered_df = ranked_df # Start with ranked data
        if aum_col in ranked_df.columns:
            # Ensure AUM column is numeric for filtering
            ranked_df[aum_col] = clean_numeric_column(ranked_df[aum_col])
            filtered_df = ranked_df[
                (ranked_df[aum_col] >= min_aum_filter) & 
                (ranked_df[aum_col] <= max_aum_filter)
            ]
        
        # Store filtered data in session (optional, maybe just return)
        # session[\'filtered_data\'] = filtered_df.to_json(orient=\'split\')
        
        # Return limited results for display
        results = filtered_df.head(100).to_dict(orient=\'records\') # Limit results sent back
        return jsonify({
            \'ranked_data\': results,
            \'columns\': filtered_df.columns.tolist(),
            \'total_filtered_count\': len(filtered_df)
        })

    except Exception as e:
        print(f"Error updating ranking: {e}")
        return jsonify({\'error\': f\'Error updating ranking: {e}\'}), 500

# --- Main Execution ---
if __name__ == \'__main__\':
    # Make sure to run with 0.0.0.0 to be accessible externally if needed
    app.run(host=\'0.0.0.0\', port=5001, debug=True) # Use a different port like 5001

