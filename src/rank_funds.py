import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define parameter sets based on the previously created file
PARAMETER_SETS = {
    'Common': [
        'AUM(in Rs. cr)', 'ExpenseRatio (%)', 'Return (%)1 yr', 'Return (%)3 yrs', 
        'Return (%)5 yrs', 'NAV', 'RupeeVestRating'
        # 'Fund Manager', 'Inception Date' are informational, not typically ranked
    ],
    'EQ': [
        'Alpha', 'Beta', 'Standard Deviation', 'Sharpe', 'Sortino', 
        'Turnover Ratio (%)', 'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)',
        'Avg. Market Cap(in Rs. cr)', 'No. ofStocks', 'Return (%)10 yrs'
        # 'Highest Sector' is categorical
    ],
    'DT': [
        'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)', 'Yield To Maturity (%)', 
        'Standard Deviation', 'Sharpe' # Sharpe has lower relevance but can be included
    ],
    'HY': [
        'Alpha', 'Beta', 'Standard Deviation', 'Sharpe', 'Sortino',
        'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)', 'Yield To Maturity (%)',
        'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)'
        # 'Highest Sector' is categorical
    ],
    'Other': [
        'Standard Deviation', 'Sharpe' # Lower relevance for Gold
    ]
}

# Define which parameters are better when higher
HIGHER_IS_BETTER = [
    'AUM(in Rs. cr)', 'Return (%)1 yr', 'Return (%)3 yrs', 'Return (%)5 yrs', 'Return (%)10 yrs',
    'NAV', 'RupeeVestRating', 'Alpha', 'Sharpe', 'Sortino', 'Yield To Maturity (%)',
    'Large Cap(%)', 'Mid Cap(%)', 'Small Cap(%)', 'Avg. Market Cap(in Rs. cr)', 'No. ofStocks'
]

# Define which parameters are better when lower
LOWER_IS_BETTER = [
    'ExpenseRatio (%)', 'Beta', 'Standard Deviation', 'Turnover Ratio (%)',
    'Avg. Maturity(in yrs)', 'Mod. Duration(in yrs)'
]

def clean_numeric_column(series):
    # Convert to string, remove specific characters, replace non-numeric markers with NaN
    series_str = series.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.replace(' cr', '', regex=False)
    # Replace common non-numeric placeholders like '-' or 'Unrated' before attempting numeric conversion
    series_cleaned = series_str.replace(['-', 'Unrated', 'nan', 'None'], np.nan, regex=False)
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(series_cleaned, errors='coerce')

def rank_funds(file_path, output_file):
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")

        # --- Data Cleaning --- 
        # Identify all potential numeric columns from parameter sets
        all_params = set(PARAMETER_SETS['Common'])
        for key in ['EQ', 'DT', 'HY', 'Other']:
            all_params.update(PARAMETER_SETS[key])
        
        numeric_cols_to_clean = [col for col in all_params if col in df.columns]
        print(f"\nCleaning {len(numeric_cols_to_clean)} potential numeric columns...")

        for col in numeric_cols_to_clean:
            df[col] = clean_numeric_column(df[col])
            # Optional: Print stats after cleaning
            # print(f"Cleaned 	'{col}	': Missing values = {df[col].isnull().sum()}")

        # Handle 'Category' missing values if any
        df = df.dropna(subset=['Category'])
        print(f"Shape after dropping rows with missing Category: {df.shape}")

        # --- Ranking Logic --- 
        df['RankScore'] = 0.0
        df['Rank'] = 0
        scaler = MinMaxScaler()

        # Get unique categories
        categories = df['Category'].unique()
        print(f"\nRanking funds within {len(categories)} categories...")

        ranked_dfs = []

        for category in categories:
            df_cat = df[df['Category'] == category].copy()
            if df_cat.empty:
                continue

            # Determine parameter set
            prefix = category.split('-')[0]
            params = list(PARAMETER_SETS['Common']) # Start with common
            if prefix == 'EQ':
                params.extend(PARAMETER_SETS['EQ'])
            elif prefix == 'DT':
                params.extend(PARAMETER_SETS['DT'])
            elif prefix == 'HY':
                params.extend(PARAMETER_SETS['HY'])
            elif prefix in ['FOF', 'GOLD']:
                params.extend(PARAMETER_SETS['Other'])
            
            # Filter params that actually exist in the dataframe and are numeric
            valid_params = [p for p in params if p in df_cat.columns and pd.api.types.is_numeric_dtype(df_cat[p])]
            
            # Keep only params with more than 1 unique non-NA value for scaling
            rankable_params = []
            for p in valid_params:
                if df_cat[p].nunique(dropna=True) > 1:
                    rankable_params.append(p)
                # else: # Handle columns with single value or all NaNs
                #     print(f"Skipping parameter 	'{p}	' for category 	'{category}	' due to insufficient unique values.")
            
            if not rankable_params:
                # print(f"No rankable parameters found for category 	'{category}	'. Skipping ranking.")
                df_cat['RankScore'] = 0 # Assign default score/rank if no params
                df_cat['Rank'] = 1
                ranked_dfs.append(df_cat)
                continue

            # Impute missing values with the median *within the category* for ranking purposes
            # Alternatively, could drop rows with NaNs in rankable_params, but imputation keeps more funds
            for p in rankable_params:
                median_val = df_cat[p].median()
                if pd.notna(median_val):
                    df_cat[p].fillna(median_val, inplace=True)
                else: # If median is NaN (e.g., all values were NaN), fill with 0 or remove param
                    df_cat[p].fillna(0, inplace=True)
            
            # Normalize rankable parameters
            try:
                df_cat[rankable_params] = scaler.fit_transform(df_cat[rankable_params])
            except ValueError as e:
                # print(f"Error scaling category 	'{category}	': {e}. Skipping ranking for this category.")
                df_cat['RankScore'] = 0
                df_cat['Rank'] = 1
                ranked_dfs.append(df_cat)
                continue

            # Calculate score (equal weight initially)
            cat_score = 0.0
            num_params = len(rankable_params)
            weight = 1.0 / num_params if num_params > 0 else 0

            for p in rankable_params:
                if p in HIGHER_IS_BETTER:
                    cat_score += df_cat[p] * weight
                elif p in LOWER_IS_BETTER:
                    cat_score += (1 - df_cat[p]) * weight
                # Parameters not in either list are ignored for scoring

            df_cat['RankScore'] = cat_score
            
            # Rank based on score (higher score is better)
            df_cat['Rank'] = df_cat['RankScore'].rank(method='dense', ascending=False).astype(int)
            ranked_dfs.append(df_cat.sort_values('Rank'))

        # Combine ranked dataframes
        if ranked_dfs:
            final_df = pd.concat(ranked_dfs)
            # Save results
            final_df.to_csv(output_file, index=False)
            print(f"\nRanking complete. Results saved to {output_file}")
            print(f"Final shape: {final_df.shape}")
        else:
            print("\nNo funds were ranked.")

    except Exception as e:
        print(f"Error ranking funds: {e}")

if __name__ == "__main__":
    input_file = '/home/ubuntu/upload/screener (2).csv'
    output_file = '/home/ubuntu/ranked_funds.csv'
    rank_funds(input_file, output_file)

