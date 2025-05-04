import pandas as pd

def analyze_fund_types(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # User specified Column C (index 2) which is 'Category'
        category_col = 'Category'
        
        unique_categories = []
        if category_col in df.columns:
            # Drop rows where 'Category' is NaN to avoid issues
            df_cleaned = df.dropna(subset=[category_col])
            unique_categories = df_cleaned[category_col].unique().tolist()
            print(f"Unique values in 	'{category_col}	':")
            for cat in sorted(unique_categories): # Sort for better readability
                print(f"- {cat}")
        else:
            print(f"Column 	'{category_col}	' not found.")
            return
            
        fund_types_subtypes = unique_categories
        
        print("\nIdentified Fund Types/Subtypes (from Category column):")
        print(sorted(fund_types_subtypes))
        
        # Save to a file for later use
        output_file = '/home/ubuntu/fund_types.txt'
        with open(output_file, 'w') as f:
            for item in sorted(fund_types_subtypes):
                f.write(f"{item}\n")
        print(f"\nSaved identified types/subtypes to {output_file}")

    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    # Use the new file path
    file_path = '/home/ubuntu/upload/screener (2).csv'
    analyze_fund_types(file_path)

