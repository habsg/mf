import pandas as pd

def generate_summary(file_path, output_summary_file):
    try:
        df = pd.read_csv(file_path)
        
        category_col = 'Category'
        
        if category_col not in df.columns:
            print(f"Error: Column 	'{category_col}	' not found.")
            return

        # Drop rows where 'Category' is NaN to avoid issues
        df_cleaned = df.dropna(subset=[category_col])

        # Count the occurrences of each fund category
        category_counts = df_cleaned[category_col].value_counts().sort_index()

        summary_text = "Initial Summary of Fund Types:\n\n"
        summary_text += "The uploaded file contains mutual funds across the following categories:\n\n"
        summary_text += "{:<15} {:<10}\n".format("Category Code", "Count")
        summary_text += "-"*26 + "\n"
        
        total_funds = 0
        for category, count in category_counts.items():
            summary_text += "{:<15} {:<10}\n".format(category, count)
            total_funds += count
            
        summary_text += "\n" + "-"*26 + "\n"
        summary_text += "{:<15} {:<10}\n".format("Total Funds", total_funds)

        print(summary_text)

        # Save the summary to a file
        with open(output_summary_file, 'w') as f:
            f.write(summary_text)
        print(f"\nSummary saved to {output_summary_file}")

    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    file_path = '/home/ubuntu/upload/screener (2).csv'
    output_file = '/home/ubuntu/fund_summary.txt'
    generate_summary(file_path, output_file)

