import pandas as pd

# Define the set of allowed amino acid characters (must be uppercase)
ALLOWED_AMINO_ACIDS = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'])

def is_protein_sequence_valid(sequence):
    """
    Checks if all characters in a protein sequence are valid.
    Assumes sequence is a string. Returns False if sequence is not a string (e.g., NaN).
    """
    if not isinstance(sequence, str):
        return False # Handles empty cells or non-string data
    for amino_acid in sequence:
        if amino_acid not in ALLOWED_AMINO_ACIDS:
            return False
    return True

def clean_protein_data(input_filepath, output_filepath, protein_column_name='Protein'):
    """
    Reads a CSV file, removes rows with invalid protein sequences,
    and saves the cleaned data to a new CSV file.
    """
    try:
        # Try reading as CSV first
        try:
            df = pd.read_csv(input_filepath)
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError reading {input_filepath}. Trying with 'latin1' encoding.")
            df = pd.read_csv(input_filepath, encoding='latin1')
        except Exception as e: # Catch other potential CSV read errors
             print(f"Could not read '{input_filepath}' as CSV. Attempting to read as Excel.")
             print(f"CSV read error: {e}")
             # If CSV fails, try Excel (common request)
             try:
                df = pd.read_excel(input_filepath)
             except Exception as ex_err:
                print(f"Error: Could not read file '{input_filepath}' as CSV or Excel.")
                print(f"Excel read error: {ex_err}")
                return
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
        return

    if protein_column_name not in df.columns:
        print(f"Error: Protein column '{protein_column_name}' not found in the input file.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    initial_row_count = len(df)
    print(f"Initial number of rows: {initial_row_count}")

    # Identify invalid rows
    # Ensure protein sequences are treated as strings, convert NaNs or other types to empty string for check
    is_valid_series = df[protein_column_name].apply(lambda x: is_protein_sequence_valid(str(x)) if pd.notna(x) else False)
    
    # Keep only rows where the protein sequence is valid
    cleaned_df = df[is_valid_series]
    
    final_row_count = len(cleaned_df)
    rows_removed = initial_row_count - final_row_count

    print(f"Number of rows removed due to invalid protein sequences: {rows_removed}")
    print(f"Final number of rows: {final_row_count}")

    if rows_removed > 0:
        # Optional: Show which sequences were problematic if you want more detail
        invalid_df = df[~is_valid_series]
        print("\n--- Example problematic protein sequences removed (first 5) ---")
        for seq in invalid_df[protein_column_name].head(5):
            print(seq)
        print("-----------------------------------------------------------------")


    try:
        # Save the cleaned dataframe
        # If output ends with .xlsx, save as Excel, otherwise CSV
        if output_filepath.lower().endswith('.xlsx'):
            cleaned_df.to_excel(output_filepath, index=False)
        else:
            cleaned_df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to '{output_filepath}'")
    except Exception as e:
        print(f"Error saving cleaned data to '{output_filepath}': {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Replace with your actual file paths
    # Your input file (the one causing the error in PSICHIC)
    input_file = r"./Dataset/screening/LB_origin.csv" #the input file path, example: "data/dataset.csv"
    
    # Where to save the cleaned file
    output_file = r"./Dataset/screening/LB_processed.csv" #the output file path, example: "data/dataset_cleaned.csv"
    
    # The name of the column containing protein sequences
    protein_col = "Protein" 
    # --- End Configuration ---

    print(f"Starting protein data cleaning for '{input_file}'...")
    clean_protein_data(input_file, output_file, protein_column_name=protein_col)
    print("Cleaning process finished.")