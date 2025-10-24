# PSICHIC Screening Script

import os
import argparse
import sys

# Add command line argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='PSICHIC fingerprint generation script')
    parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--save_interpretation', type=str, default='True', 
                       help='Whether to save interpretation results (True/False)')
    parser.add_argument('--model', type=str, default='Trained on PDBBind v2020',
                       choices=['Trained on PDBBind v2020', 'Pre-trained on Large-scale Interaction Database'],
                       help='PSICHIC model selection')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Convert string to boolean
    args.save_interpretation = args.save_interpretation.lower() == 'true'
    
    return args

# Parse command line arguments
if __name__ == '__main__':
    args = parse_arguments()
    
    # Use command line arguments to set configuration
    PSICHIC_parameters = args.model
    Save_PSICHIC_Interpretation = args.save_interpretation
    batch_size = args.batch_size
    filename = args.input_csv
    jobname = args.output_dir
else:
    # If imported as module, use default configuration
    PSICHIC_parameters = "Trained on PDBBind v2020"
    Save_PSICHIC_Interpretation = True
    batch_size = 16
    filename = r"c:\Users\Admin\Desktop\LB2\PSICHIC-main\trained_weights\mutated1_mini.csv"
    jobname = r"c:\Users\Admin\Desktop\LB2\PSICHIC-main\trained_weights\screening_results_job1"

#region 1. Configuration Parameters
# ==============================================================================
# Set your screening parameters here
# Configuration now obtained from command line arguments

# @markdown Choose the PSICHIC model.
# "Trained on PDBBind v2020": Most comprehensively evaluated model.
# "Pre-trained on Large-scale Interaction Database": Pre-trained model; consider finetuning for best performance.
# PSICHIC_parameters = "Trained on PDBBind v2020"  # @param ["Trained on PDBBind v2020", "Pre-trained on Large-scale Interaction Database"]

# @markdown Save interpretation from PSICHIC.
# If True, saves interpretation data for further analysis.
# If False, only prediction output is saved.
# Save_PSICHIC_Interpretation = True  # @param {type:"boolean"}

# @markdown Batch size for screening.
# Default is 16. Decrease if protein sequence length is very long.
# batch_size = 16  # @param {type:"integer"}

# @markdown Input CSV file path.
# This file should contain 'Protein' and 'Ligand' columns.
# Example: r'C:\data\input_data\my_screening_list.csv'
# IMPORTANT: Use raw strings (r"...") or double backslashes ("C:\\Users\\...") for Windows paths.
# filename = r"c:\Users\Admin\Desktop\LB2\PSICHIC-main\trained_weights\mutated1_mini.csv"  # <--- TODO: SET YOUR INPUT FILE PATH HERE

# @markdown Output directory name/path.
# Results will be saved in a directory with this name.
# Example: r'C:\data\screening_results_job1'
# jobname = r"c:\Users\Admin\Desktop\LB2\PSICHIC-main\trained_weights\screening_results_job1"  # <--- TODO: SET YOUR JOB/OUTPUT DIRECTORY NAME/PATH HERE
# ==============================================================================
#endregion

#region 2. Setup and Initialization
# ==============================================================================
import json
import pandas as pd
import torch
import numpy as np
# import os # Already imported above
import random
from tqdm import tqdm

# Assuming 'utils', 'models' are in the PYTHONPATH or same directory
# Ensure these custom modules are compatible with your Python environment and accessible.
try:
    from utils.utils import DataLoader, virtual_screening
    from utils.dataset import ProteinMoleculeDataset
    # from utils.trainer import Trainer # Trainer is imported but not used in this script
    from utils.metrics import * # Imports all metrics
    from utils import protein_init, ligand_init
    from models.net import net
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'utils' and 'models' directories are in your Python path or current working directory.")
    print("You might need to run 'pip install -r requirements.txt' if dependencies are missing.")
    exit(1) # Use a non-zero exit code for errors

print("--- PSICHIC Screening Script ---")

# Determine trained model path based on selection using os.path.join for OS compatibility
base_trained_weights_dir = 'trained_weights' # Assuming this is relative to the script or an accessible path
if PSICHIC_parameters == 'Trained on PDBBind v2020':
    trained_model_subpath = 'PDBv2020_PSICHIC'
elif PSICHIC_parameters == 'Pre-trained on Large-scale Interaction Database':
    trained_model_subpath = 'multitask_PSICHIC'
else:
    print(f"Error: Invalid PSICHIC_parameters value: {PSICHIC_parameters}")
    exit(1)
trained_model_path = os.path.join(base_trained_weights_dir, trained_model_subpath)

if not os.path.isdir(trained_model_path): # Check if it's a directory
    print(f"Error: Trained model directory not found: {trained_model_path}")
    print(f"Please ensure the '{base_trained_weights_dir}' directory and its subdirectories ('{trained_model_subpath}') are correctly placed relative to the script or an absolute path is configured.")
    exit(1)

screenfile = filename # Already an absolute path from config
result_path = jobname   # Already an absolute path from config

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Load model configuration
config_file_path = os.path.join(trained_model_path, 'config.json')
config = None
try:
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    print("Model configuration loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model configuration file not found: {config_file_path}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from configuration file {config_file_path}: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading configuration file {config_file_path}: {e}")
    exit(1)

print(f"Screening the CSV file: {screenfile}")

# Create results directory if it doesn't exist
try:
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Created output directory: {result_path}")
except OSError as e:
    print(f"Error creating output directory {result_path}: {e}")
    exit(1)

# Load degree dictionary
degree_file_path = os.path.join(trained_model_path, 'degree.pt')
degree_dict = None
try:
    degree_dict = torch.load(degree_file_path, map_location=device)
    print("Degree dictionary loaded successfully.")
except FileNotFoundError:
    print(f"Error: Degree file ('degree.pt') not found in {trained_model_path}")
    exit(1)
except Exception as e: # Catch other potential errors during torch.load (e.g., corrupted file)
    print(f"Error loading degree file {degree_file_path}: {e}")
    exit(1)

# Validate degree_dict content
if not isinstance(degree_dict, dict) or 'ligand_deg' not in degree_dict or 'protein_deg' not in degree_dict:
    print(f"Error: Degree dictionary from {degree_file_path} is invalid or missing required keys ('ligand_deg', 'protein_deg').")
    if isinstance(degree_dict, dict):
        print(f"Available keys: {list(degree_dict.keys())}")
    exit(1)
mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']

# Model parameters file path
param_file_path = os.path.join(trained_model_path, 'model.pt')
if not os.path.exists(param_file_path):
    print(f"Error: Model parameters file ('model.pt') not found in {trained_model_path}")
    exit(1)

# Initialize the model
print("Initializing PSICHIC model...")
try:
    model = net(mol_deg, prot_deg,
                # MOLECULE
                mol_in_channels=config['params']['mol_in_channels'], prot_in_channels=config['params']['prot_in_channels'],
                prot_evo_channels=config['params']['prot_evo_channels'],
                hidden_channels=config['params']['hidden_channels'], pre_layers=config['params']['pre_layers'],
                post_layers=config['params']['post_layers'], aggregators=config['params']['aggregators'],
                scalers=config['params']['scalers'], total_layer=config['params']['total_layer'],
                K=config['params']['K'], heads=config['params']['heads'],
                dropout=config['params']['dropout'],
                dropout_attn_score=config['params']['dropout_attn_score'],
                # output
                regression_head=config['tasks']['regression_task'],
                classification_head=config['tasks']['classification_task'],
                multiclassification_head=config['tasks']['mclassification_task'],
                device=device).to(device)

    model.reset_parameters() # As per original code
    model.load_state_dict(torch.load(param_file_path, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded and initialized successfully.")
except KeyError as e:
    print(f"Error: Missing key in model configuration (config.json) during model initialization: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during model initialization or loading weights: {e}")
    exit(1)
# ==============================================================================
#endregion

#region 3. Data Loading and Preprocessing
# ==============================================================================
print(f"Loading screening data from: {screenfile}")
screen_df = None
try:
    screen_df = pd.read_csv(screenfile)
except FileNotFoundError:
    print(f"Error: Input screen file not found: {screenfile}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: Input screen file is empty: {screenfile}")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file {screenfile}: {e}")
    exit(1)

if 'Protein' not in screen_df.columns or 'Ligand' not in screen_df.columns:
    print("Error: The input CSV file must contain 'Protein' and 'Ligand' columns.")
    exit(1)

protein_seqs = screen_df['Protein'].astype(str).unique().tolist()
print('(1) Initialising Protein Sequences to Protein Graphs...')
protein_dict = protein_init(protein_seqs) # Assuming protein_init handles potential errors and returns dict

ligand_smiles = screen_df['Ligand'].astype(str).unique().tolist()
print('(2) Initialising Ligand SMILES to Ligand Graphs...')
ligand_dict = ligand_init(ligand_smiles) # Assuming ligand_init handles potential errors and returns dict

# Clear GPU cache if CUDA is used
if device.type == 'cuda':
    torch.cuda.empty_cache()

# Drop any invalid SMILES that couldn't be processed
original_rows = len(screen_df)
screen_df = screen_df[screen_df['Ligand'].isin(list(ligand_dict.keys()))].reset_index(drop=True)
if len(screen_df) < original_rows:
    print(f"Warning: Dropped {original_rows - len(screen_df)} rows due to invalid/unparsable ligand SMILES.")

if len(screen_df) == 0:
    print("Error: No valid ligand SMILES found after initialization. Cannot proceed with screening.")
    exit(1)

# Create dataset and dataloader
print("Creating dataset and dataloader...")
screen_loader = None
try:
    screen_dataset = ProteinMoleculeDataset(screen_df, ligand_dict, protein_dict, device=device)
    screen_loader = DataLoader(screen_dataset, batch_size=batch_size, shuffle=False,
                               follow_batch=['mol_x', 'clique_x', 'prot_node_aa'])
except Exception as e:
    print(f"Error creating dataset or dataloader: {e}")
    exit(1)
# ==============================================================================
#endregion

#region 4. Run Screening
# ==============================================================================
print("(3) Starting Screening...")
screen_df_results = None
try:
    with torch.no_grad(): # Ensure no gradients are computed during inference
        screen_df_results = virtual_screening(screen_df.copy(), # Pass a copy
                                              model, screen_loader,
                                              result_path=os.path.join(result_path, "interpretation_result"),
                                              save_interpret=Save_PSICHIC_Interpretation,
                                              ligand_dict=ligand_dict,
                                              device=device)
except Exception as e:
    print(f"Error during virtual screening: {e}")
    exit(1)

if screen_df_results is None:
    print("Error: Virtual screening did not return results.")
    exit(1)

# Save screening results
output_csv_path = os.path.join(result_path, 'screening_results.csv')
try:
    screen_df_results.to_csv(output_csv_path, index=False)
    print(f"Screening completed. Results saved to: {output_csv_path}")
except Exception as e:
    print(f"Error saving results to CSV ({output_csv_path}): {e}")
    exit(1)

print("--- Script Finished Successfully ---")
# ==============================================================================
#endregion

if __name__ == '__main__':
    # This script is designed to be run directly.
    # The main execution logic is sequential.
    pass