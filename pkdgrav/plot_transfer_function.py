import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparison(file_configs, output_path):
    """
    file_configs: List of dictionaries, each containing:
        'path': path to the .dat file
        'k_col': column index for k
        't_col': column index for T_tot/k2
        'label': label for the legend
        'color': color for the plot
    """
    plt.figure(figsize=(10, 6))

    for config in file_configs:
        try:
            # Load data. Adjust skiprows if your files have different header lengths
            data = np.loadtxt(config['path'], skiprows=4)
            
            k = data[:, config['k_col']]
            t_tot = data[:, config['t_col']]
            
            # Using loglog as per your requirement
            plt.loglog(k, t_tot, label=config['label'], color=config['color'], linewidth=2)
            
        except Exception as e:
            print(f"Error processing {config['path']}: {e}")

    # Plot formatting
    plt.xlabel('k [h/Mpc]', fontsize=12)
    plt.ylabel('Transfer Function [-T(k)/k²]', fontsize=12)
    plt.title('Comparison of Matter Transfer Functions (z=0)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/comparison_transfer_function.png", dpi=300)
    print(f"Plot saved to {output_path}/comparison_transfer_function.png")
    plt.close()

# Configuration for your two files
file_configs = [
    {
        'path': '/capstor/scratch/cscs/damrein/cosmogridv1_fiducial/run_0000/transfer_fiducial.dat',
        'k_col': 0,
        't_col': 1,
        'label': 'David',
        'color': 'black'
    },
    {
        'path': '/capstor/scratch/cscs/damrein/class_backscaling/transfer_fiducial.dat',
        'k_col': 0,
        't_col': 1,
        'label': 'with nu',
        'color': 'red'
    }
]

output_path = '/capstor/scratch/cscs/damrein/outputs/plots/transfer_functions'
plot_comparison(file_configs, output_path)