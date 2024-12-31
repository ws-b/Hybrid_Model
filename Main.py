import os
import glob
from Physics_Power import process_files_power
from Train_Multi import run_evaluate, plot_rmse_results

def main():

    file_path = "" # Set BMS_Data path here
    files = glob.glob(os.path.join(file_path, "*.csv"))

    process_files_power(files)

    results_dict = run_evaluate(files)
    plot_rmse_results(results_dict, None)

if __name__ == "__main__":
    main()
