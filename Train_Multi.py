import os
import numpy as np
import random
import matplotlib.pyplot as plt
from threading import Thread
from Functions import compute_rmse
from Train_XGboost import run_workflow as xgb_run_workflow
from Train_Only_XGboost import run_workflow as only_run_workflow
from Train_LinearR import cross_validate as lr_cross_validate
from Train_Only_LR import cross_validate as only_lr_validate
import json

# This code contains comments generated using ChatGPT.


def run_xgb_cross_validate(files, adjusted_params_XGB, results_dict, size):
    """
    Calculates the RMSE of the 'Hybrid XGBoost' model using the sampled data
    and stores the result in results_dict[size].
    """
    try:
        # Execute the XGBoost workflow with the given files and adjusted parameters
        xgb_results, _ = xgb_run_workflow(
            files,               # A list of CSV file paths
            plot=False,          # Skip plotting
            save_dir=None,       # Skip saving the model
            predefined_best_params=adjusted_params_XGB
        )
        if xgb_results:
            # Extract RMSE values from the XGBoost results
            rmse_values = [xgb_result['rmse'] for xgb_result in xgb_results]
            results_dict[size].append({
                'model': 'Hybrid Model(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"XGBoost cross_validate error: {e}")


def run_only_xgb_validate(files, adjusted_params_ML, results_dict, size):
    """
    Calculates the RMSE of the 'Only ML(XGBoost)' model using the sampled data
    and stores the result in results_dict[size].
    """
    try:
        # Execute the 'Only XGBoost' workflow
        only_xgb_results, _ = only_run_workflow(
            files,
            plot=False,
            save_dir=None,
            predefined_best_params=adjusted_params_ML
        )
        if only_xgb_results:
            # Extract RMSE values from the results
            rmse_values = [only_xgb_result['rmse'] for only_xgb_result in only_xgb_results]
            results_dict[size].append({
                'model': 'Only ML(XGBoost)',
                'rmse': rmse_values
            })
    except Exception as e:
        print(f"Only ML(XGBoost) cross_validate error: {e}")


def run_evaluate(files):
    """
    1) Sample subsets of various sizes (e.g., 10, 20, 50, ...) from the input data,
       then measure RMSEs of several models (Physics, Hybrid/Only XGBoost, Hybrid/Only LR).
    2) Obtain the best parameters using the entire dataset (e.g., via Optuna) first,
       then scale those parameters for each sampled subset.
    3) Store the measurement results in results_dict and also save them to a JSON file.
    """
    # Candidate subset sizes
    candidate_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000]
    results_dict = {}
    # Get the total number of input files (trips)
    max_samples = len(files)

    # 1) Filter out subset sizes that exceed the total number of files
    filtered_sizes = [s for s in candidate_sizes if s <= max_samples]
    if max_samples not in filtered_sizes:
        filtered_sizes.append(max_samples)
    filtered_sizes = sorted(filtered_sizes)

    # 2) Use the entire dataset to find best_params
    #    - "Hybrid XGBoost"
    results_hybrid_xgb, _ = xgb_run_workflow(
        files, plot=False, save_dir=None, predefined_best_params=None
    )
    # Assume best_params are stored at results_hybrid_xgb[0]['best_params']
    best_params_hybrid_xgb = results_hybrid_xgb[0]['best_params']

    #    - "Only XGBoost"
    results_only_xgb, _ = only_run_workflow(
        files, plot=False, save_dir=None, predefined_best_params=None
    )
    best_params_only_xgb = results_only_xgb[0]['best_params']

    # 3) For each size in filtered_sizes, do multiple samplings
    for size in filtered_sizes:
        if size not in results_dict:
            results_dict[size] = []

        # Decide how many times to sample
        if size < 20:
            samplings = 200
        elif 20 <= size < 50:
            samplings = 10
        elif 50 <= size <= 100:
            samplings = 5
        else:
            samplings = 1

        # Perform repeated sampling
        for _ in range(samplings):
            sampled_files = random.sample(files, size)

            # (a) Compute the Physics-based RMSE
            #     Assume compute_rmse accepts a list of file paths
            rmse_physics = compute_rmse(sampled_files)
            if rmse_physics is not None:
                results_dict[size].append({
                    'model': 'Physics-Based',
                    'rmse': [rmse_physics]
                })

            # (b) Scale parameters by the ratio (max_samples / size)
            adjustment_factor = max_samples / size

            adjusted_params_XGB = best_params_hybrid_xgb.copy()
            for param in adjusted_params_XGB:
                # Exclude 'eta' from the scaling
                if param != 'eta':
                    adjusted_params_XGB[param] *= adjustment_factor

            adjusted_params_ML = best_params_only_xgb.copy()
            for param in adjusted_params_ML:
                # Exclude 'eta' from the scaling
                if param != 'eta':
                    adjusted_params_ML[param] *= adjustment_factor

            # (c) Execute Hybrid XGBoost & Only XGBoost in parallel
            xgb_thread = Thread(
                target=run_xgb_cross_validate,
                args=(sampled_files, adjusted_params_XGB, results_dict, size)
            )
            ml_thread = Thread(
                target=run_only_xgb_validate,
                args=(sampled_files, adjusted_params_ML, results_dict, size)
            )
            xgb_thread.start()
            ml_thread.start()
            xgb_thread.join()
            ml_thread.join()

            # (d) Hybrid Model (Linear Regression)
            try:
                hybrid_lr_results, _ = lr_cross_validate(sampled_files)
                if hybrid_lr_results:
                    hybrid_lr_rmse_values = [r['rmse'] for r in hybrid_lr_results]
                    results_dict[size].append({
                        'model': 'Hybrid Model(Linear Regression)',
                        'rmse': hybrid_lr_rmse_values
                    })
            except Exception as e:
                print(f"Linear Regression cross_validate error: {e}")

            # (e) Only LR
            try:
                only_lr_results, _ = only_lr_validate(sampled_files)
                if only_lr_results:
                    only_lr_rmse_values = [r['rmse'] for r in only_lr_results]
                    results_dict[size].append({
                        'model': 'Only ML(LR)',
                        'rmse': only_lr_rmse_values
                    })
            except Exception as e:
                print(f"Only LR cross_validate error: {e}")

    # Print final results
    print(results_dict)

    # Save the results to a JSON file
    output_file_path = os.path.join("C:\\Users\\BSL\\Desktop", "results.txt")
    with open(output_file_path, 'w') as outfile:
        json.dump(results_dict, outfile)

    return results_dict


def plot_rmse_results(results_dict, save_path=None):
    """
    1) Iterate through results_dict by size, gather RMSE values of each model.
    2) Normalize (Physics-based as reference) and plot on a log scale.
    """
    # Sort the size keys
    sizes = sorted(results_dict.keys())
    # Example: optionally filter out very small sizes if desired
    # sizes = [s for s in sizes if s >= 10]

    # Lists for storing results
    phys_rmse_mean = []
    phys_rmse_std = []
    xgb_rmse_mean = []
    xgb_rmse_std = []
    lr_rmse_mean = []
    lr_rmse_std = []
    only_ml_rmse_mean = []
    only_ml_rmse_std = []
    only_lr_rmse_mean = []
    only_lr_rmse_std = []

    for size in sizes:
        # Collect RMSE values by model
        phys_values = [
            val
            for result in results_dict[size] if result['model'] == 'Physics-Based'
            for val in result['rmse']
        ]
        xgb_values = [
            val
            for result in results_dict[size] if result['model'] == 'Hybrid Model(XGBoost)'
            for val in result['rmse']
        ]
        lr_values = [
            val
            for result in results_dict[size] if result['model'] == 'Hybrid Model(Linear Regression)'
            for val in result['rmse']
        ]
        only_ml_values = [
            val
            for result in results_dict[size] if result['model'] == 'Only ML(XGBoost)'
            for val in result['rmse']
        ]
        only_lr_values = [
            val
            for result in results_dict[size] if result['model'] == 'Only ML(LR)'
            for val in result['rmse']
        ]

        # Calculate mean and standard deviation
        if phys_values:
            p_mean = np.mean(phys_values)
            p_std = np.std(phys_values)
        else:
            p_mean = 1.0
            p_std = 0.0
        phys_rmse_mean.append(p_mean)
        phys_rmse_std.append(p_std)

        # Helper function to get mean and std
        def get_mean_std(values):
            if values:
                return np.mean(values), np.std(values)
            return (None, None)

        x_mean, x_std = get_mean_std(xgb_values)
        xgb_rmse_mean.append(x_mean)
        xgb_rmse_std.append(x_std)

        l_mean, l_std = get_mean_std(lr_values)
        lr_rmse_mean.append(l_mean)
        lr_rmse_std.append(l_std)

        om_mean, om_std = get_mean_std(only_ml_values)
        only_ml_rmse_mean.append(om_mean)
        only_ml_rmse_std.append(om_std)

        ol_mean, ol_std = get_mean_std(only_lr_values)
        only_lr_rmse_mean.append(ol_mean)
        only_lr_rmse_std.append(ol_std)

    # Normalize values relative to the Physics-based model
    def normalize_vals(vals_mean, vals_std, phys_mean):
        norm_mean = []
        norm_std = []
        for vm, vs, pm in zip(vals_mean, vals_std, phys_mean):
            if vm is not None and pm != 0:
                nm = vm / pm
                # Standard deviation is also scaled by the same ratio
                ns = vs / pm if vs is not None else 0
                norm_mean.append(nm)
                norm_std.append(ns)
            else:
                norm_mean.append(None)
                norm_std.append(None)
        return norm_mean, norm_std

    normalized_xgb_mean, normalized_xgb_std = normalize_vals(xgb_rmse_mean, xgb_rmse_std, phys_rmse_mean)
    normalized_lr_mean, normalized_lr_std = normalize_vals(lr_rmse_mean, lr_rmse_std, phys_rmse_mean)
    normalized_only_ml_mean, normalized_only_ml_std = normalize_vals(only_ml_rmse_mean, only_ml_rmse_std, phys_rmse_mean)
    normalized_only_lr_mean, normalized_only_lr_std = normalize_vals(only_lr_rmse_mean, only_lr_rmse_std, phys_rmse_mean)

    # The Physics-based model is always 1.0 (reference)
    normalized_phys_mean = [1.0] * len(phys_rmse_mean)

    plt.figure(figsize=(6, 5))

    # Physics-Based
    plt.plot(
        sizes, normalized_phys_mean,
        label='Physics-Based', linestyle='--', color='#747678ff'
    )
    # Only ML(LR)
    plt.plot(
        sizes, normalized_only_lr_mean,
        label='Only ML(LR)', marker='o', color='#0073c2ff'
    )
    # Only ML(XGB)
    plt.plot(
        sizes, normalized_only_ml_mean,
        label='Only ML(XGB)', marker='o', color='#efc000ff'
    )
    # Hybrid Model(LR)
    plt.plot(
        sizes, normalized_lr_mean,
        label='Hybrid Model(LR)', marker='o', color='#cd534cff'
    )
    # Hybrid Model(XGB)
    plt.plot(
        sizes, normalized_xgb_mean,
        label='Hybrid Model(XGB)', marker='D', color='#20854eff'
    )

    plt.xlabel('Number of Trips')
    plt.ylabel('Normalized RMSE')
    plt.title('RMSE vs Number of Trips')
    plt.legend()
    plt.grid(False)
    plt.xscale('log')
    plt.xticks(sizes, [str(s) for s in sizes], rotation=45)
    plt.xlim(min(sizes) - 1, max(sizes) + 1)
    plt.tight_layout()

    if save_path:
        save_file = os.path.join(save_path, "rmse_normalized.png")
        plt.savefig(save_file, dpi=300)

    plt.show()
