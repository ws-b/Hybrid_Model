import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# This code contains comments generated using ChatGPT.
class EV6:
    def __init__(self, re_brake=1):
        """
        A simplified vehicle class for EV6 only.
        (mass, load, Ca, Cb, Cc, aux, hvac, idle, eff, re_brake)
        """
        self.mass = 2154.564  # Vehicle mass (kg)
        self.load = 0  # Additional load (kg)

        # The following coefficients are converted from imperial units (lbf, mph) to SI units (N, m/s)
        #   Ca: Air resistance coefficient
        #   Cb: Rolling resistance coefficient
        #   Cc: Gradient resistance coefficient
        self.Ca = 36.158 * 4.44822  # lbf -> N
        self.Cb = 0.29099 * 4.44822 * 2.237  # lbf/mph -> N/(m/s)
        self.Cc = 0.019825 * 4.44822 * (2.237 ** 2)  # lbf/mph^2 -> N/(m/s^2)

        self.aux = 250  # Auxiliary power (W)
        self.hvac = 350  # HVAC power (W)
        self.idle = 0  # Idle power (W)
        self.eff = 0.9  # Overall system efficiency
        self.re_brake = re_brake  # Regenerative braking enabled (1) or disabled (0)


def process_file_power(file, EV):
    """
    Reads a single CSV file, calculates the 'Power_phys' column using EV6-specific parameters,
    and overwrites the original CSV file.

    Parameters:
        file (str): Path to the CSV file.
        EV (EV6): The EV6 vehicle instance used for physical calculations.
    """
    try:
        data = pd.read_csv(file)

        # Constants
        inertia = 0.05  # Rotational inertia factor
        g = 9.18  # Gravitational acceleration (m/s^2)

        # Sort rows by time
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        data = data.sort_values('time').reset_index(drop=True)

        # Extract speed, acceleration, external temperature
        v = data['speed'].to_numpy()  # [m/s]
        a = data['acceleration'].to_numpy()  # [m/s^2]
        ext_temp = data['ext_temp'].to_numpy()

        # Calculate each physical term
        A = EV.Ca * v / EV.eff
        B = EV.Cb * (v ** 2) / EV.eff
        C = EV.Cc * (v ** 3) / EV.eff

        # Avoid excessive growth when acceleration is close to zero
        exp_term = np.exp(0.0411 / np.maximum(np.abs(a), 0.001))

        # Positive and negative acceleration (with possible regenerative braking)
        D_positive = ((1 + inertia) * (EV.mass + EV.load) * a * v) / EV.eff
        D_negative = (((1 + inertia) * (EV.mass + EV.load) * a * v) / exp_term) * EV.eff

        # Decide whether to apply regenerative braking
        D = np.where(a >= 0, D_positive, np.where(EV.re_brake == 1, D_negative, 0))

        # HVAC and auxiliary power
        Eff_hvac = 0.81
        target_temp = 22  # (°C)
        E_hvac = np.abs(target_temp - ext_temp) * EV.hvac * Eff_hvac

        # Include idle power if speed is very low
        E = np.where(v <= 0.5, EV.aux + EV.idle + E_hvac, EV.aux + E_hvac)

        # Handling altitude-based gradient force if 'altitude' column is present
        if 'altitude' in data.columns:
            data['altitude'] = pd.to_numeric(data['altitude'], errors='coerce')
            data['altitude'] = data['altitude'].interpolate(method='linear', limit_direction='both')

            altitude = data['altitude'].to_numpy()
            altitude_diff = np.diff(altitude, prepend=altitude[0])

            time_diff = data['time'].diff().dt.total_seconds().fillna(2).to_numpy()
            distance_diff = v * time_diff

            # Calculate slope
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = np.arctan2(altitude_diff, distance_diff)
                slope = np.where(distance_diff == 0, 0, slope)

            data['slope'] = slope
            F = EV.mass * g * np.sin(slope) * v / EV.eff
        else:
            F = np.zeros_like(v)

        # Final power
        data['Power_phys'] = A + B + C + D + E + F

        # Overwrite the CSV file
        data.to_csv(file, index=False)

    except Exception as e:
        print(f"Error processing file {file}: {e}")


def process_files_power(file_lists):
    """
    For a list of CSV files, apply process_file_power in parallel using a ProcessPoolExecutor.

    Parameters:
        file_lists (list): A list of file paths.
    """
    EV = EV6()  # Instantiate EV6

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_power, file, EV) for file in file_lists]
        for _ in tqdm(futures, desc="Processing", total=len(file_lists)):
            pass  # We don't need to collect results, just ensure the processing finishes

    print('Done')
