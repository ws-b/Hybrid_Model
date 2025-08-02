import os
import pandas as pd
import glob
from config import DATA_PATH, SELECTED_VEHICLE

class DataLoader:
    """
    Handles loading, preprocessing, and sampling of the vehicle trip data.
    """
    def __init__(self):
        self.vehicle_files = self._get_vehicle_files()
        self.full_dataset = self._load_and_combine_data(self.vehicle_files)
        self._generate_features()

    def _get_vehicle_files(self):
        """Finds all CSV files for the selected vehicle."""
        search_path = os.path.join(DATA_PATH, SELECTED_VEHICLE, "*.csv")
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"No data files found for {SELECTED_VEHICLE} at {search_path}")
        return files

    def _load_and_combine_data(self, files):
        """
        Loads all CSVs into a single DataFrame, parsing dates with a specific format
        to prevent warnings and improve performance.
        """
        df_list = []
        for f in files:
            df = pd.read_csv(f)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df_list.append(df)
            
        for i, df in enumerate(df_list):
            df['trip_id'] = i
            
        combined_df = pd.concat(df_list, ignore_index=True)
        
        if combined_df['time'].isnull().any():
            print("Warning: Rows with datetime parsing errors were found and have been dropped.")
            combined_df.dropna(subset=['time'], inplace=True)
            
        return combined_df

    def _generate_features(self):
        """
        Generates the target variable and rolling features based on the paper's methodology.
        """
        # 1. Ground Truth (Target) 설정
        if 'Power_data' not in self.full_dataset.columns:
            raise KeyError("Ground truth column 'Power_data' not found in the loaded data.")
        self.full_dataset['target'] = self.full_dataset['Power_data']

        # 2. Physics-based 모델 예측값 컬럼 표준화
        if 'Power_phys' not in self.full_dataset.columns:
            raise KeyError("Physics model prediction column 'Power_phys' not found in the loaded data.")
        self.full_dataset['physics_prediction'] = self.full_dataset['Power_phys']

        # 3. 롤링 피처(Rolling Features) 생성
        print("Generating rolling features for a 10-second window...")
        window = 5
        
        # Speed
        if 'speed' in self.full_dataset.columns:
            self.full_dataset[f'speed_roll_mean_{window}'] = self.full_dataset.groupby('trip_id')['speed'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            self.full_dataset[f'speed_roll_std_{window}'] = self.full_dataset.groupby('trip_id')['speed'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        else:
            raise KeyError("Column 'speed' not found, cannot generate rolling features.")
        
        # Acceleration
        if 'acceleration' in self.full_dataset.columns:
            self.full_dataset[f'acceleration_roll_mean_{window}'] = self.full_dataset.groupby('trip_id')['acceleration'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            self.full_dataset[f'acceleration_roll_std_{window}'] = self.full_dataset.groupby('trip_id')['acceleration'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # 롤링 피처 생성 시 초반에 발생하는 NaN 값을 0으로 채움
        self.full_dataset.fillna(0, inplace=True)
        print("Feature generation complete.")

    def get_full_dataset(self):
        """Returns the entire dataset for the vehicle."""
        return self.full_dataset

    def sample_data(self, size):
        """
        Samples a specified number of trips from the full dataset.

        Args:
            size (int): The number of trips to sample.

        Returns:
            A DataFrame containing the sampled trips.
        """
        all_trip_ids = self.full_dataset['trip_id'].unique()
        if size > len(all_trip_ids):
            print(f"Warning: Sample size {size} is larger than the total number of trips {len(all_trip_ids)}.")
            sampled_trip_ids = all_trip_ids
        else:
            sampled_trip_ids = pd.Series(all_trip_ids).sample(n=size, random_state=None).values
        
        return self.full_dataset[self.full_dataset['trip_id'].isin(sampled_trip_ids)]

    def get_trip_count(self):
        """Returns the total number of unique trips."""
        return len(self.vehicle_files)
