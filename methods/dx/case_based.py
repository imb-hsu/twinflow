import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, List, Optional, Tuple
import os


class CaseBasedDiagnosis:
    """
    Case-Based Fault Diagnosis implementation.

    Uses previously observed fault cases as reference patterns for diagnosing
    new abnormal behavior. Reference cases are extracted from labeled data files
    and contain observation vectors together with corresponding fault labels,
    affected components, component types, and production areas.
    """

    def __init__(self, reference_data_paths: List[str]):
        """
        Initialize the Case-Based Diagnosis system.

        Args:
            reference_data_paths: List of paths to labeled data files containing reference cases
        """
        self.reference_cases = []
        self.observation_vectors = []
        self._load_reference_cases(reference_data_paths)

    def _load_reference_cases(self, data_paths: List[str]) -> None:
        """
        Load reference cases from labeled data files.

        Args:
            data_paths: List of paths to labeled data files
        """
        for path in data_paths:
            if not os.path.exists(path):
                print(f"Warning: Reference data file not found: {path}")
                continue

            try:
                # Load the data file (assuming CSV format with headers)
                df = pd.read_csv(path)

                # Extract observation vectors (numerical features)
                # Exclude metadata columns like labels, timestamps, etc.
                metadata_cols = [
                    "fault_label",
                    "affected_component",
                    "component_type",
                    "production_area",
                    "timestamp",
                    "scenario_id",
                ]
                feature_cols = [col for col in df.columns if col not in metadata_cols]

                # Store each row as a reference case
                for idx, row in df.iterrows():
                    observation_vector = row[feature_cols].values.astype(float)

                    case = {
                        "observation_vector": observation_vector,
                        "fault_label": row.get("fault_label", "unknown"),
                        "affected_component": row.get("affected_component", "unknown"),
                        "component_type": row.get("component_type", "unknown"),
                        "production_area": row.get("production_area", "unknown"),
                    }

                    self.reference_cases.append(case)
                    self.observation_vectors.append(observation_vector)

                print(f"Loaded {len(df)} reference cases from {path}")

            except Exception as e:
                print(f"Error loading reference cases from {path}: {e}")

        if self.observation_vectors:
            self.observation_vectors = np.array(self.observation_vectors)
            print(f"Total reference cases loaded: {len(self.reference_cases)}")
        else:
            print("Warning: No reference cases loaded")

    def diagnose(self, test_observation: np.ndarray) -> Optional[Dict]:
        """
        Diagnose a test observation by finding the most similar reference case.

        Uses Euclidean distance to measure similarity:
        c* = arg min_{c_r ∈ C} d(x_i, x_r)

        Args:
            test_observation: Observation vector to diagnose (numpy array)

        Returns:
            Dictionary containing diagnosis result with fault_label, affected_component,
            component_type, production_area, and distance to nearest case.
            Returns None if no reference cases are available.
        """
        if len(self.reference_cases) == 0:
            print("Error: No reference cases available for diagnosis")
            return None

        # Ensure test_observation is 2D for sklearn
        test_obs_2d = test_observation.reshape(1, -1)

        # Compute Euclidean distances to all reference cases
        distances = euclidean_distances(test_obs_2d, self.observation_vectors)[0]

        # Find the index of the nearest reference case (minimum distance)
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]

        # Retrieve the most similar case
        nearest_case = self.reference_cases[nearest_idx]

        # Return diagnosis result
        diagnosis = {
            "fault_label": nearest_case["fault_label"],
            "affected_component": nearest_case["affected_component"],
            "component_type": nearest_case["component_type"],
            "production_area": nearest_case["production_area"],
            "distance": min_distance,
            "confidence": self._compute_confidence(distances, nearest_idx),
        }

        return diagnosis

    def _compute_confidence(self, distances: np.ndarray, nearest_idx: int) -> float:
        """
        Compute confidence score for the diagnosis.

        Uses the ratio of the nearest distance to the second nearest distance.
        Higher values indicate more confident diagnosis.

        Args:
            distances: Array of distances to all reference cases
            nearest_idx: Index of the nearest case

        Returns:
            Confidence score between 0 and 1
        """
        if len(distances) < 2:
            return 1.0

        sorted_distances = np.sort(distances)
        nearest_dist = sorted_distances[0]
        second_nearest_dist = sorted_distances[1]

        if second_nearest_dist == 0:
            return 1.0

        # Confidence is higher when nearest is much closer than second nearest
        confidence = 1.0 - (nearest_dist / second_nearest_dist)
        return max(0.0, min(1.0, confidence))

    def batch_diagnose(self, test_observations: np.ndarray) -> List[Optional[Dict]]:
        """
        Diagnose multiple test observations.

        Args:
            test_observations: Array of observation vectors (2D numpy array)

        Returns:
            List of diagnosis results
        """
        results = []
        for observation in test_observations:
            result = self.diagnose(observation)
            results.append(result)
        return results
