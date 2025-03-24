import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from torchdiffeq import odeint, odeint_adjoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime
import time
import random
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("treatment_effect_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

set_seed()

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Define constants
VITAL_SIGNS = ["heart_rate", "sbp", "dbp", "resp_rate", "temperature", "spo2"]
LAB_TESTS = ["wbc", "hgb", "platelet", "sodium", "potassium", "chloride", "bicarbonate", 
             "bun", "creatinine", "glucose", "lactate", "ph", "pao2", "paco2"]
TREATMENTS = ["antibiotic", "vasopressor", "mechanical_ventilation", "fluid_bolus"]
FEATURES = VITAL_SIGNS + LAB_TESTS
N_FEATURES = len(FEATURES)
N_TREATMENTS = len(TREATMENTS)
MAX_TIME = 100  # Maximum time in hours for simulation

# Part 1: Synthetic Data Generation
# --------------------------------

class SyntheticMIMICGenerator:
    """Class for generating synthetic MIMIC-IV-like data."""
    
    def __init__(self, n_patients=1000, time_points_range=(20, 50), 
                 irregular_sampling=True, missing_rate=0.1):
        """
        Initialize the synthetic data generator.
        
        Args:
            n_patients: Number of patients to generate
            time_points_range: Range of time points per patient
            irregular_sampling: Whether to generate irregular time points
            missing_rate: Rate of missing values
        """
        self.n_patients = n_patients
        self.time_points_range = time_points_range
        self.irregular_sampling = irregular_sampling
        self.missing_rate = missing_rate
        logger.info(f"Initialized SyntheticMIMICGenerator with {n_patients} patients")
    
    def _generate_patient_baseline(self):
        """Generate baseline characteristics for a patient."""
        age = np.random.normal(65, 15)
        age = max(18, min(100, age))  # Clip age to reasonable range
        gender = np.random.choice(["M", "F"])
        weight = np.random.normal(70, 15) if gender == "M" else np.random.normal(60, 12)
        weight = max(40, min(150, weight))  # Clip weight
        height = np.random.normal(175, 10) if gender == "M" else np.random.normal(165, 8)
        height = max(150, min(200, height))  # Clip height
        bmi = weight / ((height/100) ** 2)
        return {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "bmi": bmi
        }
    
    def _generate_initial_state(self, baseline):
        """Generate initial clinical state based on patient baseline."""
        # Generate realistic initial vital signs
        heart_rate = np.random.normal(85, 15)  # beats per minute
        sbp = np.random.normal(120, 20)  # systolic blood pressure
        dbp = np.random.normal(80, 10)  # diastolic blood pressure
        resp_rate = np.random.normal(18, 4)  # respirations per minute
        temperature = np.random.normal(37, 0.5)  # Celsius
        spo2 = np.random.normal(96, 2)  # oxygen saturation
        
        # Generate realistic initial lab values
        wbc = np.random.normal(9, 3)  # white blood cell count
        hgb = np.random.normal(12, 2)  # hemoglobin
        platelet = np.random.normal(220, 80)  # platelet count
        sodium = np.random.normal(140, 4)  # sodium
        potassium = np.random.normal(4, 0.5)  # potassium
        chloride = np.random.normal(100, 4)  # chloride
        bicarbonate = np.random.normal(24, 3)  # bicarbonate
        bun = np.random.normal(20, 8)  # blood urea nitrogen
        creatinine = np.random.normal(1, 0.5)  # creatinine
        glucose = np.random.normal(120, 30)  # glucose
        lactate = np.random.normal(2, 1)  # lactate
        ph = np.random.normal(7.4, 0.05)  # pH
        pao2 = np.random.normal(90, 15)  # PaO2
        paco2 = np.random.normal(40, 5)  # PaCO2
        
        # Create initial state dictionary
        initial_state = {
            "heart_rate": heart_rate,
            "sbp": sbp,
            "dbp": dbp,
            "resp_rate": resp_rate,
            "temperature": temperature,
            "spo2": spo2,
            "wbc": wbc,
            "hgb": hgb,
            "platelet": platelet,
            "sodium": sodium,
            "potassium": potassium,
            "chloride": chloride,
            "bicarbonate": bicarbonate,
            "bun": bun,
            "creatinine": creatinine,
            "glucose": glucose,
            "lactate": lactate,
            "ph": ph,
            "pao2": pao2,
            "paco2": paco2
        }
        
        # Adjust based on patient characteristics
        if baseline["age"] > 80:
            initial_state["heart_rate"] += np.random.normal(5, 2)
            initial_state["sbp"] += np.random.normal(10, 5)
            initial_state["creatinine"] += np.random.normal(0.3, 0.1)
        
        if baseline["bmi"] > 30:
            initial_state["glucose"] += np.random.normal(20, 10)
            initial_state["resp_rate"] += np.random.normal(2, 1)
        
        return initial_state
    
    def _generate_latent_severity(self):
        """Generate a latent disease severity factor for the patient."""
        # Higher values indicate more severe disease
        return np.random.gamma(shape=2, scale=1.5)
    
    def _generate_treatment_pattern(self, n_timepoints, severity):
        """Generate treatment patterns based on disease severity."""
        treatments = {treatment: np.zeros(n_timepoints) for treatment in TREATMENTS}
        
        # Probability of receiving each treatment increases with severity
        p_antibiotic = min(0.8, 0.2 + severity * 0.3)
        p_vasopressor = min(0.7, 0.1 + severity * 0.3)
        p_vent = min(0.6, 0.05 + severity * 0.25)
        p_fluid = min(0.9, 0.3 + severity * 0.3)
        
        # Generate treatment patterns
        for t in range(n_timepoints):
            # Treatments may start at different times
            if t > np.random.randint(0, max(1, int(n_timepoints/4))):
                if np.random.random() < p_antibiotic:
                    # Once antibiotics start, they usually continue
                    treatments["antibiotic"][t:] = 1
                    break
                    
        for t in range(n_timepoints):
            if t > np.random.randint(0, max(1, int(n_timepoints/3))):
                if np.random.random() < p_vasopressor:
                    # Vasopressors might be given intermittently
                    duration = np.random.randint(1, max(2, int(n_timepoints/3)))
                    end = min(t + duration, n_timepoints)
                    treatments["vasopressor"][t:end] = 1
                    
        for t in range(n_timepoints):
            if t > np.random.randint(0, max(1, int(n_timepoints/3))):
                if np.random.random() < p_vent:
                    # Ventilation usually continues once started
                    treatments["mechanical_ventilation"][t:] = 1
                    break
                    
        for t in range(n_timepoints):
            # Fluid boluses are more episodic
            if np.random.random() < p_fluid / 5:  # Lower chance per timepoint
                treatments["fluid_bolus"][t] = 1
        
        return treatments
    
    def _evolve_patient_state(self, initial_state, treatments, severity, timepoints):
        """
        Evolve patient state over time based on initial condition, treatments, and severity.
        
        This uses a simple rule-based approach for data generation. In a real system,
        we would learn these dynamics from data.
        """
        n_timepoints = len(timepoints)
        states = {feature: np.zeros(n_timepoints) for feature in FEATURES}
        
        # Set initial state
        for feature in FEATURES:
            states[feature][0] = initial_state[feature]
            
        # Define some basic dynamics
        for t in range(1, n_timepoints):
            dt = timepoints[t] - timepoints[t-1]
            
            # Natural disease progression (based on severity)
            hr_change = (severity * 0.5 - 0.1) * dt
            sbp_change = (-severity * 1.0 + 0.1) * dt
            spo2_change = (-severity * 0.3 + 0.05) * dt
            lactate_change = (severity * 0.1 - 0.02) * dt
            
            # Treatment effects
            if treatments["antibiotic"][t-1] > 0:
                # Antibiotics help reduce WBC over time
                wbc_effect = -0.2 * dt
                hr_effect = -0.1 * dt
                lactate_effect = -0.05 * dt
            else:
                wbc_effect = 0
                hr_effect = 0
                lactate_effect = 0
                
            if treatments["vasopressor"][t-1] > 0:
                # Vasopressors increase blood pressure
                sbp_effect = 2.0 * dt
                hr_effect_vaso = 0.5 * dt
            else:
                sbp_effect = 0
                hr_effect_vaso = 0
                
            if treatments["mechanical_ventilation"][t-1] > 0:
                # Ventilation improves oxygenation
                spo2_effect = 0.3 * dt
                pao2_effect = 1.0 * dt
                resp_effect = -0.2 * dt
            else:
                spo2_effect = 0
                pao2_effect = 0
                resp_effect = 0
                
            if treatments["fluid_bolus"][t-1] > 0:
                # Fluid bolus temporarily increases BP but might dilute Hgb
                sbp_effect_fluid = 1.5 * dt
                hgb_effect = -0.2 * dt
            else:
                sbp_effect_fluid = 0
                hgb_effect = 0
            
            # Update state with combined effects and some random noise
            states["heart_rate"][t] = states["heart_rate"][t-1] + hr_change + hr_effect + hr_effect_vaso + np.random.normal(0, 1)
            states["sbp"][t] = states["sbp"][t-1] + sbp_change + sbp_effect + sbp_effect_fluid + np.random.normal(0, 2)
            states["dbp"][t] = states["dbp"][t-1] + sbp_change*0.5 + sbp_effect*0.5 + sbp_effect_fluid*0.5 + np.random.normal(0, 1)
            states["resp_rate"][t] = states["resp_rate"][t-1] + severity*0.1*dt + resp_effect + np.random.normal(0, 0.5)
            states["temperature"][t] = states["temperature"][t-1] + (severity*0.05 - 0.01)*dt + np.random.normal(0, 0.1)
            states["spo2"][t] = states["spo2"][t-1] + spo2_change + spo2_effect + np.random.normal(0, 0.5)
            
            # Lab values change more slowly and are measured less frequently
            states["wbc"][t] = states["wbc"][t-1] + (severity*0.2 - 0.05)*dt + wbc_effect + np.random.normal(0, 0.2)
            states["hgb"][t] = states["hgb"][t-1] + (-0.01)*dt + hgb_effect + np.random.normal(0, 0.1)
            states["platelet"][t] = states["platelet"][t-1] + (-severity*1.0 + 0.5)*dt + np.random.normal(0, 2)
            states["sodium"][t] = states["sodium"][t-1] + np.random.normal(0, 0.3)
            states["potassium"][t] = states["potassium"][t-1] + (severity*0.02 - 0.01)*dt + np.random.normal(0, 0.1)
            states["chloride"][t] = states["chloride"][t-1] + np.random.normal(0, 0.3)
            states["bicarbonate"][t] = states["bicarbonate"][t-1] + (-severity*0.1 + 0.02)*dt + np.random.normal(0, 0.2)
            states["bun"][t] = states["bun"][t-1] + (severity*0.2 - 0.05)*dt + np.random.normal(0, 0.3)
            states["creatinine"][t] = states["creatinine"][t-1] + (severity*0.03 - 0.01)*dt + np.random.normal(0, 0.05)
            states["glucose"][t] = states["glucose"][t-1] + (severity*0.5 - 0.2)*dt + np.random.normal(0, 2)
            states["lactate"][t] = states["lactate"][t-1] + lactate_change + lactate_effect + np.random.normal(0, 0.1)
            states["ph"][t] = states["ph"][t-1] + (-severity*0.01 + 0.002)*dt + np.random.normal(0, 0.01)
            states["pao2"][t] = states["pao2"][t-1] + (-severity*0.5 + 0.1)*dt + pao2_effect + np.random.normal(0, 1)
            states["paco2"][t] = states["paco2"][t-1] + (severity*0.2 - 0.05)*dt + np.random.normal(0, 0.5)
            
            # Apply some basic physiological constraints
            states["heart_rate"][t] = max(40, min(200, states["heart_rate"][t]))
            states["sbp"][t] = max(50, min(220, states["sbp"][t]))
            states["dbp"][t] = max(20, min(120, states["dbp"][t]))
            states["resp_rate"][t] = max(5, min(60, states["resp_rate"][t]))
            states["temperature"][t] = max(35, min(42, states["temperature"][t]))
            states["spo2"][t] = max(60, min(100, states["spo2"][t]))
            states["wbc"][t] = max(0.5, min(50, states["wbc"][t]))
            states["hgb"][t] = max(3, min(20, states["hgb"][t]))
            states["platelet"][t] = max(10, min(700, states["platelet"][t]))
            states["sodium"][t] = max(120, min(160, states["sodium"][t]))
            states["potassium"][t] = max(2, min(8, states["potassium"][t]))
            states["chloride"][t] = max(90, min(120, states["chloride"][t]))
            states["bicarbonate"][t] = max(10, min(40, states["bicarbonate"][t]))
            states["bun"][t] = max(5, min(150, states["bun"][t]))
            states["creatinine"][t] = max(0.3, min(15, states["creatinine"][t]))
            states["glucose"][t] = max(40, min(500, states["glucose"][t]))
            states["lactate"][t] = max(0.5, min(20, states["lactate"][t]))
            states["ph"][t] = max(6.8, min(7.8, states["ph"][t]))
            states["pao2"][t] = max(35, min(300, states["pao2"][t]))
            states["paco2"][t] = max(20, min(120, states["paco2"][t]))
        
        return states
    
    def _introduce_missing_values(self, data, timepoints):
        """Introduce realistic missing values in the data."""
        # Labs are measured less frequently than vitals
        for feature in LAB_TESTS:
            for t in range(len(timepoints)):
                # Higher chance of missing lab values
                if np.random.random() < self.missing_rate * 2:
                    data[feature][t] = np.nan
                    
        # Vitals may have some missing values too
        for feature in VITAL_SIGNS:
            for t in range(len(timepoints)):
                if np.random.random() < self.missing_rate:
                    data[feature][t] = np.nan
                    
        return data
    
    def _convert_to_dataframe(self, patient_id, timepoints, states, treatments, baseline):
        """Convert patient data to DataFrame format similar to MIMIC."""
        data = []
        
        for t in range(len(timepoints)):
            row = {
                "patient_id": patient_id,
                "time": timepoints[t]
            }
            
            # Add baseline characteristics
            for key, value in baseline.items():
                row[key] = value
                
            # Add clinical measurements
            for feature in FEATURES:
                row[feature] = states[feature][t]
                
            # Add treatments
            for treatment in TREATMENTS:
                row[treatment] = treatments[treatment][t]
                
            data.append(row)
            
        return pd.DataFrame(data)
    
    def generate(self):
        """Generate synthetic MIMIC-IV-like data for multiple patients."""
        all_patients_data = []
        
        for i in range(self.n_patients):
            if i % 100 == 0:
                logger.info(f"Generating data for patient {i+1}/{self.n_patients}")
                
            # Generate patient baseline
            patient_id = f"P{i+1:06d}"
            baseline = self._generate_patient_baseline()
            initial_state = self._generate_initial_state(baseline)
            severity = self._generate_latent_severity()
            
            # Generate time points
            n_timepoints = np.random.randint(self.time_points_range[0], self.time_points_range[1])
            
            if self.irregular_sampling:
                # Generate irregular time points (more realistic)
                base_times = np.sort(np.random.uniform(0, MAX_TIME, n_timepoints))
                # Add more frequent measurements at the beginning
                early_times = np.sort(np.random.uniform(0, MAX_TIME/5, int(n_timepoints/3)))
                timepoints = np.sort(np.concatenate([base_times, early_times]))
                timepoints = np.unique(timepoints)
            else:
                # Regular time points
                timepoints = np.linspace(0, MAX_TIME, n_timepoints)
            
            # Generate treatments
            treatments = self._generate_treatment_pattern(len(timepoints), severity)
            
            # Evolve patient state over time
            states = self._evolve_patient_state(initial_state, treatments, severity, timepoints)
            
            # Introduce missing values
            states = self._introduce_missing_values(states, timepoints)
            
            # Convert to DataFrame
            patient_df = self._convert_to_dataframe(patient_id, timepoints, states, treatments, baseline)
            patient_df["severity"] = severity  # Add true severity (for evaluation only)
            
            all_patients_data.append(patient_df)
        
        # Combine all patient data
        combined_df = pd.concat(all_patients_data, ignore_index=True)
        logger.info(f"Generated data for {self.n_patients} patients with {len(combined_df)} total observations")
        
        return combined_df


# Part 2: Data Preprocessing
# --------------------------

class ClinicalDataProcessor:
    """Class for preprocessing clinical time series data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.feature_scaler = StandardScaler()
        self.preprocessed = False
        logger.info("Initialized ClinicalDataProcessor")
    
    def preprocess(self, data: pd.DataFrame, split=True, test_size=0.2):
        """
        Preprocess the clinical data.
        
        Args:
            data: DataFrame with clinical data
            split: Whether to split into train/test sets
            test_size: Size of test set if splitting
            
        Returns:
            Preprocessed data and optionally train/test splits
        """
        logger.info("Preprocessing clinical data...")
        
        # Sort data by patient and time
        data = data.sort_values(["patient_id", "time"])
        
        # Fill forward to handle missing values within each patient's time series
        data_filled = data.copy()
        for patient in data["patient_id"].unique():
            patient_mask = data["patient_id"] == patient
            # Forward fill within each patient
            data_filled.loc[patient_mask, FEATURES] = data.loc[patient_mask, FEATURES].ffill()
            
        # For any remaining missing values (e.g., at the start), use backward fill
        data_filled[FEATURES] = data_filled[FEATURES].bfill()
        
        # If still missing, fill with column median
        for col in FEATURES:
            median_val = data_filled[col].median()
            data_filled[col] = data_filled[col].fillna(median_val)
        
        # Scale the features
        scaled_features = self.feature_scaler.fit_transform(data_filled[FEATURES])
        data_filled[FEATURES] = scaled_features
        
        self.preprocessed = True
        logger.info("Data preprocessing complete")
        
        if split:
            # Split by patient to avoid data leakage
            patients = data_filled["patient_id"].unique()
            train_patients, test_patients = train_test_split(patients, test_size=test_size, random_state=42)
            
            train_data = data_filled[data_filled["patient_id"].isin(train_patients)]
            test_data = data_filled[data_filled["patient_id"].isin(test_patients)]
            
            logger.info(f"Data split into {len(train_data)} train and {len(test_data)} test samples")
            return train_data, test_data, patients
        else:
            return data_filled
    
    def prepare_patient_tensors(self, data, patients=None):
        """
        Convert preprocessed data to tensors for each patient.
        
        Args:
            data: Preprocessed data
            patients: List of patients to include (optional)
            
        Returns:
            Dictionary with patient data as tensors
        """
        if not self.preprocessed:
            raise ValueError("Data must be preprocessed first using the preprocess method")
        
        if patients is None:
            patients = data["patient_id"].unique()
            
        patient_tensors = {}
        
        for patient in patients:
            patient_data = data[data["patient_id"] == patient].sort_values("time")
            
            # Extract time points
            times = torch.tensor(patient_data["time"].values, dtype=torch.float32)
            
            # Extract features and treatments
            features = torch.tensor(patient_data[FEATURES].values, dtype=torch.float32)
            treatments = torch.tensor(patient_data[TREATMENTS].values, dtype=torch.float32)
            
            # Store data for this patient
            patient_tensors[patient] = {
                "times": times,
                "features": features,
                "treatments": treatments
            }
            
            # Add optional metadata
            if "severity" in patient_data.columns:
                severity = patient_data["severity"].iloc[0]  # Same for all time points
                patient_tensors[patient]["severity"] = severity
        
        logger.info(f"Prepared tensor data for {len(patient_tensors)} patients")
        return patient_tensors


# Part 3: Neural ODE Models
# -------------------------

class ODEFunc(nn.Module):
    """ODE function for modeling patient dynamics."""
    
    def __init__(self, feature_dim, hidden_dim=64, treatment_dim=None):
        """
        Initialize the ODE function.
        
        Args:
            feature_dim: Dimension of clinical features
            hidden_dim: Hidden dimension of the neural network
            treatment_dim: Dimension of treatment variables (if any)
        """
        super(ODEFunc, self).__init__()
        self.feature_dim = feature_dim
        self.treatment_dim = treatment_dim
        
        # Input dimension includes features and treatments if provided
        input_dim = feature_dim
        if treatment_dim is not None:
            input_dim += treatment_dim
        
        # Neural network for dynamics
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize weights for stable dynamics
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, x):
        """
        Compute the derivative of the system at time t.
        
        Args:
            t: Time point
            x: State vector (features, or features+treatments)
            
        Returns:
            dx/dt: Derivative of the state
        """
        # Extract features and treatments if needed
        if self.treatment_dim is not None:
            features = x[:, :self.feature_dim]
            treatments = x[:, self.feature_dim:]
            # Concatenate features and treatments
            state = torch.cat([features, treatments], dim=1)
        else:
            state = x
        
        # Compute derivative
        dxdt = self.net(state)
        
        # If we have treatments, need to ensure their derivatives are zero
        # (they're controlled externally)
        if self.treatment_dim is not None:
            dx_features = dxdt
            dx_treatments = torch.zeros_like(treatments)
            dxdt = torch.cat([dx_features, dx_treatments], dim=1)
        
        return dxdt


class NeuralODEModel(nn.Module):
    """Neural ODE model for patient dynamics."""
    
    def __init__(self, feature_dim, hidden_dim=64, use_adjoint=True):
        """
        Initialize the Neural ODE model.
        
        Args:
            feature_dim: Dimension of clinical features
            hidden_dim: Hidden dimension
            use_adjoint: Whether to use the adjoint method for backward
        """
        super(NeuralODEModel, self).__init__()
        self.feature_dim = feature_dim
        self.ode_func = ODEFunc(feature_dim, hidden_dim)
        self.use_adjoint = use_adjoint
        logger.info(f"Initialized NeuralODEModel with feature_dim={feature_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, x0, ts):
        """
        Forward pass through the ODE.
        
        Args:
            x0: Initial state
            ts: Time points to evaluate at
            
        Returns:
            Predicted states at requested time points
        """
        if self.use_adjoint:
            pred_x = odeint_adjoint(self.ode_func, x0, ts, method='dopri5')
        else:
            pred_x = odeint(self.ode_func, x0, ts, method='dopri5')
        
        return pred_x


class CounterfactualODEModel(nn.Module):
    """Neural ODE model for counterfactual treatment scenarios."""
    
    def __init__(self, feature_dim, treatment_dim, hidden_dim=64, use_adjoint=True):
        """
        Initialize the Counterfactual ODE model.
        
        Args:
            feature_dim: Dimension of clinical features
            treatment_dim: Dimension of treatment variables
            hidden_dim: Hidden dimension
            use_adjoint: Whether to use the adjoint method
        """
        super(CounterfactualODEModel, self).__init__()
        self.feature_dim = feature_dim
        self.treatment_dim = treatment_dim
        self.ode_func = ODEFunc(feature_dim, hidden_dim, treatment_dim)
        self.use_adjoint = use_adjoint
        logger.info(f"Initialized CounterfactualODEModel with feature_dim={feature_dim}, "
                  f"treatment_dim={treatment_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, x0, treatments, ts):
        """
        Forward pass with treatments.
        
        Args:
            x0: Initial state (features only)
            treatments: Treatment values at each time point
            ts: Time points to evaluate at
            
        Returns:
            Predicted states at requested time points
        """
        # Combine initial state with initial treatments
        x0_combined = torch.cat([x0, treatments[0:1]], dim=1)
        
        # Define a function that interpolates treatments at arbitrary times
        def get_treatment_at_time(t):
            """Interpolate treatment at time t."""
            # Find the two closest time points
            t_val = t.item()
            times = ts.detach().cpu().numpy()
            
            # Find the closest indices
            idx = np.searchsorted(times, t_val)
            if idx == 0:
                return treatments[0]
            elif idx == len(times):
                return treatments[-1]
            else:
                # Linear interpolation
                t0, t1 = times[idx-1], times[idx]
                w = (t_val - t0) / (t1 - t0)
                interp_treatment = (1 - w) * treatments[idx-1] + w * treatments[idx]
                return interp_treatment
        
        # Custom ODE function that includes interpolated treatments
        def augmented_dynamics(t, state):
            # Get treatment at this time
            treatment = get_treatment_at_time(t)
            # Combine state with treatment
            augmented_state = torch.cat([state[:, :self.feature_dim], treatment.unsqueeze(0)], dim=1)
            # Get derivative from our model
            d_augmented = self.ode_func(t, augmented_state)
            # Return only the feature derivatives
            return d_augmented[:, :self.feature_dim]
        
        # Integrate the ODE
        if self.use_adjoint:
            pred_x = odeint_adjoint(augmented_dynamics, x0, ts, method='dopri5')
        else:
            pred_x = odeint(augmented_dynamics, x0, ts, method='dopri5')
        
        return pred_x
    
    def simulate_counterfactual(self, x0, factual_treatments, counterfactual_treatments, ts):
        """
        Simulate both factual and counterfactual scenarios.
        
        Args:
            x0: Initial state
            factual_treatments: Actual treatments given
            counterfactual_treatments: Alternative treatment scenario
            ts: Time points to evaluate at
            
        Returns:
            Factual and counterfactual trajectories
        """
        # Run forward with factual treatments
        factual_trajectory = self.forward(x0, factual_treatments, ts)
        
        # Run forward with counterfactual treatments
        counterfactual_trajectory = self.forward(x0, counterfactual_treatments, ts)
        
        return factual_trajectory, counterfactual_trajectory


# Part 4: Training and Evaluation
# -------------------------------

class ClinicalTrajectoriesDataset(Dataset):
    """Dataset for clinical trajectories."""
    
    def __init__(self, patient_tensors, window_size=10):
        """
        Initialize the dataset.
        
        Args:
            patient_tensors: Dictionary of patient tensors
            window_size: Size of time windows to use
        """
        self.patient_tensors = patient_tensors
        self.window_size = window_size
        self.samples = self._create_samples()
        logger.info(f"Created dataset with {len(self.samples)} samples")
    
    def _create_samples(self):
        """Create training samples from patient trajectories."""
        samples = []
        
        for patient_id, data in self.patient_tensors.items():
            times = data["times"]
            features = data["features"]
            treatments = data["treatments"]
            
            # Skip patients with too few observations
            if len(times) < self.window_size + 1:
                continue
            
            # Create windows of observations
            for i in range(len(times) - self.window_size):
                window_times = times[i:i+self.window_size]
                window_features = features[i:i+self.window_size]
                window_treatments = treatments[i:i+self.window_size]
                next_features = features[i+self.window_size]
                next_time = times[i+self.window_size]
                
                sample = {
                    "patient_id": patient_id,
                    "window_times": window_times,
                    "window_features": window_features,
                    "window_treatments": window_treatments,
                    "next_features": next_features,
                    "next_time": next_time
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index."""
        return self.samples[idx]


class TreatmentEffectTrainer:
    """Trainer for treatment effect models."""
    
    def __init__(self, feature_dim, treatment_dim, hidden_dim=64, 
                 lr=1e-3, weight_decay=1e-4, use_adjoint=True):
        """
        Initialize the trainer.
        
        Args:
            feature_dim: Dimension of clinical features
            treatment_dim: Dimension of treatment variables
            hidden_dim: Hidden dimension for neural networks
            lr: Learning rate
            weight_decay: Weight decay for regularization
            use_adjoint: Whether to use adjoint method for ODEs
        """
        self.model = CounterfactualODEModel(
            feature_dim=feature_dim,
            treatment_dim=treatment_dim,
            hidden_dim=hidden_dim,
            use_adjoint=use_adjoint
        ).to(device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        self.criterion = nn.MSELoss()
        logger.info(f"Initialized TreatmentEffectTrainer with feature_dim={feature_dim}, "
                  f"treatment_dim={treatment_dim}, hidden_dim={hidden_dim}, lr={lr}")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            # Get data
            window_times = batch["window_times"].to(device)
            window_features = batch["window_features"].to(device)
            window_treatments = batch["window_treatments"].to(device)
            next_features = batch["next_features"].to(device)
            next_time = batch["next_time"].to(device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass for the window
            # Use the first feature vector as initial state
            x0 = window_features[:, 0]
            
            # Concatenate next time to window times for prediction
            pred_times = torch.cat([window_times[:, 0].unsqueeze(1), next_time.unsqueeze(1)], dim=1)
            pred_times = pred_times.transpose(0, 1)  # Shape: [2, batch_size]
            
            # Predict next state
            trajectories = self.model(x0, window_treatments, pred_times)
            predicted_next = trajectories[-1]  # Last time point
            
            # Compute loss
            loss = self.criterion(predicted_next, next_features)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Batch {i+1}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader with validation data
            
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                window_times = batch["window_times"].to(device)
                window_features = batch["window_features"].to(device)
                window_treatments = batch["window_treatments"].to(device)
                next_features = batch["next_features"].to(device)
                next_time = batch["next_time"].to(device)
                
                # Forward pass for the window
                x0 = window_features[:, 0]
                
                # Concatenate next time to window times for prediction
                pred_times = torch.cat([window_times[:, 0].unsqueeze(1), next_time.unsqueeze(1)], dim=1)
                pred_times = pred_times.transpose(0, 1)  # Shape: [2, batch_size]
                
                # Predict next state
                trajectories = self.model(x0, window_treatments, pred_times)
                predicted_next = trajectories[-1]  # Last time point
                
                # Compute loss
                loss = self.criterion(predicted_next, next_features)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader with training data
            val_loader: DataLoader with validation data
            num_epochs: Number of epochs to train for
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Time: {epoch_time:.2f}s")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pt")
                logger.info("Saved best model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete")
        # Load best model
        self.model.load_state_dict(torch.load("best_model.pt"))
        return history
    
    def predict_counterfactual(self, x0, factual_treatments, counterfactual_treatments, times):
        """
        Predict counterfactual trajectories.
        
        Args:
            x0: Initial state
            factual_treatments: Actual treatments
            counterfactual_treatments: Alternative treatments
            times: Time points to evaluate at
            
        Returns:
            Factual and counterfactual trajectories
        """
        self.model.eval()
        with torch.no_grad():
            factual, counterfactual = self.model.simulate_counterfactual(
                x0, factual_treatments, counterfactual_treatments, times
            )
        return factual, counterfactual


# Part 5: Visualization and Analysis
# ---------------------------------

class TreatmentEffectVisualizer:
    """Class for visualizing treatment effects."""
    
    def __init__(self, feature_names, treatment_names, feature_scaler=None):
        """
        Initialize the visualizer.
        
        Args:
            feature_names: Names of clinical features
            treatment_names: Names of treatments
            feature_scaler: Scaler used to normalize features
        """
        self.feature_names = feature_names
        self.treatment_names = treatment_names
        self.feature_scaler = feature_scaler
        plt.style.use('seaborn-v0_8-whitegrid')
        logger.info("Initialized TreatmentEffectVisualizer")
    
    def plot_training_history(self, history):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(history["train_loss"], label="Train Loss")
        ax.plot(history["val_loss"], label="Validation Loss")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training History")
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def inverse_transform_features(self, features):
        """
        Inverse transform normalized features.
        
        Args:
            features: Normalized features
            
        Returns:
            Features in original scale
        """
        if self.feature_scaler is not None:
            # Handle different input formats
            if isinstance(features, torch.Tensor):
                features_np = features.cpu().numpy()
                # Handle different tensor shapes
                if features_np.ndim == 3:  # [time, batch, features]
                    orig_shape = features_np.shape
                    features_np = features_np.reshape(-1, orig_shape[-1])
                    features_np = self.feature_scaler.inverse_transform(features_np)
                    features_np = features_np.reshape(orig_shape)
                else:  # [batch, features] or similar
                    features_np = self.feature_scaler.inverse_transform(features_np)
                return features_np
            else:
                return self.feature_scaler.inverse_transform(features)
        else:
            if isinstance(features, torch.Tensor):
                return features.cpu().numpy()
            else:
                return features
    
    def plot_counterfactual_trajectories(self, times, factual, counterfactual, 
                                        treatment_times=None, features_to_plot=None):
        """
        Plot counterfactual trajectories.
        
        Args:
            times: Time points
            factual: Factual trajectory
            counterfactual: Counterfactual trajectory
            treatment_times: Times when treatments were changed
            features_to_plot: Which features to plot (if None, plot all)
            
        Returns:
            Figure object
        """
        if isinstance(times, torch.Tensor):
            times = times.cpu().numpy()
            
        # Convert to numpy if tensors
        factual_np = self.inverse_transform_features(factual)
        counterfactual_np = self.inverse_transform_features(counterfactual)
        
        # Determine which features to plot
        if features_to_plot is None:
            # Plot all features
            features_to_plot = list(range(len(self.feature_names)))
        elif isinstance(features_to_plot, list) and all(isinstance(f, str) for f in features_to_plot):
            # Convert feature names to indices
            features_to_plot = [self.feature_names.index(f) for f in features_to_plot]
        
        n_features = len(features_to_plot)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, feature_idx in enumerate(features_to_plot):
            ax = axes[i]
            feature_name = self.feature_names[feature_idx]
            
            # Plot factual and counterfactual
            ax.plot(times, factual_np[:, 0, feature_idx], 'b-', label='Factual')
            ax.plot(times, counterfactual_np[:, 0, feature_idx], 'r--', label='Counterfactual')
            
            # Highlight treatment times if provided
            if treatment_times is not None:
                for t in treatment_times:
                    ax.axvline(x=t, color='g', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(feature_name)
            ax.set_title(f'{feature_name} Over Time')
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_treatment_delay_effects(self, times, trajectories, treatment_times, 
                                    feature_idx, treatment_name):
        """
        Plot the effect of treatment timing.
        
        Args:
            times: Time points
            trajectories: List of trajectories with different treatment times
            treatment_times: List of times when treatment was started
            feature_idx: Feature index to plot
            treatment_name: Name of the treatment
            
        Returns:
            Figure object
        """
        if isinstance(times, torch.Tensor):
            times = times.cpu().numpy()
            
        # Convert trajectories to numpy
        trajectories_np = [self.inverse_transform_features(traj) for traj in trajectories]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
        
        feature_name = self.feature_names[feature_idx]
        
        for i, (traj, t_time) in enumerate(zip(trajectories_np, treatment_times)):
            label = f"{treatment_name} at t={t_time:.1f}h"
            ax.plot(times, traj[:, 0, feature_idx], color=colors[i], label=label)
            # Mark treatment start time
            ax.axvline(x=t_time, color=colors[i], linestyle=':', alpha=0.5)
            
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(feature_name)
        ax.set_title(f'Effect of {treatment_name} Timing on {feature_name}')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_heatmap_of_treatment_timing(self, feature_idx, treatment_idx, times, 
                                          treatment_timings, outcome_values):
        """
        Create a heatmap showing effect of treatment timing.
        
        Args:
            feature_idx: Feature index to visualize
            treatment_idx: Treatment index
            times: Evaluation time points
            treatment_timings: Different treatment start times
            outcome_values: Matrix of outcome values [time, treatment_timing]
            
        Returns:
            Figure object
        """
        feature_name = self.feature_names[feature_idx]
        treatment_name = self.treatment_names[treatment_idx]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(
            outcome_values, 
            aspect='auto', 
            origin='lower',
            extent=[times[0], times[-1], treatment_timings[0], treatment_timings[-1]],
            cmap='plasma'
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(feature_name)
        
        # Set labels
        ax.set_xlabel('Evaluation Time (hours)')
        ax.set_ylabel(f'{treatment_name} Start Time (hours)')
        ax.set_title(f'Effect of {treatment_name} Timing on {feature_name}')
        
        # Add contour lines
        contour = ax.contour(
            np.linspace(times[0], times[-1], outcome_values.shape[1]),
            np.linspace(treatment_timings[0], treatment_timings[-1], outcome_values.shape[0]),
            outcome_values,
            colors='white',
            alpha=0.6
        )
        ax.clabel(contour, inline=True, fontsize=8)
        
        plt.tight_layout()
        return fig


# Part 6: Main Execution Pipeline
# ------------------------------

def run_experiment(n_patients=500, window_size=10, batch_size=32, num_epochs=30):
    """
    Run the full experiment pipeline.
    
    Args:
        n_patients: Number of synthetic patients
        window_size: Window size for training
        batch_size: Batch size for training
        num_epochs: Number of training epochs
    """
    logger.info(f"Starting experiment with {n_patients} patients")
    
    # 1. Generate synthetic data
    logger.info("Generating synthetic data...")
    data_generator = SyntheticMIMICGenerator(n_patients=n_patients)
    mimic_data = data_generator.generate()
    
    # Save data to CSV
    mimic_data.to_csv("synthetic_mimic_data.csv", index=False)
    logger.info(f"Saved synthetic data with {len(mimic_data)} rows")
    
    # 2. Preprocess data
    logger.info("Preprocessing data...")
    data_processor = ClinicalDataProcessor()
    train_data, test_data, patients = data_processor.preprocess(mimic_data, split=True)
    
    # 3. Prepare patient tensors
    logger.info("Preparing patient tensors...")
    train_patients = train_data["patient_id"].unique()
    test_patients = test_data["patient_id"].unique()
    
    train_tensors = data_processor.prepare_patient_tensors(train_data, train_patients)
    test_tensors = data_processor.prepare_patient_tensors(test_data, test_patients)
    
    # 4. Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_dataset = ClinicalTrajectoriesDataset(train_tensors, window_size=window_size)
    test_dataset = ClinicalTrajectoriesDataset(test_tensors, window_size=window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Create and train the model
    logger.info("Creating and training the model...")
    trainer = TreatmentEffectTrainer(
        feature_dim=N_FEATURES,
        treatment_dim=N_TREATMENTS,
        hidden_dim=64,
        lr=1e-3
    )
    
    history = trainer.train(train_loader, test_loader, num_epochs=num_epochs)
    
    # 6. Visualize training results
    logger.info("Visualizing results...")
    visualizer = TreatmentEffectVisualizer(
        feature_names=FEATURES,
        treatment_names=TREATMENTS,
        feature_scaler=data_processor.feature_scaler
    )
    
    training_fig = visualizer.plot_training_history(history)
    training_fig.savefig("training_history.png")
    
    # 7. Generate counterfactual predictions for a test patient
    logger.info("Generating counterfactual predictions...")
    # Select a test patient
    test_patient_id = test_patients[0]
    patient_data = test_tensors[test_patient_id]
    
    # Get initial state and times
    x0 = patient_data["features"][0:1].to(device)  # Initial state
    times = patient_data["times"].to(device)
    factual_treatments = patient_data["treatments"].to(device)
    
    # Create a counterfactual scenario: delay a treatment
    treatment_idx = 2  # mechanical_ventilation
    counterfactual_treatments = factual_treatments.clone()
    
    # Find when mechanical ventilation was started in the factual scenario
    vent_start_idx = 0
    for i in range(len(factual_treatments)):
        if factual_treatments[i, treatment_idx] > 0:
            vent_start_idx = i
            break
    
    # Delay ventilation by a few time steps in counterfactual
    delay_steps = 5
    if vent_start_idx + delay_steps < len(counterfactual_treatments):
        counterfactual_treatments[vent_start_idx:vent_start_idx+delay_steps, treatment_idx] = 0
    
    # Generate predictions
    factual_traj, counterfactual_traj = trainer.predict_counterfactual(
        x0, factual_treatments, counterfactual_treatments, times
    )
    
    # Convert treatment start indices to times
    treatment_times = [times[vent_start_idx].item(), times[vent_start_idx+delay_steps].item()]
    
    # Visualize key features affected by ventilation
    features_to_plot = ["spo2", "pao2", "resp_rate", "heart_rate"]
    cf_fig = visualizer.plot_counterfactual_trajectories(
        times.cpu(), 
        factual_traj, 
        counterfactual_traj, 
        treatment_times=treatment_times, 
        features_to_plot=features_to_plot
    )
    cf_fig.savefig("counterfactual_trajectories.png")
    
    # 8. Vary treatment timing and analyze the effect
    logger.info("Analyzing treatment timing effects...")
    treatment_idx = 2  # mechanical_ventilation
    feature_idx = FEATURES.index("pao2")  # PaO2
    
    # Generate multiple counterfactual scenarios with different treatment timings
    n_scenarios = 5
    treatment_start_times = np.linspace(5, 30, n_scenarios)
    trajectories = []
    
    for start_time in treatment_start_times:
        cf_treatment = factual_treatments.clone()
        cf_treatment[:, treatment_idx] = 0  # Reset treatment
        
        # Find closest time index
        start_idx = (torch.abs(times - start_time)).argmin().item()
        cf_treatment[start_idx:, treatment_idx] = 1  # Apply treatment from start_time
        
        # Predict trajectory
        _, cf_traj = trainer.predict_counterfactual(x0, factual_treatments, cf_treatment, times)
        trajectories.append(cf_traj)
    
    # Plot the effect of treatment timing
    timing_fig = visualizer.plot_treatment_delay_effects(
        times.cpu(),
        trajectories,
        treatment_start_times,
        feature_idx,
        TREATMENTS[treatment_idx]
    )
    timing_fig.savefig("treatment_timing_effects.png")
    
    # 9. Create a heatmap of treatment timing vs. outcome
    logger.info("Creating treatment timing heatmap...")
    
    # Define evaluation times and treatment timings
    eval_times = np.linspace(10, 60, 50)
    treatment_timings = np.linspace(5, 30, 20)
    
    # Create a matrix to store outcomes
    outcomes = np.zeros((len(treatment_timings), len(eval_times)))
    
    # Generate predictions for each treatment timing
    for i, start_time in enumerate(treatment_timings):
        cf_treatment = factual_treatments.clone()
        cf_treatment[:, treatment_idx] = 0  # Reset treatment
        
        # Find closest time index
        start_idx = (torch.abs(times - start_time)).argmin().item()
        cf_treatment[start_idx:, treatment_idx] = 1  # Apply treatment from start_time
        
        # Predict trajectory
        _, cf_traj = trainer.predict_counterfactual(x0, factual_treatments, cf_treatment, times)
        
        # Interpolate to get values at evaluation times
        for j, eval_time in enumerate(eval_times):
            # Find closest time index
            eval_idx = (torch.abs(times - eval_time)).argmin().item()
            if eval_idx < len(times):
                # Get feature value at evaluation time
                feature_val = cf_traj[eval_idx, 0, feature_idx].item()
                outcomes[i, j] = feature_val
    
    # Create heatmap
    heatmap_fig = visualizer.create_heatmap_of_treatment_timing(
        feature_idx,
        treatment_idx,
        eval_times,
        treatment_timings,
        outcomes
    )
    heatmap_fig.savefig("treatment_timing_heatmap.png")
    
    logger.info("Experiment completed successfully")


if __name__ == "__main__":
    run_experiment(n_patients=500, num_epochs=30)
