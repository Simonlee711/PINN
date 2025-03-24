import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def generate_synthetic_patient_data(num_patients=100, max_hours=120, 
                                    irregular_sampling=True, noise_level=0.05):
    """
    Generate synthetic patient data similar to MIMIC-IV format
    
    Parameters:
    - num_patients: Number of patient trajectories to generate
    - max_hours: Maximum length of stay in hours
    - irregular_sampling: Whether to sample at irregular time intervals
    - noise_level: Amount of noise to add to the data
    
    Returns:
    - dataframe with patient data (mimicking MIMIC-IV structure)
    """
    # Define physiological variables to model
    variables = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'creatinine']
    
    # Define normal ranges for each variable
    normal_ranges = {
        'heart_rate': (60, 100),  # bpm
        'systolic_bp': (90, 120),  # mmHg
        'diastolic_bp': (60, 80),  # mmHg
        'creatinine': (0.7, 1.3)   # mg/dL
    }
    
    # Lists to store data
    all_data = []
    
    for subject_id in range(num_patients):
        # Generate length of stay (between 24 and max_hours)
        los = np.random.randint(24, max_hours)
        
        # Generate admission time
        admit_time = pd.Timestamp('2022-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        
        # Determine if patient will deteriorate
        deteriorate = np.random.random() < 0.3
        
        # Time of deterioration (if applicable)
        deteriorate_time = np.random.randint(los // 3, los) if deteriorate else None
        
        # Generate baseline values for this patient
        baselines = {}
        for var in variables:
            low, high = normal_ranges[var]
            # Generate a value within the normal range
            baselines[var] = np.random.uniform(low, high)
        
        # Define sampling times - either regular or irregular
        if irregular_sampling:
            # Generate irregular time points with varying density
            times = [0]  # Start at admission
            
            # More frequent measurements at the beginning and during deterioration
            i = 0
            while i < los:
                # Determine next measurement time
                if i < 24:  # First 24 hours: more frequent measurements
                    next_time = i + np.random.exponential(scale=2)
                elif deteriorate and abs(i - deteriorate_time) < 12:  # Near deterioration: more frequent
                    next_time = i + np.random.exponential(scale=1)
                else:  # Regular care: less frequent
                    next_time = i + np.random.exponential(scale=4)
                
                i = next_time
                if i < los:
                    times.append(i)
        else:
            # Regular hourly measurements
            times = list(range(los + 1))
        
        # Generate values for each time point
        for t in times:
            record = {
                'subject_id': subject_id,
                'hadm_id': subject_id * 10,  # Simple mapping for hospital admission ID
                'charttime': admit_time + pd.Timedelta(hours=t),
                'hours_since_admit': t
            }
            
            for var in variables:
                base = baselines[var]
                
                # Add circadian rhythm for heart rate and blood pressure
                circadian = 0
                if var in ['heart_rate', 'systolic_bp', 'diastolic_bp']:
                    # Assuming a 24-hour cycle, lowest at night (3 AM), highest in afternoon (3 PM)
                    hour_of_day = (admit_time.hour + t) % 24
                    circadian = 0.1 * base * np.sin(2 * np.pi * (hour_of_day - 3) / 24)
                
                # Add deterioration effect if applicable
                deterioration = 0
                if deteriorate and t > deteriorate_time:
                    # How far into deterioration
                    time_since = (t - deteriorate_time) / 24  # In days
                    
                    # Different effects for different variables
                    if var == 'heart_rate':
                        # Heart rate increases with deterioration
                        deterioration = 20 * (1 - np.exp(-time_since))
                    elif var == 'systolic_bp':
                        # Systolic BP might drop
                        deterioration = -15 * (1 - np.exp(-time_since))
                    elif var == 'diastolic_bp':
                        # Diastolic BP might also drop
                        deterioration = -10 * (1 - np.exp(-time_since))
                    elif var == 'creatinine':
                        # Creatinine rises with kidney dysfunction
                        deterioration = 1.5 * time_since
                
                # Add noise
                noise = np.random.normal(0, noise_level * base)
                
                # Calculate final value
                value = base + circadian + deterioration + noise
                
                # Ensure values stay positive
                value = max(value, 0.01)
                
                # Add to record
                record[var] = value
            
            all_data.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Add medication and procedure events
    med_proc_data = []
    for subject_id in range(num_patients):
        # Get admission time for this patient
        patient_data = df[df['subject_id'] == subject_id]
        admit_time = patient_data['charttime'].min()
        los = (patient_data['charttime'].max() - admit_time).total_seconds() / 3600
        
        # Add some medications
        med_list = ['norepinephrine', 'epinephrine', 'vancomycin', 'ceftriaxone', 
                   'furosemide', 'morphine', 'propofol', 'insulin']
        
        # Determine if patient deteriorated
        deteriorate = (df[(df['subject_id'] == subject_id) & 
                           (df['heart_rate'] > normal_ranges['heart_rate'][1] * 1.2)]).shape[0] > 0
        
        # Add 2-5 medications per patient
        num_meds = np.random.randint(2, 6)
        for _ in range(num_meds):
            med = np.random.choice(med_list)
            
            # More pressors and antibiotics if deteriorating
            if deteriorate and np.random.random() < 0.7:
                med = np.random.choice(['norepinephrine', 'epinephrine', 'vancomycin', 'ceftriaxone'])
            
            # Random start time (weighted toward beginning of stay)
            start_hour = np.random.exponential(scale=los/4)
            if start_hour > los:
                start_hour = los / 2
            
            # Duration depends on medication
            if med in ['vancomycin', 'ceftriaxone']:
                duration = np.random.uniform(3*24, 7*24)  # 3-7 days
            elif med in ['norepinephrine', 'epinephrine']:
                duration = np.random.uniform(12, 72)  # 12-72 hours
            else:
                duration = np.random.uniform(1, 48)  # 1-48 hours
            
            # Add medication record
            med_proc_data.append({
                'subject_id': subject_id,
                'hadm_id': subject_id * 10,
                'event_type': 'medication',
                'event_name': med,
                'starttime': admit_time + pd.Timedelta(hours=start_hour),
                'endtime': admit_time + pd.Timedelta(hours=min(start_hour + duration, los)),
                'hours_since_admit_start': start_hour,
                'hours_since_admit_end': min(start_hour + duration, los)
            })
        
        # Add procedures (0-3 per patient)
        proc_list = ['intubation', 'central_line', 'arterial_line', 'dialysis', 'CT_scan', 'MRI']
        
        num_procs = np.random.randint(0, 4)
        for _ in range(num_procs):
            proc = np.random.choice(proc_list)
            
            # Procedures more likely if deteriorating
            if deteriorate and np.random.random() < 0.8:
                proc = np.random.choice(['intubation', 'central_line', 'arterial_line', 'dialysis'])
            
            # Procedures tend to happen at specific times
            if deteriorate:
                # For deteriorating patients, procedures often follow deterioration
                deteriorate_time_data = df[(df['subject_id'] == subject_id) & 
                                          (df['heart_rate'] > normal_ranges['heart_rate'][1] * 1.2)]
                if not deteriorate_time_data.empty:
                    earliest_deteriorate = deteriorate_time_data['hours_since_admit'].min()
                    proc_time = earliest_deteriorate + np.random.uniform(0, 12)
                else:
                    proc_time = np.random.uniform(0, los)
            else:
                # For stable patients, procedures often happen early
                proc_time = np.random.exponential(scale=los/5)
            
            if proc_time > los:
                proc_time = los / 2
            
            # Add procedure record
            med_proc_data.append({
                'subject_id': subject_id,
                'hadm_id': subject_id * 10,
                'event_type': 'procedure',
                'event_name': proc,
                'starttime': admit_time + pd.Timedelta(hours=proc_time),
                'endtime': admit_time + pd.Timedelta(hours=proc_time),  # Procedures are point events
                'hours_since_admit_start': proc_time,
                'hours_since_admit_end': proc_time
            })
    
    events_df = pd.DataFrame(med_proc_data)
    
    return df, events_df

class LatentODEFunc(nn.Module):
    """Neural ODE function defining the dynamics in latent space"""
    
    def __init__(self, latent_dim, hidden_dim):
        super(LatentODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, t, x):
        """
        t: time
        x: state
        """
        return self.net(x)

class Encoder(nn.Module):
    """Encodes measurements and events into latent space"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """Decodes latent space back to measurements"""
    
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class LatentODEModel(nn.Module):
    """Full Neural ODE model for patient trajectories"""
    
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32):
        super(LatentODEModel, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.dynamics = LatentODEFunc(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)
        self.latent_dim = latent_dim
        
    def forward(self, first_obs, first_time, query_times, mask=None):
        # If no mask is provided, assume query_times is one-dimensional and common to all samples.
        if mask is None or query_times.dim() == 1:
            first_latent = self.encoder(first_obs)
            latent_traj = odeint(self.dynamics, first_latent, query_times, method='dopri5')
            # latent_traj shape: (num_times, batch_size, latent_dim)
            latent_traj = latent_traj.permute(1, 0, 2)
            batch_size, num_times, _ = latent_traj.shape
            latent_traj_reshaped = latent_traj.reshape(-1, self.latent_dim)
            pred = self.decoder(latent_traj_reshaped)
            pred = pred.reshape(batch_size, num_times, -1)
            return pred, latent_traj
        
        # Otherwise, handle each sample individually based on its valid time points.
        batch_size, max_len = query_times.shape
        device = query_times.device
        # Determine output dimension from decoder's final layer.
        out_dim = self.decoder.net[-1].out_features
        preds = torch.zeros(batch_size, max_len, out_dim, device=device)
        latent_trajs = []
        for i in range(batch_size):
            valid_len = int(mask[i].sum().item())
            # Extract the valid (non-padded) time points (a 1D tensor).
            t_i = query_times[i, :valid_len]
            latent_i = self.encoder(first_obs[i].unsqueeze(0))  # shape (1, latent_dim)
            # Solve the ODE for the i-th sample with its own time vector.
            latent_traj_i = odeint(self.dynamics, latent_i, t_i, method='dopri5')
            # Remove the extra batch dimension added by odeint.
            latent_traj_i = latent_traj_i.squeeze(1)  # shape (valid_len, latent_dim)
            pred_i = self.decoder(latent_traj_i)        # shape (valid_len, input_dim)
            preds[i, :valid_len, :] = pred_i
            latent_trajs.append(latent_traj_i)
        return preds, latent_trajs


class PatientTrajectoryDataset(Dataset):
    """Dataset for patient trajectories"""
    
    def __init__(self, vitals_df, events_df, variables, max_seq_length=48):
        self.vitals_df = vitals_df
        self.events_df = events_df
        self.variables = variables
        self.max_seq_length = max_seq_length
        
        # Get unique patient IDs
        self.subject_ids = vitals_df['subject_id'].unique()
        
        # Create scalers for normalization
        self.scalers = {}
        for var in variables:
            scaler = MinMaxScaler()
            scaler.fit(vitals_df[var].values.reshape(-1, 1))
            self.scalers[var] = scaler
    
    def __len__(self):
        return len(self.subject_ids)
    
    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        
        # Get patient data
        patient_vitals = self.vitals_df[self.vitals_df['subject_id'] == subject_id]
        patient_events = self.events_df[self.events_df['subject_id'] == subject_id]
        
        # Sort by time
        patient_vitals = patient_vitals.sort_values('hours_since_admit')
        
        # Get times and values
        times = patient_vitals['hours_since_admit'].values
        
        # Normalize times to start at 0
        first_time = times[0]
        times = times - first_time
        
        # Normalize values
        values = []
        for var in self.variables:
            var_values = patient_vitals[var].values
            var_values_norm = self.scalers[var].transform(var_values.reshape(-1, 1)).flatten()
            values.append(var_values_norm)
        
        values = np.column_stack(values)
        
        # Handle events (medications, procedures)
        # Create binary indicators for active medications/procedures at each time
        unique_events = self.events_df['event_name'].unique()
        event_indicators = np.zeros((len(times), len(unique_events)))
        
        for i, t in enumerate(times):
            actual_time = t + first_time  # Convert back to original time scale
            
            for _, event in patient_events.iterrows():
                event_idx = np.where(unique_events == event['event_name'])[0][0]
                
                # Check if event is active at this time
                if event['hours_since_admit_start'] <= actual_time <= event['hours_since_admit_end']:
                    event_indicators[i, event_idx] = 1
        
        # Combine vitals and events
        features = np.hstack([values, event_indicators])
        
        # Limit sequence length if needed
        if len(times) > self.max_seq_length:
            # Take first point and then evenly spaced points
            indices = np.append(0, np.linspace(1, len(times)-1, self.max_seq_length-1).astype(int))
            times = times[indices]
            features = features[indices]
        
        return {
            'subject_id': subject_id,
            'times': torch.FloatTensor(times),
            'values': torch.FloatTensor(features),
            'length': len(times)
        }
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences"""
        # Sort batch by sequence length (descending)
        batch = sorted(batch, key=lambda x: x['length'], reverse=True)
        
        # Get max sequence length in this batch
        max_len = batch[0]['length']
        
        # Extract data
        subject_ids = [item['subject_id'] for item in batch]
        lengths = [item['length'] for item in batch]
        
        # Pad times and values
        batch_size = len(batch)
        feature_dim = batch[0]['values'].shape[1]
        
        padded_times = torch.zeros(batch_size, max_len)
        padded_values = torch.zeros(batch_size, max_len, feature_dim)
        
        for i, item in enumerate(batch):
            seq_len = item['length']
            padded_times[i, :seq_len] = item['times']
            padded_values[i, :seq_len, :] = item['values']
        
        # Create a mask for valid time points
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        return {
            'subject_ids': subject_ids,
            'times': padded_times,
            'values': padded_values,
            'lengths': lengths,
            'mask': mask
        }

def train_model(model, train_loader, epochs=10, lr=1e-3, clip_grad=10.0):
    """Train the neural ODE model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            times = batch['times']
            values = batch['values']
            mask = batch['mask']
            
            # Get initial observations
            first_obs = values[:, 0, :]
            
            # Forward pass
            pred, _ = model(first_obs, times[:, 0].unsqueeze(1), times, mask)
            
            # Calculate loss (MSE on valid time points only)
            loss = torch.sum(((pred - values) ** 2) * mask.unsqueeze(-1)) / torch.sum(mask)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
    
    return losses

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    total_mse = 0
    count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            times = batch['times']
            values = batch['values']
            mask = batch['mask']
            
            # Get initial observations
            first_obs = values[:, 0, :]
            
            # Forward pass
            pred, _ = model(first_obs, times[:, 0], times)
            
            # Calculate MSE on valid time points only
            mse = torch.sum(((pred - values) ** 2) * mask.unsqueeze(-1)) / torch.sum(mask)
            
            total_mse += mse.item()
            count += 1
    
    avg_mse = total_mse / count
    print(f"Test MSE: {avg_mse:.4f}")
    
    return avg_mse

def interpolate_trajectory(model, patient_data, variables, time_points=None):
    """
    Interpolate a patient's trajectory at arbitrary time points
    
    Parameters:
    - model: trained LatentODEModel
    - patient_data: dictionary with times and values for a patient
    - variables: list of variable names
    - time_points: arbitrary time points to query (if None, uses 100 evenly spaced points)
    
    Returns:
    - time_points: query time points
    - true_trajectory: ground truth values at observed time points
    - pred_trajectory: predicted values at query time points
    """
    model.eval()
    
    with torch.no_grad():
        times = patient_data['times'][0]  # (seq_len,)
        values = patient_data['values'][0]  # (seq_len, feature_dim)
        
        # Get initial observation
        first_obs = values[0:1]  # (1, feature_dim)
        
        # Define query time points if not provided
        if time_points is None:
            # Create 100 evenly spaced points from 0 to max time
            max_time = times[-1].item()
            time_points = torch.linspace(0, max_time, 100).unsqueeze(0)  # (1, 100)
        
        # Forward pass to get predictions at query times
        pred, latent_traj = model(first_obs, times[0:1], time_points)
        
        return time_points.squeeze().numpy(), values.numpy(), pred.squeeze().numpy()

def visualize_patient_trajectory(time_points, true_times, true_values, pred_values, variables, variable_indices, events_df=None, patient_id=None):
    """
    Visualize a patient's true and interpolated trajectory
    
    Parameters:
    - time_points: query time points for predictions
    - true_times: observed time points
    - true_values: observed values
    - pred_values: predicted values at query time points
    - variables: list of variable names
    - variable_indices: indices of variables in the feature vector
    - events_df: DataFrame with medication and procedure events
    - patient_id: ID of the patient for filtering events
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3*n_vars), sharex=True)
    
    for i, var in enumerate(variables):
        var_idx = variable_indices[i]
        ax = axes[i] if n_vars > 1 else axes
        
        # Plot true values
        ax.scatter(true_times, true_values[:, var_idx], color='blue', marker='o', 
                   label='Observed', alpha=0.7)
        
        # Plot predicted trajectory
        ax.plot(time_points, pred_values[:, var_idx], color='red', 
                label='Neural ODE prediction', alpha=0.8)
        
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    # Add events if provided
    if events_df is not None and patient_id is not None:
        # Get events for this patient
        patient_events = events_df[events_df['subject_id'] == patient_id]
        
        for _, event in patient_events.iterrows():
            # Add vertical line for event
            start_time = event['hours_since_admit_start']
            for i in range(n_vars):
                ax = axes[i] if n_vars > 1 else axes
                ax.axvline(x=start_time, color='green', linestyle='--', alpha=0.5)
                
                # Add text label
                y_pos = ax.get_ylim()[1] * 0.9
                ax.text(start_time, y_pos, event['event_name'], 
                        rotation=90, verticalalignment='top', fontsize=8)
    
    plt.xlabel('Hours since admission')
    plt.tight_layout()
    return fig

def main():
    # Generate synthetic EHR data
    print("Generating synthetic EHR data...")
    vitals_df, events_df = generate_synthetic_patient_data(num_patients=100, max_hours=168)
    
    print(f"Generated data for {len(vitals_df['subject_id'].unique())} patients")
    print(f"Total vital sign records: {len(vitals_df)}")
    print(f"Total medication/procedure events: {len(events_df)}")
    
    # Display some example data
    print("\nExample vital signs data:")
    print(vitals_df.head())
    
    print("\nExample medication/procedure events:")
    print(events_df.head())
    
    # Define variables we want to model
    variables = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'creatinine']
    
    # Create dataset
    dataset = PatientTrajectoryDataset(vitals_df, events_df, variables)
    
    # Split into train and test sets
    n_train = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # Get feature dimension (vitals + events)
    sample = next(iter(train_loader))
    feature_dim = sample['values'].shape[2]
    
    print(f"\nFeature dimension: {feature_dim}")
    
    # Initialize model
    model = LatentODEModel(
        input_dim=feature_dim, 
        latent_dim=16,
        hidden_dim=64
    )
    
    # Train model
    print("\nTraining model...")
    losses = train_model(model, train_loader, epochs=20)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_mse = evaluate_model(model, test_loader)
    
    # Visualize training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png')
    
    # Visualize example patient trajectories
    print("\nVisualizing example patient trajectories...")
    
    # Get an example patient
    test_batch = next(iter(test_loader))
    example_id = test_batch['subject_ids'][0]
    example_data = {
        'times': test_batch['times'][:1],
        'values': test_batch['values'][:1]
    }
    
    # Get original patient data for context
    patient_vitals = vitals_df[vitals_df['subject_id'] == example_id]
    patient_events = events_df[events_df['subject_id'] == example_id]
    
    print(f"\nExample patient {example_id} has {len(patient_vitals)} vital sign records and {len(patient_events)} events")
    
    # Define variable indices in the feature vector (excluding event indicators)
    variable_indices = list(range(len(variables)))
    
    # Interpolate trajectory
    time_points, true_values, pred_values = interpolate_trajectory(model, example_data, variables)
    
    # Get true times
    true_times = patient_vitals['hours_since_admit'].values
    
    # Visualize
    fig = visualize_patient_trajectory(
        time_points, true_times, true_values, pred_values, 
        variables, variable_indices, events_df, example_id
    )
    plt.savefig('patient_trajectory.png')
    
    # Forecasting example
    print("\nDemonstrating forecasting capability...")
    
    # Use only first half of data for this patient
    half_length = len(true_times) // 2
    truncated_data = {
        'times': torch.FloatTensor(true_times[:half_length]).unsqueeze(0),
        'values': torch.FloatTensor(true_values[:half_length]).unsqueeze(0)
    }
    
    # Forecast to the full time range
    forecast_times = torch.linspace(0, true_times[-1], 100).unsqueeze(0)
    forecast_time_points, forecast_true_values, forecast_pred_values = interpolate_trajectory(
        model, truncated_data, variables, forecast_times
    )
    
    # Visualize forecast
    fig_forecast = visualize_patient_trajectory(
        forecast_time_points, true_times, true_values, forecast_pred_values, 
        variables, variable_indices, events_df, example_id
    )
    
    # Add vertical line to show where forecasting begins
    for i, var in enumerate(variables):
        ax = fig_forecast.axes[i] if len(variables) > 1 else fig_forecast.axes
        ax.axvline(x=true_times[half_length-1], color='purple', linestyle='-.',
                  label='Forecast start')
        if i == 0:
            ax.legend()
    
    plt.savefig('patient_forecast.png')
    
    # Demonstrate imputation capability
    print("\nDemonstrating imputation capability...")
    
    # Create data with gaps by removing some points
    gap_indices = np.random.choice(
        range(len(true_times)), 
        size=len(true_times)//3, 
        replace=False
    )
    
    gap_times = np.delete(true_times, gap_indices)
    gap_values = np.delete(true_values, gap_indices, axis=0)
    
    gap_data = {
        'times': torch.FloatTensor(gap_times).unsqueeze(0),
        'values': torch.FloatTensor(gap_values).unsqueeze(0)
    }
    
    # Run imputation
    impute_times = torch.FloatTensor(true_times).unsqueeze(0)
    impute_time_points, impute_true_values, impute_pred_values = interpolate_trajectory(
        model, gap_data, variables, impute_times
    )
    
    # Visualize imputation
    fig_impute = plt.figure(figsize=(12, 3*len(variables)))
    
    for i, var in enumerate(variables):
        var_idx = variable_indices[i]
        ax = plt.subplot(len(variables), 1, i+1)
        
        # Plot removed points differently
        kept_mask = np.ones(len(true_times), dtype=bool)
        kept_mask[gap_indices] = False
        
        # Points we kept in training data
        ax.scatter(true_times[kept_mask], true_values[kept_mask, var_idx], 
                   color='blue', marker='o', label='Observed', alpha=0.7)
        
        # Points we removed (ground truth for imputation)
        ax.scatter(true_times[~kept_mask], true_values[~kept_mask, var_idx], 
                   color='green', marker='x', label='Removed (ground truth)', alpha=0.7)
        
        # Imputed trajectory
        ax.plot(impute_time_points, impute_pred_values[:, var_idx], 
                color='red', label='Neural ODE imputation', alpha=0.8)
        
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.xlabel('Hours since admission')
    plt.tight_layout()
    plt.savefig('patient_imputation.png')
    
    # Latent space analysis
    print("\nAnalyzing latent space trajectories...")
    
    # Get latent trajectories for a few patients
    patient_samples = 3
    latent_fig = plt.figure(figsize=(15, 10))
    
    # Create a 3D plot for first 3 dimensions of latent space
    ax_3d = latent_fig.add_subplot(2, 2, 1, projection='3d')
    
    # 2D projections
    ax_xy = latent_fig.add_subplot(2, 2, 2)
    ax_xz = latent_fig.add_subplot(2, 2, 3)
    ax_yz = latent_fig.add_subplot(2, 2, 4)
    
    # Get a few different patients
    for i in range(patient_samples):
        test_batch = next(iter(test_loader))
        patient_id = test_batch['subject_ids'][i]
        patient_data = {
            'times': test_batch['times'][i:i+1],
            'values': test_batch['values'][i:i+1]
        }
        
        # Get patient vitals to know max time
        patient_vitals = vitals_df[vitals_df['subject_id'] == patient_id]
        max_time = patient_vitals['hours_since_admit'].max()
        
        # Query times for smooth trajectory
        query_times = torch.linspace(0, max_time, 100).unsqueeze(0)
        
        with torch.no_grad():
            # Get initial observation
            first_obs = patient_data['values'][:, 0, :]
            
            # Get latent trajectory
            _, latent_traj = model(first_obs, patient_data['times'][:, 0:1], query_times)
            
            # Extract first 3 dimensions for visualization
            latent_traj = latent_traj.squeeze().numpy()
            
            if latent_traj.shape[1] >= 3:
                x, y, z = latent_traj[:, 0], latent_traj[:, 1], latent_traj[:, 2]
                
                # Plot 3D trajectory
                ax_3d.plot(x, y, z, label=f'Patient {patient_id}')
                
                # Plot 2D projections
                ax_xy.plot(x, y, label=f'Patient {patient_id}')
                ax_xz.plot(x, z)
                ax_yz.plot(y, z)
                
                # Mark starting points
                ax_3d.scatter(x[0], y[0], z[0], marker='o')
                ax_xy.scatter(x[0], y[0], marker='o')
                ax_xz.scatter(x[0], z[0], marker='o')
                ax_yz.scatter(y[0], z[0], marker='o')
    
    ax_3d.set_title('Latent Space Trajectories (3D)')
    ax_3d.set_xlabel('Latent Dimension 1')
    ax_3d.set_ylabel('Latent Dimension 2') 
    ax_3d.set_zlabel('Latent Dimension 3')
    ax_3d.legend()
    
    ax_xy.set_title('Latent Dimensions 1 vs 2')
    ax_xy.set_xlabel('Latent Dimension 1')
    ax_xy.set_ylabel('Latent Dimension 2')
    ax_xy.legend()
    
    ax_xz.set_title('Latent Dimensions 1 vs 3')
    ax_xz.set_xlabel('Latent Dimension 1')
    ax_xz.set_ylabel('Latent Dimension 3')
    
    ax_yz.set_title('Latent Dimensions 2 vs 3')
    ax_yz.set_xlabel('Latent Dimension 2')
    ax_yz.set_ylabel('Latent Dimension 3')
    
    plt.tight_layout()
    plt.savefig('latent_space.png')
    
    # Create summary report
    print("\nCreating summary statistics...")
    
    # For each variable, calculate MAE for interpolation, forecasting, imputation
    summary_data = []
    
    for i, var in enumerate(variables):
        var_idx = variable_indices[i]
        
        # Interpolation error
        interp_mae = np.mean(np.abs(
            true_values[:, var_idx] - 
            np.interp(true_times, time_points, pred_values[:, var_idx])
        ))
        
        # Forecasting error (only on the forecasted part)
        forecast_indices = true_times > true_times[half_length-1]
        if np.any(forecast_indices):
            forecast_true = true_values[forecast_indices, var_idx]
            forecast_pred = np.interp(
                true_times[forecast_indices], 
                forecast_time_points, 
                forecast_pred_values[:, var_idx]
            )
            forecast_mae = np.mean(np.abs(forecast_true - forecast_pred))
        else:
            forecast_mae = np.nan
        
        # Imputation error (only on the imputed points)
        impute_true = true_values[gap_indices, var_idx]
        impute_pred = np.interp(
            true_times[gap_indices], 
            impute_time_points, 
            impute_pred_values[:, var_idx]
        )
        impute_mae = np.mean(np.abs(impute_true - impute_pred))
        
        summary_data.append({
            'Variable': var,
            'Interpolation MAE': interp_mae,
            'Forecasting MAE': forecast_mae,
            'Imputation MAE': impute_mae
        })
    
    # Create and display summary table
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Statistics (Mean Absolute Error):")
    print(summary_df)
    
    print("\nModel successfully trained and evaluated!")
    print("Visualizations saved as PNG files.")
    
    return model, vitals_df, events_df

if __name__ == "__main__":
    main()
