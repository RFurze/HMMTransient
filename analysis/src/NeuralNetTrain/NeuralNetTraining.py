import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_SPLIT_RATIO = 0.05  
EVALUATE_TOP_PERCENTILE = 10  

# All possible features in the dataset
feature_names = ['H', 'P', 'U', 'V', 'W', 'DPDX', 'DPDY', 'DPDZ', 'DHDX', 'DHDY', 'DHDZ', 'HDOT', 'PDOT']

# Define the combinations you want to test here:
feature_combinations = {
    "1. All Features": feature_names,
    "2. No Z-Direction": ['H', 'P', 'U', 'V', 'DPDX', 'DPDY', 'DHDX', 'DHDY', 'HDOT', 'PDOT'],
    "3. Core Flow & Transients": ['H', 'P', 'U', 'V', 'DPDX', 'DPDY', 'HDOT', 'PDOT'],
    "4. Spatial Gradients Only": ['H', 'P', 'U', 'V', 'DPDX', 'DPDY', 'DHDX', 'DHDY'],
    "5. Minimalist (H, P, DPDX, DPDY)": ['H', 'P', 'DPDX', 'DPDY']
}

# ==========================================
# 2. DATA LOADING (With Transpose Patches)
# ==========================================
print("Loading data...")
try:
    X = np.load('analysis/src/NeuralNetTrain/transient_existing_xi_d.npy')
    Y_dp = np.load('analysis/src/NeuralNetTrain/transient_existing_dp.npy')
    Y_dq = np.load('analysis/src/NeuralNetTrain/transient_existing_dq.npy')
    
    if len(X.shape) > 1 and X.shape[0] < X.shape[1] and X.shape[0] == len(feature_names):
        X = X.T
    if len(Y_dp.shape) > 1 and Y_dp.shape[0] < Y_dp.shape[1]:
        Y_dp = Y_dp.T
    if len(Y_dq.shape) > 1 and Y_dq.shape[0] < Y_dq.shape[1]:
        Y_dq = Y_dq.T
        
    min_target_len = min(len(Y_dp), len(Y_dq))
    Y_dp = Y_dp[:min_target_len]
    Y_dq = Y_dq[:min_target_len]
    
    if Y_dp.ndim == 1:
        Y_dp = Y_dp.reshape(-1, 1)
    Y = np.hstack((Y_dq, Y_dp))
    
    if len(X) > len(Y):
        X = X[:len(Y)]
    elif len(Y) > len(X):
        Y = Y[:len(X)]

except FileNotFoundError:
    print("Warning: .npy files not found. Generating mock data...")
    N_samples = 10000
    X = np.random.rand(N_samples, len(feature_names)) * 100
    Y = np.random.rand(N_samples, 3) * 10 

# ==========================================
# 3. REUSABLE EXPERIMENT FUNCTION
# ==========================================
def run_experiment(X_full, Y_full, f_names_full, target_feats, experiment_name=""):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Features ({len(target_feats)}): {', '.join(target_feats)}")
    print(f"{'='*60}")
    
    # Extract only the columns for the current combination
    feature_indices = [f_names_full.index(f) for f in target_feats]
    X_subset = X_full[:, feature_indices]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_subset, Y_full, test_size=TEST_SPLIT_RATIO, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_Y = StandardScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)

    print("Training Neural Network...")
    nn_model = MLPRegressor(hidden_layer_sizes=(128, 64), 
                            activation='relu',
                            solver='adam',
                            max_iter=1000, 
                            early_stopping=True,
                            random_state=42)

    nn_model.fit(X_train_scaled, Y_train_scaled)
    Y_pred = scaler_Y.inverse_transform(nn_model.predict(X_test_scaled))

    target_labels = ['DQX', 'DQY', 'DP']
    exp_results = {}

    for i, label in enumerate(target_labels):
        y_true_col = Y_test[:, i]
        y_pred_col = Y_pred[:, i]
        
        # R2 Scores
        r2_global = r2_score(y_true_col, y_pred_col)
        
        magnitudes = np.abs(y_true_col)
        threshold = np.percentile(magnitudes, 100 - EVALUATE_TOP_PERCENTILE)
        hard_indices = np.where(magnitudes >= threshold)[0]
        
        if len(hard_indices) > 1:
            r2_hard = r2_score(y_true_col[hard_indices], y_pred_col[hard_indices])
        else:
            r2_hard = float('nan') # Failsafe if not enough samples
            
        # Max Absolute Error Tracking
        abs_errors = np.abs(y_true_col - y_pred_col)
        max_err_idx = np.argmax(abs_errors)
        max_err_val = abs_errors[max_err_idx]
        true_val = y_true_col[max_err_idx]
        pred_val = y_pred_col[max_err_idx]
        
        exp_results[label] = {
            'R2_Global': r2_global, 
            'R2_Hard': r2_hard, 
            'MaxErr': max_err_val, 
            'True@MaxErr': true_val, 
            'Pred@MaxErr': pred_val
        }
        
        print(f"\nTarget: {label}")
        print(f"  Overall R2: {r2_global:.4f} | Hard Region R2: {r2_hard:.4f}")
        print(f"  Max Error : {max_err_val:.4e} (True: {true_val:.4e} | Pred: {pred_val:.4e})")

    return exp_results

# ==========================================
# 4. RUN COMBINATIONS LOOP
# ==========================================
summary_data = []

for name, feats in feature_combinations.items():
    results = run_experiment(X, Y, feature_names, feats, experiment_name=name)
    
    # Store data for the summary table
    summary_data.append({
        'Experiment': name,
        'Num_Features': len(feats),
        'DQX_R2_Global': results['DQX']['R2_Global'],
        'DQX_R2_Hard': results['DQX']['R2_Hard'],
        'DQY_R2_Global': results['DQY']['R2_Global'],
        'DQY_R2_Hard': results['DQY']['R2_Hard'],
        'DP_R2_Global': results['DP']['R2_Global'],
        'DP_R2_Hard': results['DP']['R2_Hard']
    })

# ==========================================
# 5. PRINT SUMMARY TABLE (NATIVE PYTHON)
# ==========================================
print("\n" + "="*115)
print("FINAL SUMMARY COMPARISON (R2 Scores: Global / Hard Region)")
print("="*115)

# Table Header
header = f"{'Experiment Name':<35} | {'Feats':<5} | {'DQX (Glob/Hard)':<18} | {'DQY (Glob/Hard)':<18} | {'DP (Glob/Hard)':<18}"
print(header)
print("-" * 115)

# Table Rows
for row in summary_data:
    exp_name = row['Experiment'][:34]
    feats = row['Num_Features']
    
    def format_r2(g_val, h_val):
        h_str = f"{h_val:.4f}" if not np.isnan(h_val) else "NaN"
        return f"{g_val:.4f} / {h_str}"

    str_dqx = format_r2(row['DQX_R2_Global'], row['DQX_R2_Hard'])
    str_dqy = format_r2(row['DQY_R2_Global'], row['DQY_R2_Hard'])
    str_dp = format_r2(row['DP_R2_Global'], row['DP_R2_Hard'])
    
    print(f"{exp_name:<35} | {feats:<5} | {str_dqx:<18} | {str_dqy:<18} | {str_dp:<18}")

print("="*115)