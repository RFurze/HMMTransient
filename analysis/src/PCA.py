import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------
# Example data generation
# -----------------------
# Suppose you have N samples and 7 inputs: [H, P, U, V, dpdx, dpdy, F]
# and 4 outputs: dQx, dQy, dPst, Fst
N = 9401
np.random.seed(123)
# load the tasks.csv file andd spllit each column in to an array
tasks = pd.read_csv('data/output/all_results/tasks.csv')
Hin = tasks['H'].to_numpy()
Pin = tasks['P'].to_numpy()
lmb1in = tasks['U'].to_numpy()
lmb2in = tasks['V'].to_numpy()
gradp1in = tasks['dpdx'].to_numpy()
gradp2in = tasks['dpdy'].to_numpy()
f_cavin = tasks['F'].to_numpy()

pst_results = pd.read_csv('data/output/all_results/pst_results.csv')
q_results = pd.read_csv('data/output/all_results/q_results.csv')
F_results = pd.read_csv('data/output/all_results/sst.csv')

# Fake inputs (X)
H     = np.asarray(Hin)
P     = np.asarray(Pin)
U     = np.asarray(lmb1in)
V     = np.asarray(lmb2in)
dpdx  = np.asarray(gradp1in)
dpdy  = np.asarray(gradp2in)
F     = np.asarray(f_cavin)

X = pd.DataFrame({
    'H': H, 'P': P, 'U': U, 'V': V, 'dpdx': dpdx, 'dpdy': dpdy, 'F': F
})

# Fake outputs (y)
dQx = q_results['Qx'].to_numpy()
dQy = q_results['Qy'].to_numpy()
dPst= pst_results['Pst'].to_numpy()
Fst = F_results['Fst'].to_numpy()

Y = pd.DataFrame({
    'dQx': dQx, 'dQy': dQy, 'dPst': dPst, 'Fst': Fst
})

# -------------------------------------------------
# 1.a) Compute correlation of inputs with outputs
# -------------------------------------------------
# Combine X and Y into a single DataFrame for convenience
df = pd.concat([X, Y], axis=1)

# Compute correlation matrix
corr_matrix = df.corr()

print("Correlation matrix between all variables (inputs & outputs):")
print(corr_matrix)

# You can extract just correlations between the inputs (rows) and outputs (columns):
input_cols = X.columns
output_cols = Y.columns
print("\nCorrelations of each input with each output:")
print(corr_matrix.loc[input_cols, output_cols])

# -------------------------------------------------
# 1.b) PCA Analysis on the input variables X
# -------------------------------------------------
# 1) Scale X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) Fit PCA
pca = PCA(n_components=None)  # or specify n_components < 7 if you want dimension reduction
pca.fit(X_scaled)

# 3) PCA results
print("\nExplained variance ratio by each principal component:")
print(pca.explained_variance_ratio_)

print("\nPCA components (loadings of each input on each principal component):")
loadings = pd.DataFrame(pca.components_, columns=input_cols,
                        index=[f'PC{i+1}' for i in range(pca.n_components_)])
print(loadings)

# 4) (Optional) You can examine correlation between principal components and each output
X_pca = pca.transform(X_scaled)  # shape (N, n_components)
PC_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# Correlate each principal component with outputs
pc_output_corr = pd.concat([PC_df, Y], axis=1).corr().loc[PC_df.columns, output_cols]
print("\nCorrelation between principal components and outputs:")
print(pc_output_corr)



import math
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
def greedy_downsample(X, radius):
    """
    Greedy downsampling: keep only points that are at least
    'radius' away from all previously selected points.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The input data to downsample.
    radius : float
        The distance threshold for inclusion.

    Returns
    -------
    selected_indices : list of int
        The indices of X that were retained.
    """
    selected_indices = []
    for i in range(len(X)):
        x_i = X[i]
        # Check if x_i is within 'radius' of any point in the selected set
        too_close = False
        for j in selected_indices:
            dist = np.linalg.norm(x_i - X[j])
            if dist < radius:
                too_close = True
                break
        if not too_close:
            selected_indices.append(i)
    return selected_indices

# -------------------------------------------------------------------------
# EXAMPLE SETUP:
# We assume you already have X, Y as DataFrames with shape:
#   X -> (n_samples, n_features)
#   Y -> (n_samples, n_outputs)
# Then:
#   X_np = X.values
#   Y_np = Y.values
#
# For demonstration, let's say Y has columns = ['dQx','dQy','dPst','Fst',...]
# Adjust to match your actual columns in Y.
# -------------------------------------------------------------------------

# EXAMPLE: We'll keep the same two theta_r0 values:
X_np = X.values
Y_np = Y.values  # shape (N, 4)
theta_values = [0.1e-6, 0.1e-7]
min_folds = 2

# Define multiple regression methods in a dictionary
# Key = model name (string), Value = instantiated regressor
models = {
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                        solver='adam', max_iter=2000, random_state=123),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=123),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),        # scale features
        ("ridge", Ridge(alpha=1.0))          # then fit ridge
    ]),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# We will collect results in a list of rows,
# then make a Pandas DataFrame for easy tabulation
all_results = []  # each element will be a dict with columns:
# [ "theta", "method", "output", "MSE", "MAE", "R2" ]

for theta_r0 in theta_values:
    # Compute radius
    r_o = - (1.0 / theta_r0) * math.log(0.5)

    # Downsample
    sel_idx = greedy_downsample(X_np, r_o)
    X_down = X_np[sel_idx, :]
    Y_down = Y_np[sel_idx, :]

    n_samples_down = X_down.shape[0]
    print(f"\nFor theta_r0={theta_r0}, radius={r_o:.4f}, downsampled points={n_samples_down}")

    # Check if we have enough samples for cross-validation
    if n_samples_down < min_folds:
        print("  Not enough samples to perform cross-validation. Skipping.")
        continue

    # KFold setup
    n_splits = min(5, n_samples_down)
    print(f"  Using n_splits={n_splits} for cross-validation.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

    # For each output column in Y
    for out_idx, out_name in enumerate(Y.columns):
        y_down = Y_down[:, out_idx]

        # For each model in our dictionary
        for method_name, model in models.items():

            # Cross-validate with multiple scoring metrics:
            #   neg_mean_squared_error, neg_mean_absolute_error, r2
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            scores = cross_validate(
                model, X_down, y_down,
                scoring=scoring,
                cv=kf,
                return_train_score=False
            )

            # Compute mean test errors (transform sign for MSE and MAE)
            mean_mse = -np.mean(scores['test_neg_mean_squared_error'])
            mean_mae = -np.mean(scores['test_neg_mean_absolute_error'])
            mean_r2  =  np.mean(scores['test_r2'])

            # Store in our results list
            result_row = {
                "theta": theta_r0,
                "method": method_name,
                "output": out_name,
                "MSE": mean_mse,
                "MAE": mean_mae,
                "R2": mean_r2
            }
            all_results.append(result_row)

# Convert all_results to DataFrame for easy viewing
results_df = pd.DataFrame(all_results)

print("\n===== Final Results Table =====")
print(results_df)

# Optionally, you can pivot or groupby for a nicer summary. For example:
summary = results_df.groupby(["theta","method","output"]).agg({
    "MSE": "mean",
    "MAE": "mean",
    "R2": "mean"
})
print("\n===== Grouped Summary =====")
print(summary)