import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# 1. Configuration
gan_files = {
    "CycleGAN": "/home/chinasa/python_projects/ganprint_removal/outputs/csvs/default/cycle_features_21_30.csv",
    "DistanceGAN": "/home/chinasa/python_projects/ganprint_removal/outputs/csvs/default/dist_features_21_30.csv",
    "DRIT": "/home/chinasa/python_projects/ganprint_removal/outputs/csvs/default/drit_features_21_30.csv",
    "StarGAN": "/home/chinasa/python_projects/ganprint_removal/outputs/csvs/default/star_features_21_30.csv",
}

output_excel_name = "/home/chinasa/python_projects/ganprint_removal/outputs/excel/scut_fft_high_noisy_Results1.xlsx"

def calculate_metrics(y_true, y_pred):
    """Calculates biometric error rates: 0=Fake, 1=Real."""
    # Ensure labels are explicitly mapped: 0 is Fake, 1 is Real
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
    acer = (apcer + bpcer) / 2
    return accuracy_score(y_true, y_pred), apcer, bpcer, acer

all_results = []

# === 2. The Main Experiment Loop ===
for gan_name, csv_path in gan_files.items():
    if not os.path.exists(csv_path):
        print(f"Skipping {gan_name}: File not found.")
        continue

    print(f"\n>>> Running Balanced Experiment for {gan_name}...")
    df = pd.read_csv(csv_path)
    
    # --- GLOBAL BALANCING ---
    # Labels: 1=Real, 0=Original Fake, 2=Synthetic Fake
    c0 = len(df[df["label"] == 0])
    c1 = len(df[df["label"] == 1])
    c2 = len(df[df["label"] == 2])
    
    # Balance all classes to the size of the smallest group
    min_count = min(c0, c1, c2)
    print(f"  [Info] Balancing classes to {min_count} samples each.")
    
    df_real = df[df["label"] == 1].sample(n=min_count, random_state=42)
    df_fake = df[df["label"] == 0].sample(n=min_count, random_state=42)
    df_synth = df[df["label"] == 2].sample(n=min_count, random_state=42)

    # Prepare Baseline (Real + Original Fake)
    df_baseline = pd.concat([df_real, df_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
    X_base = df_baseline.drop("label", axis=1).values
    y_base = df_baseline["label"].values
    
    # Prepare Synthetic Pool
    X_synth_pool = df_synth.drop("label", axis=1).values

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = KNeighborsClassifier(n_neighbors=7)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_base)):
        # PREPROCESSING (FIT ONLY ON TRAIN DATA) ---
        scaler = StandardScaler()
        selector = VarianceThreshold(threshold=0.01)

        # Scale and filter features based ONLY on the current fold's training set
        X_train_fold = selector.fit_transform(scaler.fit_transform(X_base[train_idx]))
        y_train_fold = y_base[train_idx]
        
        # Transform the test set and synthetic pool using the same parameters
        X_test_fold = selector.transform(scaler.transform(X_base[test_idx]))
        y_test_fold = y_base[test_idx]
        X_synth_fold = selector.transform(scaler.transform(X_synth_pool))

        rng = np.random.default_rng(seed=fold)

        def run_and_log(X_tr, y_tr, step):
            model.fit(X_tr, y_tr)
            acc, apcer, bpcer, acer = calculate_metrics(y_test_fold, model.predict(X_test_fold))
            all_results.append({
                "GAN_Type": gan_name, "Fold": fold + 1, "Step": step,
                "Accuracy": acc, "APCER": apcer, "BPCER": bpcer, "ACER": acer,
                "Train_Size": len(y_tr)
            })

        # Separate indices of Real and Fake in the training fold
        real_idx = np.where(y_train_fold == 1)[0]
        fake_idx = np.where(y_train_fold == 0)[0]
        
        # Determine subset sizes (Step 1 is 50% of the fold, Step 2 is 25%)
        s1_size = len(real_idx) // 2
        s2_size = len(real_idx) // 4

        # Select fixed subsets for Real and Fake to maintain the "Real" consistency
        idx_real_s1 = rng.choice(real_idx, size=s1_size, replace=False)
        idx_real_s2 = rng.choice(idx_real_s1, size=s2_size, replace=False)
        
        idx_fake_s1 = rng.choice(fake_idx, size=s1_size, replace=False)
        idx_fake_s2 = rng.choice(idx_fake_s1, size=s2_size, replace=False)

        # --- STEP 0: 100% Baseline (Full Training Fold) ---
        run_and_log(X_train_fold, y_train_fold, 0)

        # --- STEP 1: 50% Baseline ---
        X_tr1 = np.vstack((X_train_fold[idx_real_s1], X_train_fold[idx_fake_s1]))
        y_tr1 = np.hstack((np.ones(s1_size), np.zeros(s1_size)))
        run_and_log(X_tr1, y_tr1, 1)

        # --- STEP 2: 25% Baseline ---
        X_tr2 = np.vstack((X_train_fold[idx_real_s2], X_train_fold[idx_fake_s2]))
        y_tr2 = np.hstack((np.ones(s2_size), np.zeros(s2_size)))
        run_and_log(X_tr2, y_tr2, 2)

        # --- STEP 3: 50% Real + (25% Original Fake + 25% Synthetic) ---
        # Total size matches Step 1, but half of the Fakes are now Synthetic
        synth_s3 = X_synth_fold[rng.choice(len(X_synth_fold), size=s2_size, replace=False)]
        X_tr3 = np.vstack((X_train_fold[idx_real_s1], X_train_fold[idx_fake_s2], synth_s3))
        y_tr3 = np.hstack((np.ones(s1_size), np.zeros(s2_size), np.zeros(s2_size))) # Both fakes are Label 0
        run_and_log(X_tr3, y_tr3, 3)

        # --- STEP 4: 75% Real + (25% Original Fake + 50% Synthetic) ---
        idx_real_s4 = rng.choice(real_idx, size=(len(real_idx) * 3 // 4), replace=False)
        synth_s4 = X_synth_fold[rng.choice(len(X_synth_fold), size=s1_size, replace=False)]
        X_tr4 = np.vstack((X_train_fold[idx_real_s4], X_train_fold[idx_fake_s2], synth_s4))
        y_tr4 = np.hstack((np.ones(len(idx_real_s4)), np.zeros(s2_size), np.zeros(s1_size)))
        run_and_log(X_tr4, y_tr4, 4)

        # --- STEP 5: 100% Real + 100% Synthetic (No Original Fakes) ---
        # Total size matches Step 0
        synth_s5 = X_synth_fold[rng.choice(len(X_synth_fold), size=len(real_idx), replace=False)]
        X_tr5 = np.vstack((X_train_fold[real_idx], synth_s5))
        y_tr5 = np.hstack((np.ones(len(real_idx)), np.zeros(len(real_idx))))
        run_and_log(X_tr5, y_tr5, 5)

# === 3. Final Export ===

# 1. Check if we actually have data
if not all_results:
    print("ERROR: 'all_results' is empty! The KNN loop didn't collect any data.")
    print("Check your data paths and your loop logic before running the export.")
else:
    full_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_excel_name), exist_ok=True)

    print(f"Processing {len(full_df)} results...")

    try:
        with pd.ExcelWriter(output_excel_name) as writer:
            # Create summary
            summary = full_df.groupby(["GAN_Type", "Step"])[["Accuracy", "APCER", "BPCER", "ACER"]].mean().reset_index()
            
            # Write sheets
            summary.to_excel(writer, sheet_name="Final_Averages", index=False)
            full_df.to_excel(writer, sheet_name="Per_Fold_Details", index=False)

        print(f"\nDone! Experiment results saved to: {output_excel_name}")
    except Exception as e:
        print(f"Failed to save Excel file: {e}")