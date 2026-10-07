import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Define feature names based on forensics.py features + ela stats
    features = [
        "mean_brightness",
        "std_dev",
        "edge_density",
        "laplacian_variance",
        "texture_mean_abs_gradient",
        "entropy",
        "width",
        "height",
        "ela_mean",
        "ela_max",
        "ela_std"
    ]

    # Generate synthetic dataset for a basic working demo model
    np.random.seed(42)
    
    # Real image statistics (label 0)
    X_real = np.random.rand(500, len(features))
    y_real = np.zeros(500)

    # Tampered image statistics (label 1)
    X_fake = np.random.rand(500, len(features))
    # ELA responses and texture variance tend to be higher for tampered regions
    X_fake[:, 8:11] += 1.5 
    X_fake[:, 3] += 1.0 
    y_fake = np.ones(500)

    X = np.vstack((X_real, X_fake))
    y = np.concatenate((y_real, y_fake)).astype(int)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_scaled, y)

    # Save the artifacts expecting the naming convention from forensics.py
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(clf, "models/model.joblib")

    with open("models/features.json", "w") as f:
        json.dump(features, f)

    print("Machine learning model and scaler synthesized successfully in models/")

if __name__ == "__main__":
    main()
