#!/usr/bin/env python3
"""
Script to export scikit-learn LogisticRegression model coefficients
to a simple text format that can be read by the C++ implementation.
"""

import os
import sys

import joblib


def export_logistic_model(joblib_path, output_path):
    """Export logistic regression model to text format."""
    try:
        # Load the joblib model
        model = joblib.load(joblib_path)
        print(f"Loaded model from {joblib_path}")
        print(f"Model type: {type(model)}")

        # Extract coefficients and intercept
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            coef = model.coef_
            intercept = model.intercept_

            print(f"Coefficient shape: {coef.shape}")
            print(f"Intercept shape: {intercept.shape}")
            print(f"Number of classes: {len(intercept)}")
            print(f"Number of features: {coef.shape[1]}")

            # Write to text file
            with open(output_path, "w") as f:
                # Write header
                f.write("# Logistic Regression Model Export\n")
                f.write(f"# Classes: {len(intercept)}\n")
                f.write(f"# Features: {coef.shape[1]}\n")
                f.write(
                    "# Format: intercept_values, then coefficients (row-major)\n\n"
                )

                # Write intercepts
                f.write("INTERCEPT\n")
                for val in intercept:
                    f.write(f"{val:.10f}\n")

                # Write coefficients
                f.write("\nCOEFFICIENTS\n")
                for class_idx in range(coef.shape[0]):
                    for feature_idx in range(coef.shape[1]):
                        f.write(f"{coef[class_idx, feature_idx]:.10f}\n")

            print(f"Model exported to {output_path}")
            return True

        else:
            print("Error: Model doesn't have expected attributes (coef_, intercept_)")
            return False

    except Exception as e:
        print(f"Error loading or exporting model: {e}")
        return False


if __name__ == "__main__":
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    joblib_path = os.path.join(script_dir, "chess_lr.joblib")
    output_path = os.path.join(script_dir, "model_coefficients.txt")

    if len(sys.argv) > 1:
        joblib_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"Exporting model from: {joblib_path}")
    print(f"Output file: {output_path}")

    if not os.path.exists(joblib_path):
        print(f"Error: Model file not found at {joblib_path}")
        sys.exit(1)

    success = export_logistic_model(joblib_path, output_path)
    sys.exit(0 if success else 1)
