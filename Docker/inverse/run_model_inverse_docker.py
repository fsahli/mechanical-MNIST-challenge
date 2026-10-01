#!/usr/bin/env python3
"""
Script to run the inverse model inside a Docker container.

Usage:
    docker run -v /path/to/data:/data your-image:tag /data/training-set/000.npz

Or with explicit python call:
    docker run -v /path/to/data:/data your-image:tag python run_model_inverse_docker.py /data/training-set/000.npz

The .npz file should be mounted into the container using Docker volumes.
"""

import sys
import numpy as np
import argparse


def run_model(DIC_disp, instron_disp, instron_force, DIC_X):
    """Dummy function for the inverse model prediction.

    This is a placeholder that should be replaced with your actual model implementation.
    The inverse model predicts material properties given displacement and force data.

    Inputs:
        DIC_disp: np.ndarray of shape (T, H, W, 2), observed displacement field
        instron_disp: np.ndarray of shape (T,), instron displacement values
        instron_force: np.ndarray of shape (T,), measured forces
        DIC_X: np.ndarray of shape (H, W, 2), coordinates of the DIC grid in [mm]

    Outputs:
        predicted_label: np.ndarray of shape (H, W), predicted material class labels
    """
    # Get spatial dimensions from displacement field
    H, W = DIC_disp.shape[1:3]

    # TODO: Replace with actual model logic
    # For now, return a dummy prediction (all zeros, representing one material class)
    predicted_label = np.zeros((H, W), dtype=int)

    print(f"Processing {DIC_disp.shape[0]} time steps for spatial domain {H}x{W}")

    return predicted_label


def main():
    parser = argparse.ArgumentParser(
        description='Run inverse model on mechanical MNIST data inside Docker container'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input .npz file (e.g., /data/training-set/000.npz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output .npz file (optional)'
    )
    
    args = parser.parse_args()
    
    # Load the input data
    print(f"Loading data from: {args.input_file}")
    try:
        data = np.load(args.input_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
        print("Make sure the file is mounted correctly in the Docker container.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Extract required fields for inverse problem
    DIC_disp = data['DIC_disp']
    instron_disp = data['instron_disp']
    instron_force = data['instron_force']
    DIC_X = data['DIC_X']

    print(f"Data loaded successfully:")
    print(f"  - Displacement field shape: {DIC_disp.shape}")
    print(f"  - Forces shape: {instron_force.shape}")
    print(f"  - Instron displacement shape: {instron_disp.shape}")
    print(f"  - DIC_X shape: {DIC_X.shape}")

    # Run the model
    print("Running inverse model...")
    predicted_label = run_model(DIC_disp, instron_disp, instron_force, DIC_X)
    
    print(f"Model completed:")
    print(f"  - Predicted label shape: {predicted_label.shape}")
    
    # Save output if requested
    if args.output:
        print(f"Saving results to: {args.output}")
        np.savez(
            args.output,
            predicted_label=predicted_label,
        )
        print("Results saved successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
