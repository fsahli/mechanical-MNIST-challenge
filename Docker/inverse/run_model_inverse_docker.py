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


def run_model(disp, forces, instron_disp):
    """Dummy function for the inverse model prediction.
    
    This is a placeholder that should be replaced with your actual model implementation.
    The inverse model predicts material properties given displacement and force data.
    
    Inputs:
        disp: np.ndarray of shape (T, H, W, 2), displacement field
        forces: np.ndarray of shape (T,), measured forces
        instron_disp: np.ndarray of shape (T,), instron displacement values
    
    Outputs:
        predicted_label: np.ndarray of shape (H, W), predicted material class labels
    """
    # Get spatial dimensions from displacement field
    H, W = disp.shape[1:3]
    
    # TODO: Replace with actual model logic
    # For now, return a dummy prediction (all zeros, representing one material class)
    predicted_label = np.zeros((H, W), dtype=int)
    
    print(f"Processing {disp.shape[0]} time steps for spatial domain {H}x{W}")
    
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
    disp = data['disp']
    forces = data['forces']
    instron_disp = data['instron_disp']
    
    print(f"Data loaded successfully:")
    print(f"  - Displacement field shape: {disp.shape}")
    print(f"  - Forces shape: {forces.shape}")
    print(f"  - Instron displacement shape: {instron_disp.shape}")
    
    # Run the model
    print("Running inverse model...")
    predicted_label = run_model(disp, forces, instron_disp)
    
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
