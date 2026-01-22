#%%
import numpy as np





def run_model(label, instron_disp):
    """Placeholder function for the forward model prediction.
    This function should be replaced with the actual model implementation.
    Inputs:
        label: np.ndarray of shape (H, W), material class labels
        instron_disp: np.ndarray of shape (T,), instron displacement values
    Outputs:
        predicted_disp: np.ndarray of shape (T, H, W, 2), predicted displacements
        predicted_forces: np.ndarray of shape (T,), predicted forces
    """

    num_time_steps = instron_disp.shape[0]
    predicted_disp = np.zeros((num_time_steps, ) + label.shape + (2,))

    predicted_forces = np.zeros(num_time_steps)

    return predicted_disp, predicted_forces


def evaluate_displacements(true_disp, predicted_disp):
    """Evaluate displacement predictions using Mean Squared Error (MSE).
    Inputs:
        true_disp: np.ndarray of shape (T, H, W, 2), true displacements
        predicted_disp: np.ndarray of shape (T, H, W, 2), predicted displacements
    Outputs:
        mse: float, mean squared error over all time steps and spatial locations
    """

    mse = np.mean((true_disp - predicted_disp) ** 2)
    return mse

def evaluate_forces(true_forces, predicted_forces):
    """Evaluate force predictions using Mean Squared Error (MSE).
    Inputs:
        true_forces: np.ndarray of shape (T,), true forces
        predicted_forces: np.ndarray of shape (T,), predicted forces
    Outputs:
        mse: float, mean squared error over all time steps
    """
    mse = np.mean((true_forces - predicted_forces) ** 2)
    return mse




if __name__ == "__main__":
    # Run the forward model
    test_number = 3 # between 0 and 89

    data = np.load('training-set/%03d.npz' % test_number) 

    X = data['DIC_X']
    disp = data['DIC_disp']
    instron_disp = data['instron_disp']
    instron_force = data['instron_force']
    label = data['label']
    predicted_disp, predicted_forces = run_model(label, instron_disp) # here we will actually called the submitted docker model

    # Evaluate displacements
    disp_mse = evaluate_displacements(disp, predicted_disp)
    print(f"Displacement MSE: {disp_mse:.6f}")

    # Evaluate forces
    force_mse = evaluate_forces(instron_force, predicted_forces)
    print(f"Force MSE: {force_mse:.6f}")