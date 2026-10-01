#%%
import numpy as np





def run_model(displacements, instron_disp, instron_force):
    """Placeholder function for the inverse model prediction.
    This function should be replaced with the actual model implementation.
    Inputs:
        displacements: np.ndarray of shape (T, H, W, 2), observed displacements
        instron_disp: np.ndarray of shape (T,), instron displacement values
        instron_force: np.ndarray of shape (T,), instron force values
    Outputs:
        predicted_label: np.ndarray of shape (H, W), predicted material class labels
    """
    predicted_label = np.zeros(displacements.shape[1:3], dtype=np.int32)

    return predicted_label


def evaluate_label(true_label, predicted_label):
    """Evaluate displacement predictions using Mean Squared Error (MSE).
    Inputs:
        true_label: np.ndarray of shape (H, W), true material class labels
        predicted_label: np.ndarray of shape (H, W), predicted material class labels
    Outputs:
        DICE score: float, DICE score between true and predicted labels
    """

    # Get unique classes from both true and predicted labels
    classes = np.unique(np.concatenate([true_label.flatten(), predicted_label.flatten()]))
    
    dice_scores = []
    for c in classes:
        true_mask = (true_label == c)
        pred_mask = (predicted_label == c)
        
        intersection = np.sum(true_mask & pred_mask)
        total = np.sum(true_mask) + np.sum(pred_mask)
        
        if total == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(2.0 * intersection / total)
    
    dice_score = np.mean(dice_scores)
    return dice_score




if __name__ == "__main__":
    # Run the inverse model
    test_number = 3 # between 0 and 89

    data = np.load('training-set/%03d.npz' % test_number) 

    X = data['DIC_X']
    disp = data['DIC_disp']
    instron_disp = data['instron_disp']
    instron_force = data['instron_force']
    label = data['label']

    predicted_label = run_model(disp, instron_disp, instron_force)
    dice_score = evaluate_label(label, predicted_label)
    print(f"DICE score: {dice_score:.6f}")