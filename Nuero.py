from neurosity import NeurositySDK
from dotenv import load_dotenv
import importlib.util
from pathlib import Path
import os
import time
import torch
import numpy as np
from predictions_local.deeplearningpytorchpredictor import DeeplearningPytorchPredictor
from rf_model import RandomForest

load_dotenv('.env.txt')

predictor = None
rf_predictor = None
gaussian_nb = None

def load_gaussian_nb_class():
    # Try multiple possible locations for the GaussianNB file
    possible_paths = [
        # Relative to the script directory
        Path(__file__).resolve().parent / "prediction-gaussiannb" / "pytorch" / "gaussiannb_model.py",
        # Relative to current working directory
        Path.cwd() / "prediction-gaussiannb" / "pytorch" / "gaussiannb_model.py",
        # Try to find repo root and look there
        Path(__file__).resolve().parent.parent / "prediction-gaussiannb" / "pytorch" / "gaussiannb_model.py",
    ]

    gaussian_nb_path = None
    for path in possible_paths:
        if path.exists():
            gaussian_nb_path = path
            break

    if gaussian_nb_path is None:
        raise FileNotFoundError(
            f"Could not find GaussianNB file. Searched in:\n" +
            "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease ensure you have cloned the complete repository with the prediction-gaussiannb directory."
        )

    spec = importlib.util.spec_from_file_location("prediction_gaussiannb.pytorch.gaussiannb_model", gaussian_nb_path)
    gaussian_nb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gaussian_nb_module)
    return gaussian_nb_module.GaussianNB

data_buffer = []
buffer_size = 512  # Assuming 256 Hz * 2 seconds

def get_env_var(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_neurosity_client():
    device_id = get_env_var("NEUROSITY_DEVICE_ID")
    neurosity = NeurositySDK({"device_id": device_id})
    neurosity.login({
        "email": get_env_var("NEUROSITY_EMAIL"),
        "password": get_env_var("NEUROSITY_PASSWORD"),
    })
    return neurosity


def create_predictors():
    predictor_instance = DeeplearningPytorchPredictor()

    # Try multiple possible locations for the RandomForest model
    possible_rf_paths = [
        # Relative to the script directory
        Path(__file__).resolve().parent / "prediction-random-forest" / "pytorch" / "custom_rf.pt.gz",
        # Relative to current working directory
        Path.cwd() / "prediction-random-forest" / "pytorch" / "custom_rf.pt.gz",
        # Try to find repo root and look there
        Path(__file__).resolve().parent.parent / "prediction-random-forest" / "pytorch" / "custom_rf.pt.gz",
    ]

    rf_path = None
    for path in possible_rf_paths:
        if path.exists():
            rf_path = path
            break

    if rf_path is None:
        raise FileNotFoundError(
            f"Could not find Random Forest model file. Searched in:\n" +
            "\n".join(f"  - {p}" for p in possible_rf_paths) +
            "\n\nPlease ensure you have cloned the complete repository with the prediction-random-forest directory."
        )

    rf_predictor_instance = RandomForest.load(str(rf_path))
    num_features = 16
    num_classes = 6
    GaussianNB = load_gaussian_nb_class()
    gaussian_nb_instance = GaussianNB(num_features, num_classes)
    X_dummy = torch.randn(100, num_features)
    y_dummy = torch.randint(0, num_classes, (100,))
    gaussian_nb_instance.fit(X_dummy, y_dummy)
    return predictor_instance, rf_predictor_instance, gaussian_nb_instance


def main():
    global predictor, rf_predictor, gaussian_nb
    neurosity = create_neurosity_client()
    predictor, rf_predictor, gaussian_nb = create_predictors()
    unsubscribe = neurosity.brainwaves_raw(callback)
    print("Neurosity stream started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        unsubscribe()
        print("Stream stopped.")


def callback(data):
    global data_buffer
    # Assuming data['data'] is list of channels, each with samples
    if 'data' in data:
        samples = np.array(data['data'])  # Shape: (channels, samples)
        # Assuming samples is (8, n), take the latest n samples
        data_buffer.extend(samples.T)  # Transpose to (n, 8)
        
        if len(data_buffer) >= buffer_size:
            # Take the last buffer_size samples
            batch = np.array(data_buffer[-buffer_size:])  # (512, 8)
            # Preprocess: perhaps standardize, but for now assume model handles
            # Convert to tensor, but model expects [batch, length] for 1 channel?
            # The model unsqueezes if 2D, so perhaps flatten channels
            # For simplicity, take mean across channels or something
            # Actually, let's assume the model is trained on single channel or adjust
            # For now, use channel 0
            input_tensor = torch.tensor(batch[:, 0], dtype=torch.float32).unsqueeze(0)  # [1, 512]
            
            # Deep Learning prediction
            dl_input = torch.tensor(batch[:, 0], dtype=torch.float32).unsqueeze(0)  # [1, 512]
            with torch.no_grad():
                dl_output = predictor(dl_input)
                dl_predicted_class = dl_output
                dl_label = predictor.class_map.get(dl_predicted_class, "unknown")
            
            # Random Forest prediction
            # RF expects [n_samples, n_features], trained on 16 features
            if batch.shape[1] < 16:
                rf_batch = np.pad(batch, ((0,0),(0,16-batch.shape[1])), 'constant')
            else:
                rf_batch = batch[:, :16]
            rf_input = torch.tensor(rf_batch, dtype=torch.float32)
            rf_pred = rf_predictor.predict(rf_input)
            rf_predicted_class = rf_pred[0].item()  # Assuming single sample
            rf_label = predictor.class_map.get(rf_predicted_class, "unknown")
            
            # Gaussian NB prediction
            nb_pred = gaussian_nb(rf_input)
            nb_predicted_class = nb_pred[0].item()
            nb_label = predictor.class_map.get(nb_predicted_class, "unknown")
            
            print(f"DL Predicted: {dl_label}, RF Predicted: {rf_label}, NB Predicted: {nb_label}")
            
            # Clear buffer or keep sliding
            data_buffer = data_buffer[-buffer_size//2:]  # Overlap


if __name__ == "__main__":
    main()