from neurosity import NeurositySDK
from dotenv import load_dotenv
import os
import torch
import numpy as np
from predictions_local.deeplearningpytorchpredictor import DeeplearningPytorchPredictor
from rf_model import RandomForest
from prediction_gaussiannb.pytorch.gaussiannb_model import GaussianNB

load_dotenv()

neurosity = NeurositySDK({
    "device_id": os.getenv("NEUROSITY_DEVICE_ID"),
})

neurosity.login({
    "email": os.getenv("NEUROSITY_EMAIL"),
    "password": os.getenv("NEUROSITY_PASSWORD")
})

# Initialize predictors
predictor = DeeplearningPytorchPredictor()

# Load Random Forest
rf_path = "/workspaces/Avatar-Aidan-Fork/prediction-random-forest/pytorch/custom_rf.pt.gz"
rf_predictor = RandomForest.load(rf_path)

# Fit Gaussian NB with dummy data (replace with real training data for accuracy)
num_features = 16
num_classes = 6
nb_predictor = GaussianNB(num_features, num_classes)
X_dummy = torch.randn(100, num_features)
y_dummy = torch.randint(0, num_classes, (100,))
nb_predictor.fit(X_dummy, y_dummy)

# Buffer for accumulating data
data_buffer = []
buffer_size = 512  # Assuming 256 Hz * 2 seconds

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
                dl_predicted_class = torch.argmax(dl_output, dim=1).item()
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
            nb_pred = nb_predictor.predict(rf_input)
            nb_predicted_class = nb_pred[0].item()
            nb_label = predictor.class_map.get(nb_predicted_class, "unknown")
            
            print(f"DL Predicted: {dl_label}, RF Predicted: {rf_label}, NB Predicted: {nb_label}")
            
            # Clear buffer or keep sliding
            data_buffer = data_buffer[-buffer_size//2:]  # Overlap

unsubscribe = neurosity.brainwaves_raw(callback)