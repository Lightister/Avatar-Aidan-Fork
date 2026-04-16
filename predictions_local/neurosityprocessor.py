import websocket
import json
import threading
import time
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import requests
import numpy as np
from scipy.signal import butter, filtfilt  # For filtering

class NeurosityDataProcessor:
    """
    Handles EEG data streaming from Neurosity headset via WebSocket,
    with detailed channel mapping, scaling, and filtering to match Avatar's expectations.
    """

    # Neurosity Crown has 8 EEG channels in this order (10-20 system):
    NEUROSITY_CHANNELS = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    def __init__(self, api_key, device_id, sample_time=2, fs=256):
        self.api_key = api_key
        self.device_id = device_id
        self.sample_time = sample_time
        self.fs = fs  # Sampling frequency (Neurosity: ~256 Hz)
        self.ws = None
        self.data_buffer = []
        self.eeg_df = None
        self.X_tensor = None
        self.streaming = False

    def _authenticate(self):
        # Get access token from Neurosity API
        url = "https://api.neurosity.co/v1/auth/login"
        headers = {"Content-Type": "application/json"}
        data = {"apiKey": self.api_key}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()["accessToken"]
        else:
            raise Exception(f"Authentication failed: {response.text}")

    def _connect_websocket(self):
        token = self._authenticate()
        ws_url = f"wss://api.neurosity.co/v1/devices/{self.device_id}/stream?token={token}"
        self.ws = websocket.WebSocketApp(ws_url,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close)
        self.ws.run_forever()

    def _on_message(self, ws, message):
        data = json.loads(message)
        if "data" in data and "eeg" in data["data"]:
            # Append raw EEG array (list of 8 floats in microvolts)
            self.data_buffer.append(data["data"]["eeg"])

    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def _on_close(self, ws):
        print("WebSocket closed")

    def _apply_filters(self, data):
        # Apply bandpass (1-40 Hz) and notch (50 Hz) filters
        # data: shape (samples, channels)
        nyq = 0.5 * self.fs
        # Bandpass: 1-40 Hz
        low = 1 / nyq
        high = 40 / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data, axis=0)
        # Notch: 50 Hz
        notch_freq = 50 / nyq
        b_notch, a_notch = butter(4, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
        filtered = filtfilt(b_notch, a_notch, filtered, axis=0)
        return filtered

    def capture_data(self):
        self.data_buffer = []
        self.streaming = True
        ws_thread = threading.Thread(target=self._connect_websocket)
        ws_thread.start()
        time.sleep(self.sample_time)
        self.streaming = False
        self.ws.close()
        ws_thread.join()

        if self.data_buffer:
            # Convert to NumPy array: shape (samples, 8 channels)
            raw_data = np.array(self.data_buffer)
            # Apply filtering
            filtered_data = self._apply_filters(raw_data)
            # Create DataFrame with channel names
            self.eeg_df = pd.DataFrame(filtered_data, columns=[f"EEG_{i}" for i in range(8)])
        return self.eeg_df

    def preprocess_eeg(self):
        if self.eeg_df is None:
            raise ValueError("Data not captured yet. Call capture_data() first.")
        # Scaling: Standardize to mean=0, std=1 (matches BrainFlow processor)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.eeg_df)
        self.X_tensor = torch.tensor(X_scaled, dtype=torch.float)
        return self.X_tensor

    def get_tensor(self):
        self.capture_data()
        return self.preprocess_eeg()</content>
<parameter name="filePath">/workspaces/Avatar-Aidan-Fork/predictions_local/neurosityprocessor.py