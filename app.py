from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip

# Load the trained LSTM model
model = tf.keras.models.load_model("enhanced_lstm_model_full_dataset (3).h5")

# Features used during training
features = [
    "CPU cores", "CPU capacity provisioned [MHZ]", "CPU usage [MHZ]", "CPU usage [%]",
    "Memory capacity provisioned [KB]", "Memory usage [KB]", "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]", "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
    "CPU Utilization Per Core", "Memory Utilization [%]", "Disk Total Throughput [KB/s]", "Network Total Throughput [KB/s]"
]

# Auto-scaling logic
def auto_scaling_decision(pred):
    cpu = pred[0][3]  # CPU usage [%]
    mem = pred[0][11]  # Memory Utilization [%]
    
    if cpu > 0.8 and mem > 0.8:
        return "Scale Up: High CPU and Memory usage"
    elif cpu < 0.3 and mem < 0.3:
        return "Scale Down: Low CPU and Memory usage"
    else:
        return "Maintain: Resource usage is optimal"

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    decision = None

    if request.method == "POST":
        try:
            # Read input CSV file from form
            uploaded_file = request.files["file"]
            if uploaded_file.filename != '':
                df = pd.read_csv(uploaded_file, delimiter=";", skipinitialspace=True)
                df.columns = df.columns.str.strip()

                if "Timestamp [ms]" in df.columns:
                    df["Timestamp [ms]"] = pd.to_datetime(df["Timestamp [ms]"], unit='ms')
                    df = df.sort_values("Timestamp [ms]").reset_index(drop=True)

                # Feature engineering
                df["CPU Utilization Per Core"] = df["CPU usage [MHZ]"] / df["CPU capacity provisioned [MHZ]"]
                df["Memory Utilization [%]"] = df["Memory usage [KB]"] / df["Memory capacity provisioned [KB]"]
                df["Disk Total Throughput [KB/s]"] = df["Disk read throughput [KB/s]"] + df["Disk write throughput [KB/s]"]
                df["Network Total Throughput [KB/s]"] = df["Network received throughput [KB/s]"] + df["Network transmitted throughput [KB/s]"]

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                scaler = MinMaxScaler()
                df[features] = scaler.fit_transform(df[features])

                # Sequence creation
                seq_length = 5
                def create_sequences(data, seq_length=5):
                    sequences = []
                    for i in range(len(data) - seq_length):
                        sequences.append(data[i:i + seq_length])
                    return np.array(sequences)

                X_input = create_sequences(df[features].values, seq_length)

                if len(X_input) > 0:
                    last_seq = X_input[-1].reshape(1, seq_length, len(features))
                    pred = model.predict(last_seq)
                    predictions = [round(float(x), 4) for x in pred[0]]

                    decision = auto_scaling_decision(pred)
                else:
                    predictions = []
                    decision = "Insufficient data for prediction."

        except Exception as e:
            predictions = []
            decision = f"Error: {str(e)}"

    return render_template("index.html", predictions=predictions, features=features, decision=decision)

if __name__ == "__main__":
    app.run(debug=True)
