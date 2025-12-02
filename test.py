import onnxruntime as ort
import numpy as np

# Load the model
session = ort.InferenceSession("energy_gru.onnx")

# Create dummy input (Batch=1, Seq=24, Feat=4)
dummy_input = np.random.randn(1, 24, 4).astype(np.float32)

# Run Inference
inputs = {session.get_inputs()[0].name: dummy_input}
output = session.run(None, inputs)

print(f"ONNX Model Prediction: {output[0][0][0]:.4f}")