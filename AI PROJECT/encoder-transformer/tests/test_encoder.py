import sys
import os

# Add the project root directory to sys.path so that 'src' can be imported as a package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch  # PyTorch library for tensor operations and neural networks
from src.encoder.encoder import Encoder  # Import the custom Transformer Encoder implementation

def test_encoder():
    # Model hyperparameters
    d_model = 512        # Dimensionality of input/output features
    num_layers = 6       # Number of encoder layers (blocks)
    num_heads = 8        # Number of attention heads in each multi-head attention block
    d_ff = 2048          # Dimensionality of the feed-forward network inside each encoder block
    dropout = 0.1        # Dropout rate for regularization
    seq_len = 10         # Length of the input sequence
    batch_size = 2       # Number of samples in a batch

    # Create a random input tensor simulating a batch of sequences
    # Shape: (batch_size, seq_len, d_model)
    x = torch.rand(batch_size, seq_len, d_model)
    # Create a mask tensor (all ones means no masking)
    # Shape: (batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)

    # Instantiate the Encoder model with the specified hyperparameters
    encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
    # Pass the input and mask through the encoder
    output = encoder(x, mask)

    # Check that the output shape matches the expected shape
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"

# Only run the test when this script is executed directly
if __name__ == "__main__":
    print("Running Encoder tests...")
    test_encoder()
    print("✅ All Encoder tests passed!")