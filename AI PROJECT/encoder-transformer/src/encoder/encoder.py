import torch  # PyTorch main library for tensor operations
import torch.nn as nn  # PyTorch neural network module
import math  # Standard Python math library for mathematical operations

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps  # Small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias parameter

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.
        Args:
            x: Input tensor of shape (..., features)
        Returns:
            Normalized tensor of the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding lookup table

    def forward(self, x):
        """
        Converts input token indices to embeddings and scales them.
        Args:
            x: Tensor of token indices (batch_size, seq_len)
        Returns:
            Embedded tensor (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        pe = torch.zeros(max_seq_len, d_model)  # Positional encoding matrix
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # Position indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Frequency terms
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)  # Register as buffer (not a parameter)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads  # Dimensionality per head
        self.num_heads = num_heads  # Number of attention heads
        self.w_q = nn.Linear(d_model, d_model)  # Linear layer for queries
        self.w_k = nn.Linear(d_model, d_model)  # Linear layer for keys
        self.w_v = nn.Linear(d_model, d_model)  # Linear layer for values
        self.w_o = nn.Linear(d_model, d_model)  # Output linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout for attention weights

    def attention(self, query, key, value, mask=None):
        """
        Computes scaled dot-product attention.
        Args:
            query, key, value: Projected input tensors (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor for attention (batch_size, 1, 1, seq_len)
        Returns:
            Weighted sum and attention weights.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Ensure mask shape is broadcastable to scores
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v, mask=None):
        """
        Applies multi-head attention to the input.
        Args:
            q, k, v: Input tensors (batch_size, seq_len, d_model)
            mask: Optional mask tensor (batch_size, 1, 1, seq_len)
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = q.size(0)
        # Project and reshape input for multi-head attention
        query = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Compute attention
        x, attn = self.attention(query, key, value, mask)
        # Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization(features)  # Layer normalization
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    def forward(self, x, sublayer):
        """
        Applies a residual connection followed by layer normalization and dropout.
        Args:
            x: Input tensor
            sublayer: Function/layer to apply to normalized input
        Returns:
            Output tensor after residual connection
        """
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.dropout = nn.Dropout(dropout)        # Dropout
        self.linear_2 = nn.Linear(d_ff, d_model)  # Second linear layer

    def forward(self, x):
        """
        Applies a two-layer feed-forward network with ReLU and dropout.
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)  # Self-attention layer
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)           # Feed-forward network
        # Two residual connections: one for attention, one for feed-forward
        self.residuals = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        Forward pass for a single encoder block.
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Padding mask (batch_size, 1, 1, seq_len)
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Apply self-attention with residual connection
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, mask))
        # Apply feed-forward with residual connection
        x = self.residuals[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Stack of encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)  # Final layer normalization

    def forward(self, x, mask):
        """
        Forward pass for Transformer Encoder.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) - input sequence representations
            mask: Padding mask of shape (batch_size, 1, 1, seq_len) - indicates valid positions
        Returns:
            Output of shape (batch_size, seq_len, d_model) for use in downstream tasks.
        """
        # Pass input through each encoder block
        for layer in self.layers:
            x = layer(x, mask)
        # Apply final normalization
        x = self.norm(x)
        return x  # Return the full sequence, not pooled
