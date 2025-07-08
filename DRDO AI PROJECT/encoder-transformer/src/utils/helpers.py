# Function to create a padding mask for a sequence tensor.
# Args:
#   seq: Tensor of shape (batch_size, seq_len), where 0 indicates a padding token.
# Returns:
#   A boolean mask of shape (batch_size, 1, 1, seq_len) suitable for use in attention mechanisms.
def create_padding_mask(seq): 
    # (seq != 0) is True for non-padding tokens, False for padding (0) tokens.
    # unsqueeze(1) and unsqueeze(2) add singleton dimensions for broadcasting.
    return (seq != 0).unsqueeze(1).unsqueeze(2)
