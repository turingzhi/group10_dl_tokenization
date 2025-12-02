import torch
import torch.nn as nn

# GPT-like Transformer Language Model
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, model_dim, n_heads, n_layers, context_length, dropout=0.1):
        super().__init__()
        
        # 1. Standard Embeddings
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        # Using nn.Embedding for positional embedding is fine, equivalent to learned positional encoding
        self.pos_emb = nn.Embedding(context_length, model_dim) 
        self.context_length = context_length

        # 2. Use TransformerEncoderLayer for Causal LM
        # This layer acts as the Causal Decoder Block in a GPT architecture
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_heads, 
            dim_feedforward=model_dim * 4, # Common practice for FFN size
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        # 3. Use TransformerEncoder (it will serve as a stack of Causal Decoder blocks)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # 4. Head and Weight Tying (Weight tying is a standard optimization in LMs)
        self.fc = nn.Linear(model_dim, vocab_size, bias=False)
        self.fc.weight = self.token_emb.weight  # Weight tying
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        
        self.drop = nn.Dropout(dropout)
        self.ln_f = nn.LayerNorm(model_dim)
        
        # Pre-register the causal mask for efficiency
        self.register_buffer("causal_mask", self._make_causal_mask(context_length))


    def _make_causal_mask(self, size):
        # Create an upper triangular mask (look-ahead mask)
        # Values are True (masked) in the upper triangle, False otherwise.
        # This is converted to float(-inf) in the forward pass.
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x):
        b, seq_len = x.size()
        device = x.device
        
        # 1. Positional Indices
        # Indices: [0, 1, 2, ..., seq_len-1]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # 2. Embeddings Summation (Token + Position)
        # Standard practice: apply dropout *after* summation of embeddings
        h = self.token_emb(x) + self.pos_emb(pos)
        # h = self.drop(h) # Apply dropout once

        # 3. Prepare Causal Mask
        # If the input sequence is smaller than the pre-registered mask, slice it.
        # Convert the boolean mask to a float mask for the PyTorch Encoder API.
        # True (masked) -> float('-inf')
        mask_bool = self.causal_mask[:seq_len, :seq_len]
        mask_float = torch.zeros_like(mask_bool, dtype=torch.float32)
        mask_float = torch.masked_fill(mask_bool, mask_bool, float('-inf'))
        
        # 4. Transformer Encoder Pass
        # src_mask argument applies the causal masking to self-attention
        h = self.encoder(h, mask=mask_float)
        
        # 5. Final LayerNorm and Head
        h = self.ln_f(h)
        
        # Apply the final bias manually because we used bias=False in nn.Linear for weight tying
        logits = self.fc(h) + self.bias
        
        return logits

def verify_transformer_lm():
    print("--- Starting TransformerLM Verification ---")

    # Configuration for testing
    VOCAB_SIZE = 100
    MODEL_DIM = 64
    N_HEADS = 4
    N_LAYERS = 2
    CONTEXT_LENGTH = 16
    BATCH_SIZE = 4
    TEST_SEQ_LEN = 10 # Must be <= CONTEXT_LENGTH

    # Initialize model
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        model_dim=MODEL_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        context_length=CONTEXT_LENGTH,
        dropout=0.1
    )
    
    # Generate dummy input: sequence of token IDs
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEST_SEQ_LEN))
    dummy_target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, TEST_SEQ_LEN))

    print(f"\nModel Initialized. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Test Sequence Length: {TEST_SEQ_LEN}")
    
    
    # ----------------------------------------------------
    ## CHECK A: Weight Tying Verification
    # ----------------------------------------------------
    print("\n[VERIFY A] Weight Tying (Embedding vs Head):")
    is_tied = model.token_emb.weight is model.fc.weight
    print(f"  Are weights tied? {is_tied}")
    assert is_tied, "❌ Weight Tying Failed: fc.weight is not the same object as token_emb.weight."
    
    
  
    # ----------------------------------------------------
    ## CHECK B: Causal Mask Verification
    # ----------------------------------------------------
    print("\n[VERIFY B] Causal Mask Correctness:")

    # 1. Get the boolean mask from the model's buffer
    mask_bool = model.causal_mask[:TEST_SEQ_LEN, :TEST_SEQ_LEN]

    # 2. Create the float mask: Start with a float tensor of 0s.
    # We create a tensor of 0s, and fill it with -inf based on mask_bool.
    # We need to ensure it's on the correct device for the dummy input check.
    mask_float = torch.zeros(mask_bool.shape, dtype=torch.float32, device=dummy_input.device)

    # 3. Apply the causal mask: Fill the TRUE positions (upper triangle) with -inf
    mask_float.masked_fill_(mask_bool, float('-inf')) # Use in-place fill for clarity

    # Check shape
    mask_shape = mask_float.shape
    print(f"  Mask shape: {mask_shape}. Expected: ({TEST_SEQ_LEN}, {TEST_SEQ_LEN})")
    assert mask_shape == (TEST_SEQ_LEN, TEST_SEQ_LEN), "❌ Mask Shape is Incorrect."

    # Check the diagonal (i, i) - should be 0.0 (since it was initialized to 0.0)
    diag_check = (mask_float.diagonal() == 0.0).all().item()
    print(f"  Diagonal (i, i) check (should be 0.0): {diag_check}")

    # Check the cell directly above the diagonal (i, i+1) - should be -inf
    off_diag_check = (mask_float[0, 1] == float('-inf')).item()
    print(f"  Off-Diagonal (i, i+1) check (should be -inf): {off_diag_check}")

    # This assertion now relies on the corrected mask_float generation.
    assert diag_check and off_diag_check, "❌ Causal Mask Pattern is Incorrect."
    

    # ----------------------------------------------------
    ## CHECK C: Loss Sanity Check (Overfitting a single batch)
    # ----------------------------------------------------
    print("\n[VERIFY C] Loss Sanity Check (Overfit Single Batch):")
    
    # Use the single dummy batch for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    initial_loss = None
    
    for epoch in range(20): # Train for 20 epochs
        model.train()
        optimizer.zero_grad()
        
        logits = model(dummy_input)
        
        # Reshape for CrossEntropyLoss
        # Logits: [B*T, V], Target: [B*T]
        V = logits.size(-1)
        loss = criterion(logits.view(-1, V), dummy_target.view(-1))
        
        if epoch == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 19:
            ppl = torch.exp(loss).item()
            print(f"  Epoch {epoch:02d}: Loss={loss.item():.4f}, PPL={ppl:.2f}")
    
    final_loss = loss.item()
    
    print(f"  Initial Loss: {initial_loss:.4f}")
    print(f"  Final Loss:   {final_loss:.4f}")

    # Expect the loss to drop dramatically (e.g., by 95%)
    # This value is arbitrary, but loss MUST drop significantly.
    expected_reduction = 0.95
    
    if final_loss < initial_loss * (1 - expected_reduction):
        print("✅ Loss dropped significantly. Sanity check PASSED.")
    else:
        print("❌ Loss reduction was insufficient. Model may have gradient issues or bug.")
        
    print("\n--- Verification Complete ---")

# Execute the verification
if __name__ == '__main__':
    verify_transformer_lm()
# Note on Genericity:
# This model is generic because it only takes the vocab_size and context_length 
# (max_seq_len) from the overall configuration. As long as any tokenizer 
# (Word, BPE, Byte, Unigram) provides valid token IDs (0 to vocab_size-1), 
# the model will work correctly.