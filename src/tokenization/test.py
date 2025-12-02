# Assuming your ByteTokenizer class is defined above this test function
from byte_tokenizer import ByteTokenizer
def run_byte_integrity_test(tokenizer_wrapper):
    """
    Tests the ByteTokenizer's index mapping, special token handling, 
    and reversibility using diverse text.
    """
    
    # 1. Test Case 1: Standard ASCII text (1 byte per char)
    test_text_ascii = "Hello World!"
    
    # 2. Test Case 2: Multi-byte UTF-8 character (e.g., Euro sign '€' is 3 bytes)
    test_text_utf8 = "Test € Symbol." 
    
    # 3. Test Case 3: Edge case text with spaces and punctuation
    test_text_edge = "   \n Padding & $ % 99" 
    
    tests = [
        ("ASCII Text", test_text_ascii), 
        ("Multi-byte UTF-8", test_text_utf8),
        ("Punctuation/Spaces", test_text_edge)
    ]

    print("\n--- Byte Tokenizer Integrity Test ---")
    
    for name, test_text in tests:
        # Encode (with special tokens)
        encoded_ids = tokenizer_wrapper.encode(test_text, add_special_tokens=True)
        
        # Decode (skip special tokens)
        decoded_text = tokenizer_wrapper.decode(encoded_ids, skip_special_tokens=True)
        
        # --- Verification ---
        is_reversible = (decoded_text == test_text)
        
        # Check special tokens (BOS/EOS)
        is_special_token_count_correct = (
            encoded_ids[0] == tokenizer_wrapper.bos_id and 
            encoded_ids[-1] == tokenizer_wrapper.eos_id and
            len([i for i in encoded_ids if i < tokenizer_wrapper.num_special]) == 2
        )
        
        # Check sequence length for non-special tokens
        expected_len = len(test_text.encode("utf-8"))
        actual_token_len = len(encoded_ids) - 2 # Subtract BOS/EOS
        is_len_correct = (actual_token_len == expected_len)

        print(f"\n🧪 Test: {name}")
        print(f"  Input: '{test_text}'")
        print(f"  Decoded: '{decoded_text}'")
        print(f"  Reversibility: {'✅ SUCCESS' if is_reversible else '❌ FAILED'}")
        print(f"  Special Tokens: {'✅ SUCCESS' if is_special_token_count_correct else '❌ FAILED'}")
        print(f"  Token Count (Bytes): {'✅ SUCCESS' if is_len_correct else f'❌ FAILED (Expected {expected_len}, Got {actual_token_len})'}")
        
    print("\n--- Test Complete ---")


# Example usage (assuming ByteTokenizer is accessible):
tokenizer_instance = ByteTokenizer.load()
run_byte_integrity_test(tokenizer_instance)
