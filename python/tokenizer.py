"""
Fari-llama Tokenizer
Wraps TinyLlama's BPE Tokenizer for:
- encode(text) -> list of token IDs
- decode(ids) -> text

This is the refrenece implementation
The c++ tokenizer must produce identical output
"""
from transformers import AutoTokenizer
class Tokenizer:
    def __init__(self, model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(self, text: str, add_special_tokens:bool = True)-> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens=True)-> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def vocab_size(self)-> int:
        return self.tokenizer.vocab_size

if __name__ == "__main__":
    print(f"Loading TinyLlama Tokenizer")
    tokenizer = Tokenizer()

    # Test1 - Encode/Decode
    text = "Hello, my name is Fari"
    ids = tokenizer.encode(text)
    recovered = tokenizer.decode(ids)

    print(f"Text: {text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {recovered}")
    print(f"Roundtrip: {'PASS' if text == recovered else 'FAILED'}\n")

    # Test2 - Vocab size
    print(f"Vocab Size: {tokenizer.vocab_size()}\n")

    #Test3 - Subword Decomp
    words = ["unbelieveable", "finetuning", "Farillama", "ChatGPT"]
    print("Subword: ")

    for word in words:
        tokens = tokenizer.tokenizer.convert_ids_to_tokens(tokenizer.encode(word, False))
        print(f" {word:20} -> {tokens}")