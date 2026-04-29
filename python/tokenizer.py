"""
Fari-llama Tokenizer
Wraps TinyLlama's BPE Tokenizer for:
- encode(text) -> list of token IDs
- decode(ids) -> text

This is the refrenece implementation
The c++ tokenizer must produce identical output
"""

class Tokenizer:
    def __init__(self, model_path: str):
        #TODO: load vocabulary
        pass

    def encode(self, text: str)-> list[int]:
        #TODO: convert text to tokenIDs
        pass

    def decode(self, ids: list[int])-> str:
        #TODO: conver tokenIDs to text
        pass

if __name__ == "__main__":
    tokenizer = Tokenizer("models/tokenizer.model")
    text = "Hello, my name is Fari"
    ids = tokenizer.encode(text)
    print(f"Text:{text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")