#pragma once 
#include <string>
#include <vector>
#include <unordered_map>

namespace farillama{

    class Tokenizer{
        public:
            // Constructor - Loads vocabulary from file
            explicit Tokenizer(const std::string& vocab_path);

            // Encode text to tokenIDs
            std::vector<int> encode(const std::string& text) const;

            // Decode tokenIDs back to text 
            std::string decode(const std::vector<int>& ids) const;

            // Vocabulary size
            int vocabSize() const;

            // Check if tokenID is end of sequence
            bool isEOS(int token_id) const;

        private:
            std::unordered_map<std::string, int> token_to_id;
           std::unordered_map<std::string, int> id_to_token;
           int vocab_size;
           int eos_token_id;
           int bos_token_id;
    };
}

