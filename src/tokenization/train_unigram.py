from .unigram_tokenizer import UnigramTokenizer

def main():
    corpus = "data/corpus.txt"
    UnigramTokenizer.train(corpus_path=corpus, vocab_size=5000)

if __name__ == "__main__":
    main()
