from .word_tokenizer import WordTokenizer

def main():
    corpus = "data/corpus.txt"
    WordTokenizer.train(corpus_path=corpus, vocab_size=33000)

if __name__ == "__main__":
    main()
