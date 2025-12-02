python -m src.tokenization.build_corpus

python -m src.tokenization.train_word
python -m src.tokenization.train_bpe
python -m src.tokenization.train_unigram

python -m src.training.train word transformer
python -m src.training.train bpe transformer
python -m src.training.train unigram transformer
python -m src.training.train byte transformer


python -m src.training.train word lstm
python -m src.training.train bpe lstm
python -m src.training.train unigram lstm
python -m src.training.train byte lstm