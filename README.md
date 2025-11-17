# Tokenization Strategies for Tiny Language Models  

**02456 Deep Learning – Group 10 (DTU)**

This repository investigates how different tokenization strategies affect the training and performance of small-scale language models.  
We compare **Word**, **BPE**, **Unigram**, and **Byte-level** tokenizers on the **TinyStories** dataset, using both **Transformer** and **LSTM** models.

---

Clone the repository:

```bash
git clone https://github.com/turingzhi/group10_dl_tokenization.git
cd group10_dl_tokenization
```



Init the environment:

```bash
conda activate token
pip install -r requirements.txt
```

Build the corpus

```bash
python -m src.tokenization.build_corpus
```



Build the tokenization

```bash
python -m src.tokenization.train_word
python -m src.tokenization.train_bpe
python -m src.tokenization.train_unigram

```






Then train the models with different tokenizers

```bash
python -m src.training.train <tokenizer> <model_type>
```



Available tokenizers:

word
bpe
unigram
byte



Available models:

transformer
lstm



Examples:

```bash
python -m src.training.train word transformer
```





The results look like:

checkpoints/<tokenizer>_<model_type>_epochX.pt

Example: checkpoints/word_transformer_epoch4.pt



Evaluate the results:

```bash
python -m src.evaluation.eval_model <tokenizer> <model_type> <checkpoint_path>
```

Example:

```bash
python -m src.evaluation.eval_model bpe transformer checkpoints/bpe_transformer_epoch4.pt
```


The results look like:

nll/token = 0.0457
perplexity = 1.0468



All training commands:

```bash
python -m src.training.train word transformer
python -m src.training.train bpe transformer
python -m src.training.train unigram transformer
python -m src.training.train byte transformer



python -m src.training.train word lstm
python -m src.training.train bpe lstm
python -m src.training.train unigram lstm
python -m src.training.train byte lstm
```





