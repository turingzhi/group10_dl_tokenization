# Exploring Tokenization Strategies for Large Language Models  

**02456 Deep Learning – Group 10 (DTU)**

This repository investigates how different tokenization strategies affect the training and performance of small-scale language models.  
We compare **Word**, **BPE**, **Unigram**, and **Byte-level** tokenizers on the **WikiText2** dataset, using both **Transformer** and **LSTM** models.

---

## **1. Setup**

### **Clone the repository**

```bash
git clone https://github.com/turingzhi/group10_dl_tokenization.git
cd group10_dl_tokenization
```

**Create the Conda environment**

```bash
conda create -n token python=3.10
conda activate token
pip install -r requirements.txt
```

## **2. Data Preparation**

**Build the corpus**

```bash
python -m src.tokenization.build_corpus
```

**Train tokenizers**

```bash
python -m src.tokenization.train_word
python -m src.tokenization.train_bpe
python -m src.tokenization.train_unigram

```

## **3. Training (Command Line)**

**Transformer models**


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

## **4. Training (Jupyter Notebook)**

**Register a notebook kernel**

```bash
python -m ipykernel install --user --name token --display-name "Python (token)"
```

**Start a remote Jupyter server**

```bash
jupyter notebook \
  --no-browser \
  --port=8890 \
  --ip=n-62-12-19 \
  --ServerApp.token='' \
  --ServerApp.password=''
```


### **Connect from notebook / VS Code**

1. Open Jupyter Notebook or VS Code
2. Select **Existing Jupyter Server** from **select kernel** from top right corner
3. Enter: `http://n-62-12-19:8890`
4. Select kernel: `Python (token)`
5. press `enter`(no inputs from now and afterwards) to get access to the server if it asks for password

Now you can run all model training inside the notebook environment.



## **5. Output**

- Tokenizer files → data/
- Metrics (loss, perplexity, etc.) → results/
- Model checkpoints → checkpoints/



------
