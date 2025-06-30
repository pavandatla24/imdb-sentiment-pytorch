import pandas as pd
import torch
import spacy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load spaCy tokenizer
nlp = spacy.load("en_core_web_sm")

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab or self.build_vocab(texts)
        self.max_len = max_len

    def build_vocab(self, texts, min_freq=2):
        token_freq = {}
        for text in texts:
            for token in nlp(text.lower()):
                if token.is_alpha:
                    token_freq[token.text] = token_freq.get(token.text, 0) + 1
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for token, freq in token_freq.items():
            if freq >= min_freq:
                vocab[token] = len(vocab)
        return vocab

    def encode(self, text):
        tokens = [token.text.lower() for token in nlp(text) if token.is_alpha]
        ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        return torch.tensor(ids)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx])

def load_data(file_path, test_size=0.2):
    df = pd.read_csv(file_path)
    df = df.dropna()
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=test_size, random_state=42)
    train_data = IMDBDataset(X_train.tolist(), y_train.tolist())
    test_data = IMDBDataset(X_test.tolist(), y_test.tolist(), vocab=train_data.vocab)
    return train_data, test_data, train_data.vocab
