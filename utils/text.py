from typing import List, Callable
import string
import numpy as np
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

class SparseTextProcessor:

    def __init__(self,
                 tokenizer: Callable[[str], List[str]],
                 lower_case: bool = True,
                 stop_words: List[str] = None,
                 stemmer: Callable[[str], str] = None):
        
        self.tokenizer = tokenizer
        self.lower_case = lower_case
        self.stop_words = stop_words
        self.stemmer = stemmer

    def _convert_to_lower_case(self, texts: List[str]) -> List[str]:

        return [text.lower() for text in texts]
    
    def _remove_punctuation(self, texts: List[str]) ->  List[str]:

        puct_table = str.maketrans({key: None for key in string.punctuation})
        
        return [text.translate(puct_table) for text in texts]
    
    def _tokenize(self, texts: List[str]) -> List[List[str]]:

        tokenized_texts = [self.tokenizer(text) for text in texts]

        if self.stop_words:

            no_stop_words_tokens = []
            
            for tokens in tokenized_texts:
                no_stop_words_tokens.append(
                    [token for token in tokens if token not in self.stop_words]
                )

            tokenized_texts = no_stop_words_tokens

        return tokenized_texts

    def _stemming(self, tokenized_texts: List[List[str]]) -> List[List[str]]:

        processed_tokens = []

        for tokens in tokenized_texts:
            proc_tokens = [self.stemmer(token) for token in tokens]
            processed_tokens.append(proc_tokens)
        
        return processed_tokens

    def process(self, texts: List[str]) -> List[str]:

        # Lower case
        if self.lower_case:
            texts = self._convert_to_lower_case(texts)

        # Remoção de pontuação
        texts = self._remove_punctuation(texts)

        # Tokenização
        text_tokens = self._tokenize(texts)

        # Stemming
        if self.stemmer:
            text_tokens = self._stemming(text_tokens)
        
        # Reconstrução das strings
        texts_str = [" ".join(tokens) for tokens in text_tokens]

        return texts_str

def plot_wordclouds(text: str, title: str, **kargs) -> None:

    sw = stopwords.words("portuguese")

    wc = WordCloud(stopwords=sw, **kargs)
    cloud = wc.generate(text)
    ax = plt.imshow(cloud, interpolation='bilinear').axes
    ax.grid(False)
    ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
        