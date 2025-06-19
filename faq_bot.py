import os
import string
import logging
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

class FaqBot:
    def __init__(self, model_dir='model_assets'):
        self.model_dir = model_dir
        self.w2v_path = os.path.join(self.model_dir, 'word2vec.model')
        self.vectors_path = os.path.join(self.model_dir, 'question_vectors.npy')
        self.df_path = os.path.join(self.model_dir, 'processed_data.pkl')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.w2v_model = None
        self.question_vectors = None
        self.qa_data = None
    
    def load(self):
        logging.info(f"Loading model assets from '{self.model_dir}'...")
        if not os.path.isdir(self.model_dir):
            logging.error(f"Model directory not found at '{self.model_dir}'. Please run train.py first.")
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        try:
            self.w2v_model = Word2Vec.load(self.w2v_path)
            self.question_vectors = np.load(self.vectors_path)
            with open(self.df_path, 'rb') as f:
                self.qa_data = pickle.load(f)
            logging.info("✅ Bot loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model assets. Error: {e}")
            raise

    def _get_pos(self, tag):
        if tag.startswith('J'): 
          return wordnet.ADJ
        elif tag.startswith('V'): 
          return wordnet.VERB
        elif tag.startswith('N'): 
          return wordnet.NOUN
        elif tag.startswith('R'): 
          return wordnet.ADV
        else: 
          return wordnet.NOUN

    def _preprocess_text(self, text):
        if not isinstance(text, str): return []
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        pos_tags = nltk.pos_tag(tokens)
        return [self.lemmatizer.lemmatize(word, pos=self._get_pos(tag)) for word, tag in pos_tags]

    def _sent2vec(self, tokens):
        vectors = [self.w2v_model.wv[word] for word in tokens if word in self.w2v_model.wv]
        if not vectors: return np.zeros(self.w2v_model.vector_size)
        return np.mean(vectors, axis=0)

    def get_answer(self, user_question, threshold=0.7):
        if not self.w2v_model:
            raise RuntimeError("Bot is not loaded. Please call the `load()` method first.")
            
        user_tokens = self._preprocess_text(user_question)
        if not user_tokens:
            return "I'm sorry, I couldn't understand your question. Please try rephrasing."
        
        user_vector = self._sent2vec(user_tokens).reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.question_vectors)
        
        best_match_idx = np.argmax(similarities)
        max_similarity = np.max(similarities)
        
        logging.debug(f"Max similarity: {max_similarity:.4f}")

        if max_similarity > threshold:
            return self.qa_data[best_match_idx]['Answer']
        else:
            return "I'm sorry, I don't have a relevant answer. Could you ask in a different way?"