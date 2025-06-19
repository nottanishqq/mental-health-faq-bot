import nltk
import pandas as pd
import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import os
import logging
import pickle
import argparse
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _check_nltk_data():
    required_packages = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }
    for path, package_id in required_packages.items():
        try:
            nltk.data.find(path)
            logging.info(f"NLTK package '{package_id}' found.")
        except LookupError:
            logging.warning(f"NLTK package '{package_id}' not found. Downloading...")
            nltk.download(package_id)

class FaqBotTrainer:
    def __init__(self, dataset_path, vector_size=100, window=5, min_count=1, model_dir='model_assets'):
        self.dataset_path = dataset_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model_dir = model_dir
        
        self.w2v_path = os.path.join(self.model_dir, 'word2vec.model')
        self.vectors_path = os.path.join(self.model_dir, 'question_vectors.npy')
        self.df_path = os.path.join(self.model_dir, 'processed_data.pkl')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

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

    def _sent2vec(self, w2v_model, tokens):
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if not vectors: return np.zeros(w2v_model.vector_size)
        return np.mean(vectors, axis=0)
    
    def train_and_save(self):
        logging.info("Starting new training process...")
        try:
            df = pd.read_csv(self.dataset_path)
            if 'input' in df.columns and 'output' in df.columns:
                df.rename(columns={'input': 'Question', 'output': 'Answer'}, inplace=True)
            df.dropna(subset=['Question', 'Answer'], inplace=True)
            if 'Question' not in df.columns or 'Answer' not in df.columns:
                raise ValueError("CSV must have 'Question'/'Answer' or 'input'/'output' columns.")
        except FileNotFoundError:
            logging.error(f"Dataset file not found at '{self.dataset_path}'. Cannot train.")
            raise
        
        logging.info("Preprocessing all questions...")
        df['Tokenized_Questions'] = df['Question'].apply(self._preprocess_text)
        
        logging.info("Training Word2Vec model...")
        w2v_model = Word2Vec(
            sentences=df['Tokenized_Questions'].tolist(),
            vector_size=self.vector_size, window=self.window,
            min_count=self.min_count, workers=os.cpu_count() or 1
        )
        
        logging.info("Generating question vectors...")
        question_vectors = np.array([self._sent2vec(w2v_model, tokens) for tokens in df['Tokenized_Questions']])

        logging.info(f"Saving model assets to '{self.model_dir}'...")
        os.makedirs(self.model_dir, exist_ok=True)
        
        w2v_model.save(self.w2v_path)
        
        np.save(self.vectors_path, question_vectors)

        with open(self.df_path, 'wb') as f:
            pickle.dump(df[['Question', 'Answer']].to_dict('records'), f)
        
        logging.info("Training complete. Artifacts saved successfully.")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = "Train the FAQ Bot model from a CSV Dataset.",
      formatter_class = argparse.ArgumentDefaultsHelpFormatter      
  )

  parser.add_argument(
      "dataset_path",
      type = str,
      help = "Path to the input CSV dataset for training. The CSV must contain 'Question' and 'Answer' columns."
  )

  args = parser.parse_args()
  _check_nltk_data()
  
  trainer = FaqBotTrainer(dataset_path=args.dataset_path)
  trainer.train_and_save()
  print("\n✅ Training finished. The `model_assets` directory is ready for your application.")
