import gensim
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import re
import hyperloglog

class text_profiler(object):

  def __init__(self, spacy_nlp_model='en_core_web_sm'):
    self.raw_data = {}
    self.profiled_data = {}
    self.filtered_ids = []
    # Initialize spacy model, keeping only tagger component (for efficiency)
    self.spacy_nlp_model = spacy.load(spacy_nlp_model, disable=['parser', 'ner'])
    self.stop_words = stopwords.words('english')
    self.approx_lengths = {}

  def _sent_to_words(self, sentences):
    for sentence in sentences:
      yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

  # Define functions for stopwords, bigrams, trigrams and lemmatization
  def _remove_stopwords(self, texts):
    return [[word for word in doc if word not in self.stop_words] for doc in texts]

  def _make_bigrams(self, bigram_model, texts):
    return [bigram_model[doc] for doc in texts]

  def _lemmatize(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    for text in texts:
      doc = self.spacy_nlp_model(" ".join(text)) 
      yield [token.lemma_ for token in doc if token.pos_ in allowed_postags]

  def _nlp(self, texts):
    # Remove new line characters 
    data = [re.sub('\s+', ' ', text) for text in texts]  
    # Remove distracting single quotes 
    data = [re.sub("\'", "", sent) for sent in data] 
    # to words
    data_words = list(self._sent_to_words(data))
    # Remove Stop Words
    data_words_nostops = self._remove_stopwords(data_words)
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # Form Bigrams
    data_words_bigrams = self._make_bigrams(bigram_mod, data_words_nostops)
    # Do lemmatization keeping only nouns
    data_lemmatized = list(self._lemmatize(data_words_bigrams, allowed_postags=['NOUN']))
    # Create Dictionary 
    id2word = gensim.corpora.Dictionary(data_lemmatized)  
    # Filter out words that occur less than 1% documents, or more than 30% of the documents.
    id2word.filter_extremes(no_below=int(0.0*len(self.raw_data)), no_above=0.3)
    # Term Document Frequency 
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    # corpus to words
    data_processed = [[[id2word[idx]] * freq for (idx, freq) in c] for c in corpus]
    # flattening
    data_processed = [[i for e in d for i in e] for d in data_processed]
    return data_processed

  def process_dir(self, dir_path):
    tqdm.write(f"Processing directory {dir_path} for text profiling")
    for entry in tqdm(os.scandir(dir_path)):
      if entry.is_dir():
        continue
      name = entry.name
      text = name + '\n' # including file name
      with open(os.path.join(dir_path, name), 'r') as f:
        text = name + f.read()
      self.raw_data[name] = text

    profiled_all = self._nlp(self.raw_data.values())
    #tqdm.write(f'Profilier returned {len(profiled_all)} documents')
    for name, profiled in zip(self.raw_data.keys(), profiled_all):
      if profiled is None or len(profiled) == 0:
        self.filtered_ids.append(name)
      else:
        self.profiled_data[name] = profiled
        # approaximate length
        hll = hyperloglog.HyperLogLog(0.01)
        [hll.add(p) for p in profiled]
        self.approx_lengths[name] = len(hll)
    

  def get_ids(self):
    return self.profiled_data.keys()

  def get_raw(self, tid):
    if tid in self.raw_data:
      return self.raw_data[tid]
    else:
      return []

  def get_profiled(self, tid):
    if tid in self.profiled_data:
      return self.profiled_data[tid]
    else:
      return []

  def num_filtered_ids(self):
    return len(self.filtered_ids)

  def get_cardinalities(self):
    return list(self.approx_lengths.values())

  def get_cardinality(self, id):
    assert id in self.approx_lengths
    return self.approx_lengths[id]

if __name__ == "__main__":
  p = text_profiler('en_core_web_sm')
  p.process_dir('../inputs/mlopen-text/')
  print(f'filtered data: {p.num_filtered_ids()}')
  ids = p.get_ids()
  for i in ids:
    print(f'\n----Doc: {i} \n Profiled: {p.get_profiled(i)}')

  print(f'Total docs: {len(ids)}')
  import numpy as np
  card = p.get_cardinalities()
  card_np = np.array(card)
  print(f'Mean card: {card_np.mean()}, std card: {card_np.std()}')
