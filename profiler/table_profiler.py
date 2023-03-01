from nltk.corpus import stopwords
from tqdm import tqdm
import collections
import os
import pandas as pd
import re
import hyperloglog
import spacy

class table_profiler(object):

  def __init__(self, table_names=None, spacy_nlp_model='en_core_web_sm'):
    self.profiled_ids = []
    self.filtered_ids = []
    self.other_count = 0 #non-string columns
    self.string_count = 0
    self.approx_lengths = {}
    self.table_names = table_names # limit profile to only the list of names provided here 
    self.spacy_nlp_model = None
    if spacy_nlp_model is not None:
      self.spacy_nlp_model = spacy.load(spacy_nlp_model, disable=['parser', 'ner'])
      self.spacy_nlp_model.max_length = 50000000
    self.stop_words = stopwords.words('english')
    self.processed = False
    self.max_word_limit = 50000 # hardcoding max words in a content list

  # Define functions for stopwords, bigrams, trigrams and lemmatization
  def _remove_stopwords(self, text):
    if text is None or len(text)==0: return []
    text = [word for word in text if word not in self.stop_words]  
    if len(text) < 5000: return text # small sized
    ## cut frequently appearing words
    max_length = 0.0
    hist = collections.Counter(text)
    if len(hist) < 50: return text # not worth filtering
    most_common = [w for w, freq in hist.most_common(int(0.05*len(hist)))] # top 5 %
    text = [w for w in text if w not in most_common]
    return text

  ## Too slow on long columns
  def _lemmatize(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    if text is None: return []
    if self.spacy_nlp_model is None: return text 
    doc = self.spacy_nlp_model(" ".join(text), disable=['parser', 'ner'])
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

  ## break the text at puntuations, camel cases, lower case it
  def _separate_text(self, text):
    if text is None or len(text)==0: return []
    text = re.findall('[A-Z]?[a-z]+', text)
    return [w.lower() for w in text if len(w)>2 and len(w)<20] # not allowing big words such as gene sequences

  ## break the text at puntuations, camel cases, lower case it, remove stopwords, and lemmatize
  def _nlp(self, texts):
    texts = self._remove_stopwords(texts)
    texts = self._lemmatize(texts, allowed_postags=['NOUN'])
    return texts

  def _error_os_walk(self, e):
    tqdm.write(f'Error in file, {e}')

  ## yields (tblname, colname, is_string, colvalues)
  def process_dir(self, dir_path, sep='\t'):
    tqdm.write(f"Processing directory {dir_path} for table profiling")
    chunksize = 10 ** 6
    for root, dirs, files in os.walk(dir_path, onerror=self._error_os_walk):
     for name in tqdm(files):
      if self.table_names is not None and name not in self.table_names:
        continue ## skip the table
      try:
        ## TODO: Handle chunkwise
        tbl_data = pd.read_csv(os.path.join(dir_path, name), sep=sep, header=0, engine='python', error_bad_lines=False)
      except Exception as e:
        tqdm.write(f'Error reading file: {name}, msg: {e}')
        continue
      for col in tbl_data.columns:
        id = name + ',' + col
        tblname_list = self._separate_text(name)
        colname_list = self._separate_text(col)
        content_list = []
        dtype = tbl_data.dtypes[col]
        is_string = False
        cardinality = 0
        if dtype.kind in 'biufc': ## binary, integer, unsigned, float, or complex
          self.other_count += 1
        else:
          self.string_count += 1
          is_string = True
          # process column content
          content_list = [self._separate_text(t) for t in tbl_data[col].fillna('').to_numpy(dtype='str')]
          content_list = [w for lst in content_list for w in lst] #flattened
          content_list = self._nlp(content_list)[:self.max_word_limit]
          # approaximate length
          hll = hyperloglog.HyperLogLog(0.01)
          [hll.add(p) for p in content_list]
          cardinality = len(hll)
          self.approx_lengths[id] = cardinality
        self.profiled_ids.append(id)

        yield id, tblname_list, colname_list, is_string, content_list, cardinality

    self.processed = True

  def get_ids(self):
    assert self.processed
    return self.profiled_ids

  def num_filtered_ids(self):
    assert self.processed
    return len(self.filtered_ids)

  def num_string_columns(self):
    assert self.processed
    return self.string_count

  def num_other_columns(self):
    assert self.processed
    return self.other_count

  def get_cardinalities(self):
    assert self.processed
    return list(self.approx_lengths.values())

  def get_cardinality(self, id):
    assert self.processed
    assert id in self.approx_lengths
    return self.approx_lengths[id]

if __name__ == "__main__":
  p = table_profiler(None)
  for id, tbl, col, is_string, data, card in p.process_dir('../inputs/mlopen-tables', ','):
    print(f'id: {id}, tbl: {tbl}, col: {col}, data: {data[:2]}')
  print(f'filtered data: {p.num_filtered_ids()}')
  print(f'profiled data: {len(p.get_ids())}')
  import numpy as np
  card = p.get_cardinalities()
  card_np = np.array(card)
  print(f'Mean card: {card_np.mean()}, std card: {card_np.std()}')
  print(f'num strings: {p.num_string_columns()}')

