from nltk.corpus import stopwords
from tqdm import tqdm
import collections
import os
import pandas as pd
import re
import hyperloglog
import spacy

class table_profiler_lm(object):

  def __init__(self, table_names=None, 
          spacy_nlp_model='en_core_web_sm', 
          tokenizer=None,
          max_tuples_per_table=0):
    self.profiled_ids = []
    self.filtered_ids = []
    self.other_count = 0 #non-string columns
    self.string_count = 0
    self.approx_lengths = {}
    self.is_string_type = {}
    self.table_names = table_names # limit profile to only the list of names provided here 
    self.spacy_nlp_model = None
    if spacy_nlp_model is not None:
      self.spacy_nlp_model = spacy.load(spacy_nlp_model, disable=['parser', 'ner'])
      self.spacy_nlp_model.max_length = 50000000
    self.stop_words = stopwords.words('english')
    self.vocab = tokenizer.get_vocab() if tokenizer is not None else None
    self.processed = False
    self.max_word_limit = 50000 # hardcoding max words in a content list
    self.max_tuples_per_table = max_tuples_per_table

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

  ## break each word into pieces in order to avoid out of vocabulary words
  def _word_pieces(self, words):
      if not self.vocab:
          return words
      tokens = []
      for word in words:
        while len(word) > 0:
          i = len(word)
          while i > 0 and word[:i] not in self.vocab:
            i -= 1
          if i == 0:
            tokens.append("<unk>")
            continue
          tokens.append(word[:i])
          word = word[i:]
      return tokens

  ## break the text at puntuations, camel cases, lower case it
  def _separate_text(self, text):
    if text is None or len(text)==0: return []
    #text = re.findall('[A-Z]?[a-z]+', text)
    text = re.findall('[A-Z][a-z]+|[0-9A-Z]+(?=[A-Z][a-z])|[0-9A-Z]{2,}|[a-z0-9]{2,}|[a-zA-Z0-9]', text)
    #text = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', text)
    #print('text: ', text)
    return self._word_pieces([w.lower() for w in text if len(w)>1 and len(w)<20]) # not allowing big words such as gene sequences

  ## break the text at puntuations, camel cases, lower case it, remove stopwords, and lemmatize
  def _nlp(self, texts):
    texts = self._remove_stopwords(texts)
    #texts = self._lemmatize(texts, allowed_postags=['NOUN'])
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
      tblname_list = self._separate_text(name)

      try:
        ## TODO: Handle chunkwise
        tbl_data = pd.read_csv(os.path.join(dir_path, name), sep=sep, header=0, engine='python', error_bad_lines=False)
      except Exception as e:
        tqdm.write(f'Error reading file: {name}, msg: {e}')
        continue

      str_cols = []
      colname_list = []
      for col in tbl_data.columns:
        #if tbl_data.dtypes[col].kind not in 'biufc':
          str_cols.append(col)
          colname_list.append(self._separate_text(col))
      if len(colname_list) == 0:
          print('No columns when cols: ', str_cols)
      if self.max_tuples_per_table > 0:
          tbl_data = tbl_data.head(self.max_tuples_per_table)
      for row in tbl_data.itertuples(): 
        id = name + ',' + str(row.Index)
        #colname_list = [str(row.Index)] #self._separate_text(col)
        content_list = []
        cardinality = 0
        for col in str_cols:
          # process column content
          val = str(tbl_data[col][row.Index])
          #print('val: ', val)
          #if not pd.isna(val):
          content_list.append(self._separate_text(val))

        #content_list = [w for lst in content_list for w in lst] #flattened
        #content_list = self._nlp(content_list)[:self.max_word_limit]
        self.string_count += 1
        self.is_string_type[id] = True
        is_string = True

        # approaximate length
        hll = hyperloglog.HyperLogLog(0.01)
        #[hll.add(p) for p in content_list]
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

  def is_string_column(self, id):
    assert self.processed
    assert id in self.is_string_type
    return self.is_string_type[id]

