from annoy import AnnoyIndex
from gensim.models.fasttext import FastText #load_facebook_vectors
from gensim.models.keyedvectors import FastTextKeyedVectors
import numpy as np
import os, re
import csv
from tqdm import tqdm

class WEM():
  def __init__(self, ft_model_path, dim):
    tqdm.write(f'Loading wem model from file {ft_model_path}')
    self.ft_model = FastTextKeyedVectors.load(ft_model_path, mmap='r') #load_facebook_vectors(ft_model_path)
    self.dim = dim

  def __call__(self, word_list):
    assert self.ft_model is not None
    assert word_list is not None
    vectors = [self.ft_model.wv[w] for w in word_list]
    return np.nanmean(vectors, axis=0)

  def get_vector(self, word_list):
    return self.__call__(word_list)

  def get_dim(self):
    return self.dim

  def __delete__(self, instance):
    del self.ft_model


class WEMIndexer(object):
  def __init__(self, index_name, wem_model, mode='table'):
    self.index_name = index_name
    self.wem_model = wem_model
    self.mode = mode
    self.committed = False
    self.id_map = {}

  def create_index(self):
    # delete index if exists
    if os.path.isfile(self.index_name):
      print(f'Deleting WEM index {self.index_name}')
      os.remove(self.index_name)
    self.annoy_index = AnnoyIndex(self.wem_model.get_dim(), 'euclidean')
    self.id_count = 0

  def index_doc(self, id, content):
    assert self.wem_model is not None
    self.id_map[self.id_count] = id
    self.annoy_index.add_item(self.id_count, self.wem_model(content))
    self.id_count += 1

  def commit_index(self):
    self.annoy_index.build(10)
    self.annoy_index.save(self.index_name)
    ## write id map to file
    with open(self.index_name + '-idmap', 'w') as f:
      csvf = csv.writer(f)
      for k, v in self.id_map.items():
        csvf.writerow([k, v])
    tqdm.write(f'Saved wem index to disk')
    self.committed = True

  def load_index(self):
    assert os.path.isfile(self.index_name)
    self.annoy_index = AnnoyIndex(self.wem_model.get_dim(), 'euclidean')
    self.annoy_index.load(self.index_name)
    ## load id map from file
    with open(self.index_name + '-idmap', 'r') as f:
      csvf = csv.reader(f)
      for row in csvf:
        self.id_map[int(row[0])] = row[1]
    tqdm.write(f'Read wem index from disk')
    self.committed = True

  def search(self, query, limit):
    assert limit>0
    assert self.committed
    if query is None or len(query) == 0: return []
    query_v = self.wem_model.get_vector(query)
    (results, distances) = self.annoy_index.get_nns_by_vector(query_v, 2*limit, include_distances=True) # asking for 2x results
    if results is None: return []
    #tqdm.write(f'Search for query {query} returned {len(results)} semantic matches')

    hits_dict = {}
    for r, d in zip(results, distances): # asking for twice the results      
      if self.mode == 'table': # combine results by table from keys (table, col)
        r_t = re.split(r',\s*(?![^"]*\"\,)', self.id_map[r])[0] #self.id_map[r].split(',')[0]
        score = hits_dict.get(r_t, 0)
        hits_dict[r_t] = score + (1 / (1 + d))
      else:
        hits_dict[self.id_map[r]] = 1 / (1 + d) # distance to similarity

    return sorted([(k, v) for k, v in hits_dict.items()], key=lambda tup: tup[1], reverse=True)[:limit]


if __name__ == "__main__":
  pass 
