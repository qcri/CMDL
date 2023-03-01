from annoy import AnnoyIndex
import numpy as np
import os
import csv
from tqdm import tqdm

class TrainedEmbeddings():
  def __init__(self, id_list, emb_file_path):
    tqdm.write(f'Loading wem model from file {emb_file_path}')
    self.id_list = id_list
    self.vectors = np.load(emb_file_path) 
    #tqdm.write(f'emb size: {self.vectors.shape}, ids: {len(self.id_list)}')
    assert len(self.id_list) == self.vectors.shape[0]

  def __call__(self, id):
    assert self.id_list is not None
    assert self.vectors is not None
    idx = self.id_list.index(id)
    return self.vectors[idx, :]

  def get_vector(self, id):
    return self.__call__(id)

  def get_dim(self):
    return self.vectors.shape[1]

  def __delete__(self, instance):
    del self.id_list
    del self.vectors


class TrainedEmbeddingsIndexer(object):
  def __init__(self, index_name, emb_object, mode='table'):
    self.index_name = index_name
    self.emb_object = emb_object
    self.mode = mode
    self.committed = False
    self.id_map = {}

  def create_index(self):
    # delete index if exists
    if os.path.isfile(self.index_name):
      print(f'Deleting WEM index {self.index_name}')
    self.annoy_index = AnnoyIndex(self.emb_object.get_dim(), 'euclidean')
    self.id_count = 0

  def index_doc(self, id):
    assert self.emb_object is not None
    self.id_map[self.id_count] = id
    self.annoy_index.add_item(self.id_count, self.emb_object(id))
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
    self.annoy_index = AnnoyIndex(self.emb_object.get_dim(), 'euclidean')
    self.annoy_index.load(self.index_name)
    ## load id map from file
    with open(self.index_name + '-idmap', 'r') as f:
      csvf = csv.reader(f)
      for row in csvf:
        self.id_map[int(row[0])] = row[1]
    tqdm.write(f'Read wem index from disk')
    self.committed = True

  def search(self, query_v, limit):
    assert limit>0
    assert self.committed
    (results, distances) = self.annoy_index.get_nns_by_vector(query_v, 2*limit, include_distances=True) # asking for 2x results
    if results is None: return []
    #tqdm.write(f'Search for query {query} returned {len(results)} semantic matches')

    hits_dict = {}
    for r, d in zip(results, distances): # asking for twice the results      
      if self.mode == 'table': # combine results by table from keys (table, col)
        r_t = self.id_map[r].split(',')[0]
        score = hits_dict.get(r_t, 0)
        hits_dict[r_t] = (score + (1 / (1 + d)))
      else:
        hits_dict[self.id_map[r]] = 1 / (1 + d) # distance to similarity

    return sorted([(k, v) for k, v in hits_dict.items()], key=lambda tup: tup[1], reverse=True)[:limit]

    #return [(self.id_map[hit], 1.0) for hit in results]

if __name__ == "__main__":
  pass 
