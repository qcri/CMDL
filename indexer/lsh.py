from datasketch import MinHashLSH, MinHashLSHEnsemble, MinHash
import csv
import os, re
import numpy as np
from tqdm import tqdm

def get_mh(data, num_perm):
  mh = MinHash(num_perm)
  for d in data:
    mh.update(d.encode('utf8'))
  return mh

class LSH():
  def __init__(self, threshold=0.7, num_perm=128):
    self.num_perm = num_perm
    self.threshold = threshold
    self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    self.committed = False

  def index_doc(self, key, data):
    mh = get_mh(data, self.num_perm)
    self.lsh.insert(key, mh)

  def commit_index(self):
    ## nothing to do here
    self.committed = True

  def search(self, q):
    if not self.committed:
      self.commit_index()
    mh = get_mh(q, self.num_perm)
    try:
      return [r for r in self.lsh.query(mh)]
    except:
      return []

class LSHEnsemble():
  def __init__(self, index_name, threshold=0.7, num_perm=128, num_part=32, mode='table'):
    self.index_name = index_name
    self.threshold = threshold
    self.num_perm = num_perm
    self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
    self.index = []
    self.mode = mode
    self.committed = False

  def create_index(self):
    # delete index if exists
    if os.path.isfile(self.index_name):
      tqdm.write(f'deleting existing lsh index {self.index_name}')
      os.remove(self.index_name)

  def index_doc(self, key, data, card):
    mh = get_mh(data, self.num_perm)
    self.index.append((key, mh, card))

  def commit_index(self):
    ## save data to file
    with open(self.index_name, 'w') as f:
      csvf=csv.writer(f)
      for (k, mh, c) in self.index:
        csvf.writerow([k, ",".join(map(str, mh.hashvalues)), c])
      tqdm.write(f'saving new lsh index {self.index_name}')

    self.index_dict = {key: tuple(rest) for key, *rest in self.index}
    if self.index is not None and len(self.index)>0:
      self.lshensemble.index(self.index)
      self.committed = True

  def load_index(self):
    ## load from file
    with open(self.index_name, 'r') as f:
      csvf = csv.reader(f)
      for row in tqdm(csvf):
        self.index.append((row[0], MinHash(num_perm=self.num_perm, hashvalues=np.array(row[1].split(','), dtype=np.uint64)), int(row[2])))
    self.index_dict = {key: tuple(rest) for key, *rest in self.index}
    if self.index is not None and len(self.index)>0:
      tqdm.write(f'Loaded LSHE index from file {self.index_name}')
      self.lshensemble.index(self.index)
      self.committed = True

  def search(self, q, qcard, limit):
    assert self.committed
    if q is None or len(q) == 0:
      return []
    mh = get_mh(q, self.num_perm)
    try:
      results = self.lshensemble.query(mh, qcard)
      hits_dict = {}
      ## approximate score computation
      for r in results:
        (r_mh, r_card) = self.index_dict[r]
        jaccard = r_mh.jaccard(mh)
        contain = (1.0 + r_card/qcard) * jaccard / (1 + jaccard) if qcard != 0 else 0
        contain = min(contain, 1.0)
        if self.mode == 'table': # combine results by table from keys (table, col)
          r_t = re.split(r',\s*(?![^"]*\"\,)', r)[0] #r.split(',')[0]
          score = hits_dict.get(r_t, 0)
          hits_dict[r_t] = score + contain
        else:
          hits_dict[r] = contain
      return sorted([(k, v) for k, v in hits_dict.items()], key=lambda tup: tup[1], reverse=True)[:limit]
    except Exception as e:
      print(e)
      return []
