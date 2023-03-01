from tqdm import tqdm
from .elastic_search import ElasticIndexer
from .lsh import LSH, LSHEnsemble
from .wem import WEM, WEMIndexer

class table_indexer(object):

  def __init__(self, datalake, lshe_threshold, wem_model_path, wem_dim, mode='table'):
    self.datalake = datalake
    self.lshe = LSHEnsemble(datalake + '-lshe', threshold=lshe_threshold, mode=mode)
    self.eindex = ElasticIndexer(datalake + '-elastic', mode=mode)
    self.wem_model = WEM(wem_model_path, wem_dim)
    self.wemindex = WEMIndexer(datalake + '-wem', self.wem_model, mode=mode)
    self.processed = False

  def index_tables(self, profiler, table_dir, sep):
    self.wemindex.create_index()
    self.eindex.create_index()
    self.lshe.create_index()

    for (id, tblname, colname, is_string, colvalues, card) in profiler.process_dir(table_dir, sep):
      if is_string and len(colvalues) > 0:
        self.lshe.index_doc(id, colvalues, card)
        self.wemindex.index_doc(id, colvalues)
        self.eindex.index_doc(id, ' '.join(tblname), ' '.join(colname), ' '.join(colvalues))
      else:
        self.eindex.index_doc(id, ' '.join(tblname), ' '.join(colname), '')

    ## Save indexes
    self.wemindex.commit_index()
    self.eindex.commit_index()
    self.lshe.commit_index()
    self.processed = True

  def load_indexes(self):
    self.wemindex.load_index()
    self.eindex.load_index()
    self.lshe.load_index()
    self.processed = True

  def search_elastic_tblname(self, query, limit):
    assert self.processed
    return self.eindex.search_tblname(' '.join(query), limit)

  def search_elastic_colname(self, query, limit):
    assert self.processed
    return self.eindex.search_colname(' '.join(query), limit)

  def search_elastic_content(self, query, limit):
    assert self.processed
    return self.eindex.search_content(' '.join(query), limit)

  def search_lshe_content(self, query, card, limit):
    assert self.processed
    return self.lshe.search(query, card, limit)

  def search_wem_content(self, query, limit):
    assert self.processed
    return self.wemindex.search(query, limit)


if __name__ == "__main__":
  pass
