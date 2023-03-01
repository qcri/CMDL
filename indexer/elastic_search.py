from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import json
import requests
import re
from tqdm import tqdm
 
class ElasticIndexer():

  def __init__(self, index_name="default", mode='table'):
    self.client = Elasticsearch()
    self.index_name = index_name
    self.mode = mode
    self.committed = False

  def create_index(self):
    # delete index if exists
    if self.client.indices.exists(self.index_name):
      print('deleted es index')
      self.client.indices.delete(index=self.index_name)

    request_body = {
      "settings" : {
        "number_of_shards": 1,
        "number_of_replicas": 1
      },
      "mappings": {
        "properties": {
          "tblid": {"type": "keyword"},
          "tblname": {"type": "text"},
          "colname": {"type": "text"},
          "content": {"type": "text"}
        }
      }
    }

    self.client.indices.create(index=self.index_name, body=request_body)
    print('created es index')

  def index_doc(self, tblid, tblname, colname, content):
    assert tblid is not None or len(tblid)>0
    assert tblname is not None or len(tblid)>0
    assert colname is not None or len(tblid)>0

    rec = {'tblid': tblid, 'tblname': tblname, 'colname': colname, 'content': content}
    try: 
      self.client.index(index=self.index_name, body=rec)
    except Exception as e:
      print(e)

  def commit_index(self): 
    self.committed = True

  def load_index(self):
    self.committed = True

  def _search(self, query, limit):
    assert limit>0
    results = Search().using(self.client).index(self.index_name).query(query).execute()
    if results.hits.total['value'] == 0: return []
    #tqdm.write(f'Search for query {query} returend {results.hits.total} matches')

    hits_dict = {}
    for r in results[:2*limit]: # asking for twice the results      
      if self.mode == 'table': # combine results by table from keys (table, col)
        r_t = re.split(r',\s*(?![^"]*\"\,)', r.tblid)[0] #r.tblid.split(',')[0]
        score = hits_dict.get(r_t, 0)
        hits_dict[r_t] = score + r.meta.score
      else:
        hits_dict[r.tblid] = r.meta.score

    return sorted([(k, v) for k, v in hits_dict.items()], key=lambda tup: tup[1], reverse=True)[:limit]

  def _orquery(self, query, limit):
    should_query = []
    for qword in query.split(' '):
      should_query.append(Q("term", content=qword))
    return self._search(Q("bool", should=should_query), limit)

  def _andquery(self, query, limit):
    must_query = []
    for qword in query.split(' '):
      must_query.append(Q("term", content=qword))
    return self._search(Q("bool", must=must_query), limit)

  def search_tblname(self, query, limit):
    if query is None or len(query) == 0: return []
    return self._search(Q("match", tblname=query), limit)

  def search_colname(self, query, limit):
    if query is None or len(query) == 0: return []
    return self._search(Q("match", colname=query), limit)

  def search_content(self, query, limit):
    if query is None or len(query) == 0: return []
    return self._search(Q("match", content=query), limit)

  def search_all(self, query, limit):
    if query is None or len(query) == 0: return []
    return self._search(Q("multi_match", query=query, fields=['tblname', 'colname', 'content']), limit)

if __name__ == "__main__":
  es = ElasticIndexer()
  #es.create_index()
  #es.index_doc("a1", "b", "c", "def foo")
  #es.index_doc("a2", "x", "y", "def bar")
  #es.index_doc("a3", "b", "y", "foo bar")
  print(es.search_all("def foo x", 3)) 
 
