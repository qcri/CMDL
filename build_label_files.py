from profiler.table_profiler import table_profiler
from profiler.text_profiler import text_profiler
from indexer.table_indexer import table_indexer
from tqdm import tqdm
import numpy as np
import csv

def fetch_unique_tableids(fp):
  if fp is None: return None
  table_ids = set()
  with open(fp, 'r') as f:
    csvf = csv.reader(f)
    [table_ids.add(r[1]) for r in csvf]
  tqdm.write(f'Unique tables found in ground truth {fp} = {len(table_ids)}')
  return table_ids

if __name__ == "__main__":
  datalake = 'mlopen' 
  text_path = 'inputs/mlopen-text' 
  table_path = 'inputs/mlopen-tables'
  table_sep = ',' 
  wem_model_path = 'resources/fasttext/cc/cc.en.300.bin'
  wem_dim = 300
  lshe_threshold = 0.5
  mode = 'column' ## column/table
  topn = 10 
  gt_file = None 
   
  ## new profilers
  text_p = text_profiler('en_core_web_sm')
  text_p.process_dir(text_path)
  tqdm.write(f"filtered text data: {text_p.num_filtered_ids()}")
  text_ids = text_p.get_ids()

  ## set table ids to the tables seen in ground truth, hack to restrict searches to those
  table_ids = fetch_unique_tableids(gt_file)
  table_p = table_profiler(table_ids, None)

   
  ## table indexer
  table_i = table_indexer(datalake, lshe_threshold, wem_model_path, wem_dim, mode=mode)
  tqdm.write(f'Building all indexes over tables')
  table_i.index_tables(table_p, table_path, table_sep)

  ## stats
  table_cards = np.array(table_p.get_cardinalities())
  text_cards = np.array(text_p.get_cardinalities())
  tqdm.write(f'Mean table cardinalities: {np.mean(table_cards)} +- {np.std(table_cards)}')
  tqdm.write(f'Mean text cardinalities: {np.mean(text_cards)} +- {np.std(text_cards)}')
  
  
  ## load indexes from files
  table_i = table_indexer(datalake, lshe_threshold, wem_model_path, wem_dim, mode=mode) # get labels for columns
  table_i.load_indexes()

  ## query table indexer for top 10 results
  tqdm.write(f'Querying all indexes over tables')
  tblname_labels = []
  colname_labels = []
  content_labels = []
  lshe_labels = []
  wem_labels = []
  for tid in tqdm(text_ids):
    card = text_p.get_cardinality(tid)
    query = text_p.get_profiled(tid)
    #tqdm.write(f'Searching for tables related to {tid}')
    tblname_res = table_i.search_elastic_tblname(query, topn)
    colname_res = table_i.search_elastic_colname(query, topn)
    content_res = table_i.search_elastic_content(query, topn)
    lshe_res = table_i.search_lshe_content(query, card, topn)
    wem_res = table_i.search_wem_content(query, topn)
    for lst, res in zip([tblname_labels, colname_labels, content_labels, lshe_labels, wem_labels], [tblname_res, colname_res, content_res, lshe_res, wem_res]):
      #tqdm.write(res)
      [lst.append((tid, rid)) for rid, rscore in res]

  ## write results to label files
  column_labels_dir = 'column_labels'
  tblname_file = os.path.join(column_labels_dir, datalake + '-tblname.lbl')
  colname_file = os.path.join(column_labels_dir, datalake + '-colname.lbl')
  content_file = os.path.join(column_labels_dir, datalake + '-content.lbl')
  lshe_file = os.path.join(column_labels_dir, datalake + '-lshe.lbl')
  wem_file = os.path.join(column_labels_dir, datalake + '-wem.lbl')

  for lst, fp in zip([tblname_labels, colname_labels, content_labels, lshe_labels, wem_labels], [tblname_file, colname_file, content_file, lshe_file, wem_file]):
    with open(fp, 'w') as f:
      csvf = csv.writer(f)
      for row in lst:
        #tqdm.write(f'writing row: {row} to file {fp}')
        csvf.writerow(row)
  tqdm.write(f'Wrote all label files')
  
