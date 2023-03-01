import csv
import os
import torch
from tqdm import tqdm
from trainer import text_featurizer, table_featurizer
from profiler import text_profiler, table_profiler
from indexer.wem import WEM

def write_csv(fp, rows):
  with open(fp, 'w') as f:
    csvf = csv.writer(f)
    [csvf.writerow([r]) for r in rows]

if __name__ == "__main__":
  datalake = 'mlopen' 
  text_path = 'inputs/mlopen-text' 
  table_path = 'inputs/mlopen-tables' 
  table_sep = ',' 
  wem_model_path = 'resources/fasttext/cc/cc.en.300.bin'
  wem_dim = 300
  gt_file = 'inputs/mlopen-text-tables.gt' 
  op_dir = 'features'

  ## TEMP: read gt file to create a list of text ids and table ids
  def read_gt(file_path):
    tid = set()
    cid = set()
    with open(file_path, 'r') as f:
      csvf = csv.reader(f)
      for r in csvf:
        tid.add(r[0])
        cid.add(r[1])
    return list(tid), list(cid)
  text_ids, table_ids = read_gt(gt_file) 
  tqdm.write(f'Filtering ids based on ground truth resulting in: {len(text_ids)} texts and {len(table_ids)} tables') ## col_ids are too few, not a good idea to restrict to the set

  ## new profilers
  text_p = text_profiler.text_profiler('en_core_web_sm')
  ## set table ids to the tables seen in ground truth, hack to restrict searches to those
  table_p = table_profiler.table_profiler(None, None)

  wem = WEM(wem_model_path, wem_dim)

  ## call featurizer to build features
  text_ids, text_f = text_featurizer.text_featurizer(wem, text_ids).featurize(text_p, text_path)
  col_ids, table_f = table_featurizer.table_featurizer(wem, None).featurize(table_p, table_path, table_sep)

  ## write outputs
  write_csv(os.path.join(op_dir, datalake + '-textids.list'), text_ids)
  write_csv(os.path.join(op_dir, datalake + '-colids.list'), col_ids)
  torch.save(text_f, os.path.join(op_dir, datalake + '-textfeatures.pt'))
  torch.save(table_f, os.path.join(op_dir, datalake + '-columnfeatures.pt'))
