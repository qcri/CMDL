import os
import csv
from profiler.text_profiler import text_profiler
from profiler.table_profiler import table_profiler
from indexer.table_indexer import table_indexer
from indexer.trained_embeddings import TrainedEmbeddings, TrainedEmbeddingsIndexer
from tqdm import tqdm

### Evaluation for Tasks 1A, 1B, 1C from the paper are covered in evaluate_trained.py
### This evaluate_table_to_table() function covers evaluation for experiments 2A 2B 2C 2D 3A 3B

def text2table(gt_file, sep=','):
  gt_map = {}
  #tqdm.write(f'Reading GT from {gt_file}')
  with open(gt_file, 'r') as f:
    csvf = csv.reader(f, delimiter=sep)
    for line in tqdm(csvf):
      key = line[0]
      value = line[1]
      new_values = gt_map.get(key, [])
      new_values.append(value)
      gt_map[key] = new_values
  return gt_map

def eval_matches(gt_map, predictions):
  tp = 0
  fp = 0
  fn = 0
  cnt = 0
  for idx in predictions:
    #tqdm.write(f'Working on {idx}')
    gt_matches = set(gt_map.get(idx, []))
    if len(gt_matches) == 0: continue
    pred_matches = set(predictions.get(idx, []))
    true_matches = len(gt_matches.intersection(pred_matches))
    false_matches = len(pred_matches) - true_matches
    non_matches = len(gt_matches) - true_matches

    tp += true_matches
    fp += false_matches
    fn += non_matches
    cnt += 1
  #tqdm.write(f'TP={tp}, FP={fp}, FN={fn}')
  fp = 1 if (tp + fp) == 0 else fp
  prec = 1.0 * tp / (tp + fp)
  rec = 1.0 * tp / (tp + fn)
  f1 = 0 if tp == 0 else 2.0 * prec * rec / (prec + rec)
  #tqdm.write(f'Precision={prec}, Recall={rec}')
  return prec, rec, f1

## given ground truth to evaluate against, a search function, and topn parameter, return prec, recall
def evaluate(gt_map, ip_fn, search_fn, topn, card_map=None):
  predictions = {}
  for idx in gt_map:
    input = ip_fn(idx)
    results = search_fn(input, topn) if card_map is None else search_fn(input, card_map.get(idx, 0), topn)
    predictions[idx] = [tid for (tid, score) in results]
  return eval_matches(gt_map, predictions)

if __name__ == "__main__":
  datalake = 'mlopen' 
  text_path = 'inputs/mlopen-text' 
  table_path = 'inputs/mlopen-tables'
  table_sep = ','
  lshe_threshold = 0.5
  wem_model_path = 'resources/fasttext/cc/cc.en.300.bin'
  wem_dim = 300 
  mode = 'table' # column/table
  result_file = datalake + '.results'

  
  ## new profilers
  text_p = text_profiler('en_core_web_sm')
  text_p.process_dir(text_path)
  text_ids = text_p.get_ids()
  text_cards = text_p.get_cardinalities()

  ## load indexes from files
  table_i = table_indexer(datalake, lshe_threshold, wem_model_path, wem_dim, mode=mode)
  table_i.load_indexes()
  
  
  ## read gt
  gt_path = 'inputs/mlopen-text-tables.gt' 
  gt_map = text2table(gt_path)

  ## query eval results
  f = open(result_file, 'w')
  csvf = csv.writer(f)
  csvf.writerow(['Formula', 'K', 'P', 'R', 'F1'])

  ## parameters
  n = [1, 2, 4, 6, 8, 10, 12, 15, 18]

  
  algo_names = ['schema-s', 'content-s', 'contain', 'semantic']
  query_fns = [table_i.search_elastic_colname, table_i.search_elastic_content, table_i.search_lshe_content, table_i.search_wem_content]
  input_fn = text_p.get_profiled
  
  ## evaluations
  for algo, query_fn in zip(algo_names, query_fns):
    tqdm.write(f'Querying {algo}')
    for topn in tqdm(n):
      card_map = None
      if algo == 'contain':
        card_map = dict(zip(text_ids, text_cards))
      P, R, F1 = evaluate(gt_map, input_fn, query_fn, topn, card_map)
      csvf.writerow([algo, topn, P, R, F1])

  f.close()

def evaluate_table_to_table(table_dir, topn, gt_path):

  gt_map = text2table(gt_path)
  mode = "column"
  profiler = table_profiler()
  sep=','

  ## Make Features
  table_indexed = table_indexer(datalake, lshe_threshold, wem_model_path, wem_dim, mode=mode)
  ## Make Indexes for the features
  table_indexed.index_tables(profiler, table_dir, sep)
  ## Load indexes of features
  table_i.load_indexes()

  ## Base Indexes for Searching
  base_fns = table_i.search_elastic_colname # Other Options: [table_i.search_elastic_colname, table_i.search_elastic_content, table_i.search_lshe_content, table_i.search_wem_content]

  final_results = []
  ## Loop over Probe Tables
  for (id, tblname, colname, is_string, colvalues, cardinality) in profiler.process_dir(table_dir, sep):
      
      ## For Table2Table PKFK Join - Experiment 2D: uncomment the following code and put all the following lines inside that if condition
      # if cardinality/len(colvalues) >= 0.95:
    
      ## input/probe for each table
      input_fn = colname # ' '.join(colvalues)

      ## returned matches (topn)
      results = base_fns.search(input_fn, topn*5) # 5 is arbitrary factor for filtering out same table name matches
      # For lsh: base_fns.search(input_fn, len(input_fn.split()) , topn)
      
      count = 0
      ## Filtering Function
      for result in results:
        if result["table_name"] != tblname:
          final_results.append("{},{},{},{}".format(tblname,colname,result["table_name"],result["column_name"]))
          count+=1
        if count == topn:
          break
        
  ## Final Metrics Calculation
  P, R, F1 = eval_matches(gt_map, final_results)
  return P, R, F1
  ################################################################################################################################################################