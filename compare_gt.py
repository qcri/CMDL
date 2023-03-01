import os
import csv
from profiler.text_profiler import text_profiler
from indexer.table_indexer import table_indexer
from indexer.trained_embeddings import TrainedEmbeddings, TrainedEmbeddingsIndexer
from tqdm import tqdm

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
  gt_path = 'inputs/mlopen-text-tables.gt' 
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

  '''
  ## trained embeddings eval
  feature_dir = 'features'
  def read_ids(file_path):
    with open(file_path, 'r') as f:
        return f.read().split('\n')

  text_ids = read_ids(os.path.join(feature_dir, datalake + '-textids.list'))
  col_ids = read_ids(os.path.join(feature_dir, datalake + '-colids.list'))
  text_emb_file = os.path.join(feature_dir, datalake + '-trainedtext.npy')
  col_emb_file = os.path.join(feature_dir, datalake + '-trainedcolumns.npy')

  text_emb = TrainedEmbeddings(text_ids, text_emb_file)
  col_emb = TrainedEmbeddings(col_ids, col_emb_file)

  # build ann index on column vectors
  col_emb_ind = TrainedEmbeddingsIndexer(datalake + '-trained', col_emb, mode)
  col_emb_ind.create_index()
  [col_emb_ind.index_doc(id) for id in col_ids]
  col_emb_ind.commit_index()

  # search preparations
  input_fn = text_emb
  trained_query_fn = col_emb_ind.search

  ## evaluations
  for topn in n:
    for algo, query_fn in [('trained', trained_query_fn)]:
      tqdm.write(f'Querying {algo} for {topn} matches')
      P, R, F1 = evaluate(gt_map, input_fn, query_fn, topn, None)
      csvf.writerow([algo, topn, P, R, F1])
  '''
  f.close()
