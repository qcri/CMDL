import os
import sys, getopt
import csv
from indexer.trained_embeddings import TrainedEmbeddings, TrainedEmbeddingsIndexer
from tqdm import tqdm
from compare_gt import text2table, evaluate

def read_ids(file_path):
  with open(file_path, 'r') as f:
    csvf = csv.reader(f, delimiter='\n')
    return [r[0] for r in csvf]

def main(argv):
  feature_dir = 'features'
  feature_tmp_dir = feature_dir
  datalake = 'mlopen'
  features_id = 0
  try:
    opts, args = getopt.getopt(argv,"hf:d:i:g:")
    #print(opts)
  except getopt.GetoptError:
    print('evaluate_trained.py -f <feature_dir> -d <datalake_name> -i <features_id> -g <ground_truth_file>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('evaluate_trained.py -f <feature_dir> -d <datalake_name> -i <features_id> -g <ground_truth_file>')
      sys.exit()
    elif opt == "-f":
      feature_tmp_dir = arg
    elif opt == "-d": 
      datalake = arg
    elif opt == "-i": 
      features_id = int(arg)
    elif opt == "-g":
      gt_path = arg

  mode = 'table' # column/table

  gt_map = text2table(gt_path)

  ## query eval results
  eval_results = []

  ## parameters
  n = [1, 2, 4, 6, 8, 10, 12, 15, 18] 

  ## trained embeddings eval
  text_ids = read_ids(os.path.join(feature_dir, datalake + '-textids.list'))
  col_ids = read_ids(os.path.join(feature_dir, datalake + '-colids.list'))
  text_emb_file = os.path.join(feature_tmp_dir, datalake + '-' + str(features_id) + '-trainedtext.npy')
  col_emb_file = os.path.join(feature_tmp_dir, datalake + '-' + str(features_id) + '-trainedcolumns.npy')

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
  for topn in tqdm(n):
    for algo, query_fn in [('trained', trained_query_fn)]:
      #tqdm.write(f'Querying {algo} for {topn} matches')
      P, R, F1 = evaluate(gt_map, input_fn, query_fn, topn, None)
      eval_results.append([features_id, topn, algo, P, R, F1])
  return eval_results

if __name__ == "__main__":
  eval_results = main(sys.argv[1:])

