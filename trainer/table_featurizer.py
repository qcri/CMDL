from tqdm import tqdm
import numpy as np
import torch

class table_featurizer(object):

  ## if col_ids (<tbl_name>,<col_name>) is passed, features are created in the same order
  def __init__(self, wem_model, col_ids=None):
    self.wem_model = wem_model
    self.col_ids = col_ids

  def featurize(self, profiler, table_dir, sep):
    tblname_feat = {}
    colname_feat = {}
    colvalues_feat = {}
    for (id, tblname, colname, is_string, colvalues, card) in profiler.process_dir(table_dir, sep):
      #tqdm.write(f'Got inputs: {id} with content: {len(colvalues)}')
      if self.col_ids is not None and id not in self.col_ids:
        tqdm.write(f'Received {id} which is not in list')
        continue
      tblname_feat[id] = self.wem_model(tblname) if len(tblname) > 0 else np.zeros(self.wem_model.get_dim())
      colname_feat[id] = self.wem_model(colname) if len(colname) > 0 else np.zeros(self.wem_model.get_dim())
      if is_string and len(colvalues) > 0:
        colvalues_feat[id] = self.wem_model(colvalues)
      else:
        colvalues_feat[id] = np.zeros(self.wem_model.get_dim())
    ## save features  
    insertion_order = self.col_ids if self.col_ids is not None else list(tblname_feat.keys())
    feature_tensor = torch.Tensor(len(insertion_order), 3*self.wem_model.get_dim())
    tqdm.write(f'Building feature vector on {len(insertion_order)} columns')
    for i, id in tqdm(enumerate(insertion_order)):
      if id in tblname_feat:
        merged = np.concatenate((tblname_feat[id], colname_feat[id], colvalues_feat[id]), axis=0)
      else:
        merged = np.zeros(3*self.wem_model.get_dim())
      feature_tensor[i, :] = torch.tensor(merged)
    tqdm.write(f'Built a feature tensor of dimension {feature_tensor.shape}')
    return insertion_order, feature_tensor

if __name__ == "__main__":
  pass
