from tqdm import tqdm
import torch

class text_featurizer(object):

  ## if text_ids (<tbl_name>,<col_name>) is passed, features are created in the same order
  def __init__(self, wem_model, text_ids=None):
    self.wem_model = wem_model
    self.text_ids = text_ids

  def featurize(self, profiler, text_dir):
    text_feat = {}
    profiler.process_dir(text_dir)
    profiled_ids = profiler.get_ids()
    for id in tqdm(profiled_ids):
      if self.text_ids is not None and id not in self.text_ids:
        continue
      text_feat[id] = self.wem_model(profiler.get_profiled(id))
    ## save features  
    insertion_order = self.text_ids if self.text_ids is not None else list(text_feat.keys())
    feature_tensor = torch.Tensor(len(insertion_order), self.wem_model.get_dim())
    for i, id in tqdm(enumerate(insertion_order)):
      if id in text_feat:
        feature_tensor[i, :] = torch.tensor(text_feat[id])
      else:
        feature_tensor[i, :] = torch.zeros(feature_tensor.size(-1))
    tqdm.write(f'Build a feature tensor of dimension {feature_tensor.shape}')
    return insertion_order, feature_tensor

if __name__ == "__main__":
  pass
