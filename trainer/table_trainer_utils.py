import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from transformers import RobertaModel

class TupleAugmentor():
    def __init__(self):
        random.seed(42)
    
    def augment(self, tblname_list, colname_list, content_list):
        tblname_list = tblname_list.copy()
        colname_list = colname_list.copy()
        content_list = content_list.copy()
        rand = random.randint(0, 3)
        ## with probability 50% swap two columns (swap)
        if rand < 2:
            if len(colname_list) > 1:
                [pos0, pos1] = random.sample(range(0, len(colname_list)), 2)
                colname_list[pos0], colname_list[pos1] = colname_list[pos1], colname_list[pos0]
                content_list[pos0], content_list[pos1] = content_list[pos1], content_list[pos0]
    
        ## with probability 25% randomly drop an attribute (drop)
        if rand == 2:
            pos = 0
            if len(colname_list) > 0:
                pos = random.randint(0, len(colname_list)-1)
            else:
                print('Exceptional tuple: ', tblname_list)
            del colname_list[pos]
            del content_list[pos]
    
        ## with probability 25% reorder header text
        if rand == 3:
            random.shuffle(tblname_list)
    
        return tblname_list, colname_list, content_list


class TupleEncoder():
    def __init__(self, tokenizer, conf):
        self.tokenizer = tokenizer
        self.max_header_len = conf.max_header_encoding_length
        self.max_row_len = conf.max_tuple_content_encoding_length
        self.header_token = '<head>'
        self.key_token = '<col>'
        self.value_token = '<val>'
        # Extend tokenizer with new tokens
        [self.tokenizer.add_tokens(t) for t in [self.header_token, self.key_token, self.value_token]]
        self.pad_token = self.tokenizer.pad_token
    
    def get_header_token(self):
        return self.header_token
    
    def get_row_key_token(self):
        return self.key_token
    
    def get_row_value_token(self):
        return self.value_token
    
    def encode_row(self, tblname_list, colname_list, content_list):
        if len(tblname_list) > self.max_header_len:
            print('Too long header at: ', tblname_list)
            return None
        else:
            pad_text = [self.pad_token] * (self.max_header_len - len(tblname_list))#' '.join([PAD] * (MAX_HEADER_LEN - len(tblname_list)))
            header_text = tblname_list + pad_text #' '.join([header_text, pad_text])

        row_text_list = []
        num_cols = len(colname_list)
        max_content_len = (self.max_row_len - 4*num_cols) // num_cols if num_cols > 0 else self.max_row_len ##approx limit
        for att, val in zip(colname_list, content_list):
            if len(val) > max_content_len:
                val = val[:max_content_len]
            #row_text_list.append(' '.join([KEY_SEP, ' '.join(att), VAL_SEP, ' '.join(val)]))
            row_text_list.extend([self.key_token] + att + [self.value_token] + val)
        #row_text = ' '.join(row_text_list)
        row_size = len(row_text_list)#len(row_text.split())
        if row_size > self.max_row_len:
            print('Exceptional row at: ', tblname_list)
            return
        else:
            pad_text = [self.pad_token] * (self.max_row_len - row_size) #' '.join([PAD] * (MAX_ROW_LEN - row_size))
            row_text_list = row_text_list + pad_text #' '.join([row_text, pad_text])
        
        #full_row_text = ' '.join([NEW_ROW_SEP, row_text, HEADER_SEP, header_text])
        return row_text_list + [self.header_token] + header_text#full_row_text


class TupleDataset(Dataset):
    def __init__(self, tokenizer, tupleids, orig_tokens, aug_tokens, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.tokenizer = tokenizer
        self.tupleids = tupleids
        self.device = device
        orig_ids = self._tokens_to_ids(orig_tokens)
        self.orig_ids = torch.tensor(orig_ids, device=self.device)
        aug_ids = self._tokens_to_ids(aug_tokens)
        self.aug_ids = torch.tensor(aug_ids, device=self.device)
        self.orig_attn = self._build_attn(orig_ids)
        self.aug_attn = self._build_attn(aug_ids)
        
    def _tokens_to_ids(self, tokens):
        return [self.tokenizer.encode(t, is_pretokenized=True) for t in tokens]

    def _build_attn(self, tokenlist):
        # attend everything except pad tokens
        pad_id = self.tokenizer.pad_token_id
        return torch.tensor([[int(not t==pad_id) for t in tokens] for tokens in tokenlist], 
                            device=self.device)
        
    def __len__(self):
        return len(self.tupleids)
    
    def __getitem__(self, index):
        piv, piv_attn = self.orig_ids[index], self.orig_attn[index]
        pos, pos_attn = self.aug_ids[index], self.aug_attn[index]
        idx = random.randint(0, len(self.tupleids)-1)
        coin = random.randint(0, 1)
        neg, neg_attn = self.orig_ids[idx]*coin + self.aug_ids[idx]*(1-coin), self.orig_attn[idx]*coin + self.aug_attn[idx]*(1-coin)
        return {"pivot_ids": piv, "positive_ids": pos, "negative_ids": neg,
                "pivot_attn": piv_attn, "positive_attn": pos_attn, "negative_attn": neg_attn
               }

    def get_tuple_id(self, index):
        return self.tupleids[index]
    
    
class TupleDataloader():    
    def __init__(self, tuple_dataset, conf):
        random.seed(42)
        self.training_data, self.test_data = random_split(tuple_dataset, 
                                                          [conf.train_size, conf.test_size],
                                                          torch.Generator().manual_seed(42))
        self.batch_size = conf.batch_size
    
    def training_dataloader(self, shuffle=True):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=shuffle)
    
    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=shuffle)
    
    
## Models taken from EMBER
class BertPooler(nn.Module):
    def __init__(self, config, final_size, pooling, pool_activation):
        super().__init__()
        self.pooling = pooling
        self.dense = nn.Linear(config.hidden_size, final_size)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.pool_activation = pool_activation

    def forward(self, hidden_states):
        if self.pooling == "CLS":
            pooled = hidden_states[:, 0]
        elif self.pooling == "MEAN":
            pooled = hidden_states.mean(axis=1)
        # Then we run it through a linear layer and optional activation     
        pooled_output = self.dense(pooled)
        if self.pool_activation == 'tanh':
            pooled_output = self.tanh(pooled_output)
        elif self.pool_activation == 'gelu':
            pooled_output = self.gelu(pooled_output)
            
        return pooled_output

class TripletSingleBERTModel(nn.Module):
    def __init__(self, final_size, pooling, model_path, tokenizer, pool_activation=None): 
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.pooler = BertPooler(self.model.config, final_size, pooling, pool_activation)
        if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

    def forward(self, a, p, n, a_mask, p_mask, n_mask):
        output_a = self.model(input_ids=a, attention_mask=a_mask).last_hidden_state
        output_p = self.model(input_ids=p, attention_mask=p_mask).last_hidden_state
        output_n = self.model(input_ids=n, attention_mask=n_mask).last_hidden_state
        
        output_a = self.pooler(output_a)
        output_p = self.pooler(output_p)
        output_n = self.pooler(output_n)

        return output_a, output_p, output_n 
    
    def return_emb(self, a, a_mask):
        output_a = self.model(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler(output_a)
        return output_a

