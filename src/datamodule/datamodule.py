import os
import re
import copy
import json
import pandas as pd
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import BertTokenizerFast, ElectraTokenizer, PreTrainedTokenizerFast, T5Tokenizer

class DataModule(Dataset):
    def __init__(self, 
            tokenizer=None, 
            data_path=None, 
            **kwargs
        ):

        self.config = DictConfig(kwargs)
        self.set_variable()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token

        self.data = self.load_data(data_path)
        logging.info(f"LOAD {data_path}")
        self.label_list = self.set_label_list() 

        if self.config.check_length:
            max_source_length = self.check_length([d['inputs'] for d in self.data])
            max_target_length = self.check_length([d['labels'] for d in self.data])
        if not self.config.max_source_length:
            self.config.max_source_length = max_source_length
        if not self.config.max_target_length:
            self.config.max_target_length = max_target_length

        assert self.config.max_source_length > 0, \
                f'Need datamodule.config.max_source_length > 0 or datamodule.config.check_length is True. '+\
                f'But datamodule.config.max_source_length is {self.config.max_source_length} and datamodule.config.check_length is {self.config.check_length}'


    def __len__(self):
        return len(self.data)

    ############ TODO: custom zone #####################
    def set_variable(self):
        self.inputs_name = 'source'
        self.labels_name = 'target'
        self.inputs_form = f'{self.config.task_prompt}'+'{}'+f'{self.config.prompt_prefix}'
        self.labels_form = '{}'+f'{self.config.prompt_suffix}'

    def __getitem__(self, index):
        data = self.data[index].copy()
        source_tensor = self.convert_sentence_to_input(data['inputs'], self.config.max_source_length, side='left')
        target_tensor = self.convert_sentence_to_input(data['labels'], self.config.max_target_length, side='right')
        target_tensor = source_tensor+target_tensor

        return {'inputs':source_tensor, 'labels':target_tensor, 'data':data}

    def load_data(self, filename):
        with open(filename, 'r') as f:
            data = [json.loads(d) for d in f]
        
        result = list()
        for item in data:
            item = dict(
                   inputs=item['source'], 
                   labels=item['target'],

#                   inputs='$'.join([item['sub_input'],item['input']]),
#                   labels=item['gen_label'],
                   raw=copy.deepcopy(item)
                   )
                    
            item['inputs'] = self.inputs_form.format(item['inputs'])
            item['labels'] = self.labels_form.format(item['labels'])

            result.append(item)
        return result

    #########################################

    def set_label_list(self):
        if self.config.label_file:
            labels_path = os.path.join(self.config.data_dir, self.config.label_file)
            with open(labels_path,'r') as f:
                return [d.strip() for d in f]
        else:
            targets = [d['labels'] for d in self.data]
            return sorted(set(targets))

    def get_label_list(self):
        return self.label_list

    def check_length(self, data):
        length = list()
        for item in data:
            tokenized_source = self.tokenizer.tokenize(item)
            length.append(len(tokenized_source))
        logging.info(f'CHECK Length:\n{pd.Series(length).describe()}')
        max_length = max(length+[0]) 
        return max_length
        
    def get_dataset(self):
        ## equal self class
        return self 

    def get_dataloader(self, sampler=None):
        dataloader = DataLoader(self,
                batch_size = self.config.batch_size, 
                shuffle = self.config.shuffle,
                num_workers = self.config.num_workers,
                sampler = sampler,
                collate_fn = lambda data: self.collate_fn(data))
        return dataloader

    def clean_text(self, text):
        text = text.strip()
        #text = re.sub('\[[^\]]*\]','',text) ## 대괄호 제거
        #text = re.sub('\([^\)]*\)','',text) ## 소괄호 제거
        #text = re.sub('[^ㅏ-ㅣㄱ-ㅎ가-힣0-9a-zA-Z\.%, ]',' ', text) ## 특수문자 모두 제거
        text = re.sub('  *',' ',text).strip() ## 다중 공백 제거
        return text

    def convert_sentence_to_input(self, inputs, max_len, side='left', special_token=False):
        ## tokenizer.encode(text, max_length=max_length, padding='max_length', truncation=True)
        inputs = self.tokenizer.tokenize(inputs)
        return self.convert_tokens_to_input(inputs, max_len, side=side, special_token=special_token)
    
    def convert_tokens_to_input(self, inputs, max_len, side='left', special_token=False):
        if special_token:
            inputs = [self.tokenizer.cls_token] + inputs + [self.tokenizer.sep_token] ## for bert

        dif = abs(max_len - len(inputs))
        if side == 'left':
            if len(inputs) < max_len:  inputs = ( [self.tokenizer.pad_token] * dif ) + inputs
            elif max_len < len(inputs):  inputs = inputs[dif:]
        else:
            if len(inputs) < max_len:  inputs += [self.tokenizer.pad_token] * dif
            elif max_len < len(inputs):  inputs = inputs[:max_len]

        inputs = self.tokenizer.convert_tokens_to_ids(inputs)
        return inputs

    def convert_input_to_tokens(self, inputs, special_token=False):
        return self.tokenizer.convert_ids_to_tokens(inputs, skip_special_tokens=special_token)

    def convert_input_to_sentence(self, inputs, special_token=False):
        return self.tokenizer.decode(inputs, skip_special_tokens=special_token)

    def generate_mask(self, data, max_len) :
    ## mask shape (batch, seq_len, vocab_size) 
    # vocab size 열 길이를 0으로 채우고 input id에 해당하는 인덱스를 1로 채움 
    # 똑같은 행렬을 seq_len 만큼 행으로 늘림 
    # 그리고 batch마다 다른 행렬을 가짐 
        
        ids = [d['inputs'] for d in data]
        size = self.tokenizer.vocab_size
        spId = list(set(self.tokenizer.all_special_ids))
        spMask = torch.nn.functional.one_hot(torch.tensor(spId), num_classes=size)
        data_len = len(data)  # batch size 
     
        result = list()
        for item in ids :
            item = list(set(item))
            mask = torch.nn.functional.one_hot(torch.tensor(item), num_classes=size)
            mask = torch.sum(mask, dim=0)
            
            # sptok 제외 
            #spMask = torch.sum(spMask, dim=0)
            #mask = mask - spMask
            
            #mask = torch.stack([mask]*max_len, dim=0)
            mask = mask.unsqueeze(0).repeat(max_len, 1)
            result.append(mask)


        #result = torch.stack(result, dim=0)
        res_len = len(result)
        res = torch.empty((res_len, *result[0].shape)) 

        for i in range(res_len) :
            res[i] = result[i]

        return res.tolist() 

    def collate_fn(self, data):
        input_mask = self.generate_mask(data, self.config.max_source_length+self.config.max_target_length) 
        output_mask = self.generate_mask(data, self.config.max_source_length)

        result = {
                'input_ids': [d['inputs'] for d in data],
                'labels': [d['labels'] for d in data],
                'data': [d['data'] for d in data], 
                'input_mask' : input_mask,
                'output_mask' : output_mask
                }
        for key in [d for d in result if d not in ['data']]:
            result[key] = torch.tensor(result[key])
            
        return result 



