import os
import sys
import re
import json
from tqdm import tqdm

from omegaconf import DictConfig
import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

import torch
import transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.datamodule.datamodule import DataModule
from src.model.gpt import SequenceClassification

class Agent():
    def __init__(self,
            **kwargs
        ):
        self.config = DictConfig(kwargs)

        self.checkpoint_path = self.config.save_dir
        os.makedirs(self.checkpoint_path, exist_ok=True)

        ## set tokenizer
        self.tokenizer = self.set_tokenizer(self.config.tokenizer)
        ## set data and labels
        self.set_data(self.config.mode)
        ## set model
        self.model = self.set_model(self.config.model)
        self.optimizer = self.set_optimizer(self.config.optimizer)
                
    def run(self):
        if self.config.mode == 'train':
            self.fit()
        elif self.config.mode == 'predict':
            self.predict()
        else:
            raise NotImplementedError('OPTION "{}" is not supported'.format(self.config.mode))


    ######## SETTING #######################################################
    ################# 각 config에 명시된 클래스 경로 사용
    ################# 아래에서는 클래스에 전달할 config 설정

    def set_tokenizer(self, config):
        ## source file: src/tokenizer/, config file: config/tokenizer/
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.path)
        tokenizer.padding_side='left'
        tokenizer.truncation_side='left'
        return tokenizer

    def set_model(self, config):
        ## source file: src/model/, config file: config/model/
        config = dict(config)

        model = SequenceClassification(**config)

        if torch.cuda.is_available():
            model = model.to('cuda')
        return model
    
    def set_optimizer(self, config):
        optimizer = self.model.configure_optimizers(lr=float(config.lr))
        return optimizer

    def set_dataloader(self, config, **kwargs):
        ## source file: src/datamodule/, config file: config/datamodule/
        config = dict(config)
        config['data_path'] = kwargs['data_path']
        config['tokenizer'] = kwargs['tokenizer']

        datamodule = DataModule(**config)
        dataloader = datamodule.get_dataloader()
        return dataloader

    def set_data(self, mode):
        ## config file: config/config.yaml, config/datamodule/
        def get_train_dataloader():
            return self.set_dataloader(self.config.datamodule, tokenizer=self.tokenizer, 
                                        data_path=os.path.join( self.config.datamodule.data_dir, 
                                                                self.config.datamodule.train_data)
                                        )
        def get_valid_dataloader():
            return self.set_dataloader(self.config.datamodule, tokenizer=self.tokenizer, 
                                        data_path=os.path.join( self.config.datamodule.data_dir, 
                                                                self.config.datamodule.valid_data)
                                        )
        def get_test_dataloader():
            return self.set_dataloader(self.config.datamodule, tokenizer=self.tokenizer, 
                                        data_path=os.path.join( self.config.datamodule.data_dir, 
                                                                self.config.datamodule.test_data)
                                        )
        if mode in ['train']:
            self.train_dataloader = get_train_dataloader()
            self.valid_dataloader = get_valid_dataloader()
            if self.config.agent.predict_after_training or self.config.agent.predict_after_all_training:
                self.test_dataloader = get_test_dataloader()
        elif mode in ['predict']:
            self.test_dataloader = get_test_dataloader()

    ########################################################################

    
    def fit(self): ## TRAINING
        earlystop_threshold = self.config.agent.patience
        patience = 0
        max_loss = 999

        for epoch in range(self.config.agent.epochs):
            print('',flush=True)
            ## training step
            self.model.train()
            tr_loss = tr_acc = 0
            dataloader = tqdm(self.train_dataloader)#, ascii=True)
            for index, item in enumerate(dataloader): ## 1 epoch
                batch = {'input_ids':item['labels'], 'labels':item['labels'], 'mask': item['input_mask']}
                output = self.model.training_step(batch, index)
                if index == 0:
                    pred = torch.argmax(output['logits'], dim=-1)
                    print(self.tokenizer.decode(item['labels'][0]))
                    print(self.tokenizer.decode(pred[0]))
                    print('', flush=True)

                loss = output.pop('loss', None)
                acc = output.pop('acc', None)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                tr_loss += loss.item()
                tr_acc += acc.item()

                dataloader.set_description(f"[TRAIN] Epoch{epoch}-L{tr_loss/(index+1):.3f}-A{tr_acc/(index+1):.3f}")
            logging.info(f"[TRAIN] Epoch_{epoch}-L_{tr_loss/(index+1):.3f}-A_{tr_acc/(index+1):.3f}")

            ## validation step
            self.model.eval()
            val_loss = val_acc = 0
            dataloader = tqdm(self.valid_dataloader)#, ascii=True)
            for index, item in enumerate(dataloader): ## 1 epoch
                batch = {'input_ids':item['labels'], 'labels':item['labels'], 'mask': item['input_mask']}
                output = self.model.validation_step(batch, index)

                loss = output.pop('loss', None)
                acc = output.pop('acc', None)

                if not torch.isnan(loss):
                    val_loss += loss.item()
                else:
                    val_loss += val_loss
                val_acc += acc.item()

                dataloader.set_description(f"[VALID] Epoch{epoch}-L{val_loss/(index+1):.3f}-A{val_acc/(index+1):.3f}")
            logging.info(f"[VALID] Epoch_{epoch}-L_{val_loss/(index+1):.3f}-A_{val_acc/(index+1):.3f}")

            if self.config.agent.model_all_save:
                path = os.path.join(
                        self.checkpoint_path,
                        f"train" +\
                        f"_lr_{self.config.optimizer.lr}" +\
                        f"_batch_{self.config.datamodule.batch_size}" +\
                        f"_pat_{self.config.agent.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss/(index+1):.4f}" +\
                        f"_acc_{val_acc/(index+1):.4f}"
                        )
                self.model.save_model(path)
                if self.config.agent.predict_after_all_training: self.predict(dir_path=path)

            ## earlystop
            val_loss = val_loss/(index+1)
            if val_loss < max_loss:
                max_loss = val_loss
                patience = 0
                path = os.path.join(
                        self.checkpoint_path,
                        f"valid" +\
                        f"_lr_{self.config.optimizer.lr}" +\
                        f"_batch_{self.config.datamodule.batch_size}" +\
                        f"_pat_{self.config.agent.patience}" +\
                        f"_epoch_{epoch:07d}" +\
                        f"_loss_{val_loss/(index+1):.4f}"+\
                        f"_acc_{val_acc/(index+1):.4f}"
                        )
                self.model.save_model(path)
                if self.config.agent.predict_after_all_training: self.predict(dir_path=path)
                path = os.path.join( self.checkpoint_path, "trained_model" )
                self.model.save_model(path)
            else:
                patience += 1
                if patience > earlystop_threshold:
                    logging.info('Ran out of patience.')
                    break ## STOP training

        ## predict best model
        if self.config.agent.predict_after_training or self.config.agent.predict_after_all_training:
            path = os.path.join( self.checkpoint_path, "trained_model" )
            self.predict(dir_path=path)

    
    def predict(self, dir_path=None):
        self.model.eval()
        if not dir_path:
            dir_path = self.config.model.path

        if self.config.predict_file_path:
            ofp_name = os.path.join( dir_path, self.config.predict_file_path )
            ofp = open(ofp_name,'w')
            logging.info(f"WRITE {ofp_name}")
        else:
            if self.config.mode == 'train':
                raise ValueError('No empty predict_file_path supported. Make sure predict_file_path has a value.')
            ofp = sys.stdout
            logging.info(f"WRITE sys.stdout")

        dataloader = tqdm(self.test_dataloader)#, ascii=True)
        for index, item in enumerate(dataloader): ## 1 epoch
            batch = {'input_ids':item['input_ids'], 'labels':item['labels'], 'mask': item['output_mask']}
            output = self.model.predict_step(batch, index, max_length=self.config.datamodule.max_source_length+self.config.datamodule.max_target_length)

            for origin_data, model_output in zip(item['data'], output):
                model_output['inputs'] = self.tokenizer.decode(model_output['inputs'], skip_special_tokens=True)
                for key in ['preds', 'labels']:
                    decoded_output = self.tokenizer.decode(model_output[key], skip_special_tokens=True)
                    model_output[f'origin_{key}'] = decoded_output
                    #decoded_output = self.extract_labels(decoded_output, prefix=self.config.datamodule.prompt_prefix, suffix=self.config.datamodule.prompt_suffix)
                    try:
                        model_output[key] = decoded_output.replace(model_output['inputs'], '')
                        model_output[key] = model_output[key].strip()
                    except IndexError as e:
                        model_output[key] = ''

                result = {'data':origin_data, 'output':model_output}
                ofp.write(json.dumps(result, ensure_ascii=False)+'\n')
            dataloader.set_description(f"[PREDICT]")


    def extract_labels(self, data, index=None, prefix=None, suffix=None, prompt=None):
        result = re.findall(f'{prefix}[^{suffix}]*{suffix}', data)
        try:
            if index != None: result = [result[index]]
        except IndexError as e:
            result = [f'{prefix}{suffix}']
        result = [d[d.index(prefix)+len(prefix):] for d in result]
        result = [d[:d.index(suffix)] for d in result]
        result = [d.strip() for d in result]
        return result


