import os
import sys
import tarfile

import random
import numpy as np
import torch

import yaml
from argparse import ArgumentParser
from omegaconf import DictConfig, open_dict

import logging
logging.basicConfig(level=logging.INFO)
logging = logging.getLogger(__name__)

from src.agent.gpt_agent import Agent as gpt 

def main():
    config = load_config()

    logging.info('COMMAND: python '+' '.join(sys.argv))
    if config.seed != None:
        logging.info(f'SET seed {config.seed}')
        set_seed(config.seed)

    logging.info('<CONFIG>\n'+'\n'.join(print_config(config)))
    agent = gpt(**config)
    #save_code(config.hydra.run.dir, mode=config.mode)
    agent.run()

    return None 

def load_config():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='training.yaml')

    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    args = DictConfig(data)
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_code(path, mode='train'):
    codelist = [d.replace('.','/')+'.py' for d in sys.modules.keys() if 'src.' in d]
    codelist = [d for d in codelist if os.path.isfile(d)]
    codelist.append( os.path.join(path,'.hydra/') )
    with tarfile.open(os.path.join(path, f'code_{mode}.tar.gz'),'w:gz') as f:
        for filename in codelist:
            f.add(filename)

def print_config(config_dict, level=0):
    if type(config_dict) != dict:
        config_dict = dict(config_dict)
    result = list()
    for key in config_dict:
        if type(config_dict[key]) == DictConfig:
            result.append(f"{'  '*level}[ {key} ]:\t(dict)")
            result += print_config(config_dict[key], level=level+1)
        else:
            result.append(f"{'  '*level}[ {key} ]:\t({type(config_dict[key]).__name__})\t{config_dict[key]}")
    return result

if __name__ == "__main__":
    main()

