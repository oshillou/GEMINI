import os
from datetime import datetime
import yaml
from easydict import EasyDict
import logging

from utils.utils import load_file_as_module

def create_config(args):    
    # config of experiments: training and model parameters data handling
    with open(args.config,'r') as file:
        config_exp=yaml.safe_load(file)
    
    setup_exp_config(config_exp)

    # Join the two configs together
    config=dict()
    for k,v in config_exp.items():
        config[k]=v

    if args.model_name!='':
        config['model_name']=args.model_name
    config['storage_path']=os.path.join(config['result_path'],config['model_name'])
    if not os.path.exists(config['storage_path']):
        os.makedirs(config['storage_path'])
    logging.info(f'Results will be stored in the folder {config["storage_path"]}')

    config['no_csv']=args.no_csv

    if args.seed!=-1:
        logging.info(f'Setting random seed to {args.seed}')
        import torch
        torch.manual_seed(0)
    
    return EasyDict(config)

def setup_exp_config(config_exp):
    config_keys=list(config_exp.keys())


    """Verifies that the configuration file for the environment
    is complete. Then, proceeds to create the appropriate folders"""
    # First assert that the configuration is well achieved
    message='Environment configuration misses a "{}" option'

    # Assert the main keys
    for key in ['gemini','dataset','similarity_fct','regularisations',\
        'model','epochs','optimiser','learning_rate','batch_size',\
        'num_workers','metrics','result_path','data_path']:
        assert key in config_keys, message.format(key)
    assert os.path.exists(config_exp['data_path']), f"Non-existant data path: {config_exp['data_path']}"
    logging.debug(f'Dataset files can be found in {config_exp["data_path"]}')


    # Add a path in which we will store the checkpoints of the model
    # The model name is simply the name of the user and the date time at which it was produced
    date=datetime.now().strftime('%y-%m-%d_%H-%M-%S_%f')
    model_name=f'{date}'

    config_exp['model_name']=model_name

    # Just create the path to the result directory
    if not os.path.exists(config_exp['result_path']):
        os.makedirs(config_exp['result_path'])

    
    check_key_or_replace(config_exp,'use_cuda',False)
    check_key_or_replace(config_exp,'optimiser_kwargs',dict())
    check_key_or_replace(config_exp,'print_freq_step',100)
    
    check_gemini(config_exp)

    check_dataset(config_exp)

    check_similarities(config_exp)

    check_regularisations(config_exp)

    check_model(config_exp)

    check_metrics(config_exp)

def check_dataset(config_exp):
    # Assert the configuration of the dataset file
    full_path=os.path.join(os.curdir,config_exp['dataset'])
    assert os.path.exists(full_path),f'Absent dataset file: {full_path}'
    check_module(config_exp['dataset'],'get_train_dataset')
    check_key_or_replace(config_exp,'dataset_kwargs',dict())

    logging.debug(f'Using dataset file: {config_exp["dataset"]}')

def check_similarities(config_exp):
    # Assert the similarity function. We only need for mmd and wasserstein
    if config_exp['gemini']['distance']=='wasserstein':
        allowed_distances=['euclidean','sqeuclidean','cosine']
        assert config_exp['similarity_fct'] is not None, f'No similarity_fct was provided for gemini wasserstein'
        assert config_exp['similarity_fct'] in allowed_distances or os.path.exists(config_exp['similarity_fct']), f'For Wasserstein GEMINI, similarity_fct should be one of {" ".join(allowed_distances)} or a valid path'
    elif config_exp['gemini']['distance']=='mmd':
        allowed_kernels=['linear','gaussian']
        assert config_exp['similarity_fct'] is not None, f'No similarity_fct was provided for gemini mmd'
        assert config_exp['similarity_fct'] in allowed_kernels or os.path.exists(config_exp['similarity_fct']), f'For MMD GEMINI, similarity_fct should be one of {" ".join(allowed_kernels)} or a valid path'
    check_key_or_replace(config_exp,'similarity_kwargs',dict())

    logging.debug(f'Using similarities: {config_exp["similarity_fct"]}')

def check_regularisations(config_exp):
    # Assert that the 3 regularisation weights are present
    check_key_or_replace(config_exp['regularisations'],'entropy_weight',0.0)
    assert config_exp['regularisations']['entropy_weight']>=0.0,'Cannot assigned negative weight to regularisations/entropy_weight'
    check_key_or_replace(config_exp['regularisations'],'vat_weight',0.0)
    assert config_exp['regularisations']['vat_weight']>=0.0,'Cannot assigned negative weight to regularisations/vat_weight'
    check_key_or_replace(config_exp['regularisations'],'reconstruction_weight',0.0)

    logging.info(f'Regularisations weights are (entropy: {config_exp["regularisations"]["entropy_weight"]:.3f}, \
        vat: {config_exp["regularisations"]["vat_weight"]:.3f}')
    
def check_model(config_exp):
    # Assert the presence of an encoder, a  clustering head
    check_key_or_replace(config_exp['model'],'encoder','encoder_mlp.py',False)
    check_key_or_replace(config_exp['model'],'encoder_kwargs',dict())
    assert os.path.exists(config_exp['model']['encoder']),f'Non existant encoder file : {config_exp["model"]["encoder"]}'
    check_module(config_exp['model']['encoder'],'get_model')

    check_key_or_replace(config_exp['model'],'clustering_head','clustering_head_mlp.py',False)
    check_key_or_replace(config_exp['model'],'clustering_head_kwargs',{'num_clusters':10})
    assert os.path.exists(config_exp["model"]["clustering_head"]), f'Non existant clustering head file: {config_exp["model"]["clustering_head"]}'
    check_module(config_exp['model']['clustering_head'],'get_model')

def check_gemini(config_exp):
    # Assert GEMINI configuration
    check_key_or_replace(config_exp['gemini'],'distance','tv',silent=False)
    check_key_or_replace(config_exp['gemini'],'ovo',True,silent=False)

    authorised_geminis=['kl','tv','hellinger','wasserstein','mmd','opt_wasserstein']
    if not config_exp['gemini']['distance'] in authorised_geminis:
        logging.warn(f'Unknown GEMINI: {config_exp["gemini"]["distance"]}, setting TV instead')
        config_exp['gemini']['distance']='tv'
    
    if config_exp['gemini']['distance']=='opt_wasserstein':
        check_key_or_replace(config_exp['gemini'],'N',5,silent=False)

    logging.debug(f'GEMINI is: {config_exp["gemini"]["distance"]} (OvO: {config_exp["gemini"]["ovo"]})')

def check_metrics(config_exp):
    check_key_or_replace(config_exp['metrics'],'ari',True)
    check_key_or_replace(config_exp['metrics'],'accuracy',True)
    check_key_or_replace(config_exp['metrics'],'purity',True)
    check_key_or_replace(config_exp['metrics'],'used_clusters',True)
    check_key_or_replace(config_exp['metrics'],'stability',True)
    check_key_or_replace(config_exp['metrics'],'normalised_conditional_entropy', True)

def check_key_or_replace(dictionary,key,default_value,silent=True):
    if not key in list(dictionary.keys()):
        if not silent:
            logging.warn(f'Unexistant key in configuration: {key}. Setting it to {default_value} by default')
        dictionary[key]=default_value
    elif dictionary[key] is None and type(default_value)==dict:
        dictionary[key]=default_value

def check_module(module_path,function):
    module=load_file_as_module(module_path)
    assert function in dir(module), f'Erroneous module: {module_path}, no function {function} provided'