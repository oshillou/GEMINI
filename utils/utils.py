from data.similarities import *
from losses.gemini import *
import importlib
import os
from data.datasets import SimilarityDataset
from models.model import DiscriminativeModel
from torch.utils.data import *
from torch import optim
import inspect
import pandas as pd
import logging

def str2similarity(string,**kwargs):
    if string is None:
        return None
    elif string=='euclidean':
        return EuclideanDistance(**kwargs)
    elif string=='sqeuclidean':
        return SQEuclideanDistance(**kwargs)
    elif string=='linear':
        return LinearKernel(**kwargs)
    elif string=='gaussian':
        return GaussianKernel(**kwargs)
    elif os.path.exists(string):
        # Load the module containing a custom similarity
        similarity_module=load_file_as_module(string)
        # Find the class that performs similarity operations
        found_similarity=None
        for (elem_name,elem) in filter(lambda x: '__' not in x[0], inspect.getmembers(similarity_module)):
            try:
                if elem.__bases__[0].__name__== Similarity.__name__:
                    found_similarity=elem
                    break
            except AttributeError as e:
                logging.error(e)
                pass
        if found_similarity is None:
            logging.warn(f'Did not find any Similarity inheriting class in file {string}, returning None')
            return found_similarity
        else:
            logging.debug(f'Using similarity {found_similarity.__name__} found in file {string}')
            return found_similarity(**kwargs)
    else:
        return None
    
def get_loss(config):
    if config.gemini.distance=='kl':
        if config.gemini.ovo:
            return FDivWrapper(kl_ovo)
        else:
            return FDivWrapper(kl_ova)
    elif config.gemini.distance=='tv':
        if config.gemini.ovo:
            return FDivWrapper(tv_ovo)
        else:
            return FDivWrapper(tv_ova)
    elif config.gemini.distance=='hellinger':
        if config.gemini.ovo:
            return FDivWrapper(hellinger_ovo)
        else:
            return FDivWrapper(hellinger_ova)
    elif config.gemini.distance=='mmd':
        if config.gemini.ovo:
            return mmd_ovo
        else:
            return mmd_ova
    elif config.gemini.distance=='wasserstein':
        if config.gemini.ovo:
            return wasserstein_ovo
        else:
            return wasserstein_ova
    elif config.gemini.distance=='opt_wasserstein':
        return lambda y_pred,D: optimised_wovo(y_pred,D,N=config.gemini.N)

def load_file_as_module(filename):
    # First, convert the potentially relative path of the filename to an absolute path
    filename=filename.replace('~',os.path.expanduser('~'))
    filename=os.path.abspath(filename)
    module_name=filename.split(os.path.sep)[-1].replace('.py','')
    module_spec=importlib.util.spec_from_file_location(module_name,filename)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module

def load_and_build_train_dataset(config):

    logging.debug(f'Loading dataset file {config.dataset}')
    # Load the file containing the dataset that we want
    dataset_module=load_file_as_module(config.dataset)

    # Fetch training dataset
    train_dataset=dataset_module.get_train_dataset(data_path=config.data_path,**config.dataset_kwargs)
    
    logging.debug(f'Loading similarities: {config.similarity_fct}')
    # We can now build the similarities
    similarities=str2similarity(config.similarity_fct,**config.similarity_kwargs)

    # Build the augmented dataset with similarities
    logging.debug(f'Building similarity dataset for {len(train_dataset)} training samples')
    if len(train_dataset)>2000:
        logging.debug(f'This may take a while')
        B=500
    else:
        B=100
    ds=SimilarityDataset(train_dataset,similarities,similarity_batch=B,num_workers=config.num_workers)

    return ds

def load_and_build_val_dataset(config):

    logging.debug(f'Loading dataset file {config.dataset}')
    # Load the file containing the dataset that we want
    dataset_module=load_file_as_module(config.dataset)

    # Fetch training and validation dataset (if existing)
    if 'get_val_dataset' not in dir(dataset_module):
        logging.info(f'No validation dataset provided, skipping! (Include a function "get_val_dataset" to provide one)')
        return None
    val_dataset=dataset_module.get_val_dataset(data_path=config.data_path,**config.dataset_kwargs)

    return val_dataset

def build_model(config):
    # Load the encoder
    encoder=load_file_as_module(config.model.encoder).get_model(**config.model.encoder_kwargs)

    # Load the clustering head
    clustering_head=load_file_as_module(config.model.clustering_head).get_model(**config.model.clustering_head_kwargs)
    
    model=DiscriminativeModel(encoder,clustering_head)

    if 'wasserstein' in config.gemini.distance:
        model=model.double()

    if config.use_cuda and torch.cuda.is_available():
        logging.debug(f'Using CUDA with {torch.cuda.device_count()} GPUs')
        if torch.cuda.device_count()>1:
            model=torch.nn.DataParallel(model)
        device=torch.device('cuda')
        model.to(device)
    else:
        logging.debug(f'Running model on CPU (CUDA available? {torch.cuda.is_available()})')
    
    return model

def get_optimiser(config,parameters):

    lr=config.learning_rate

    if config.optimiser=='adam':
        optimiser=optim.Adam(parameters,lr=lr,**config.optimiser_kwargs)
    elif config.optimiser=='sgd':
        optimiser=optim.SGD(parameters,lr=lr,**config.optimiser_kwargs)
    elif config.optimiser=='rmsprop':
        optimiser=optim.RMSProp(parameters,lr=lr,**config.optimiser_kwargs)
    
    return optimiser

def export_csv_results(config,metrics, train=True):
    # Create the series that contain every information

    series=dict()

    # Model parameters
    series['model_name']=config.model_name
    series['epochs']=config.epochs
    series['optimiser']=config.optimiser
    series['learning_rate']=config.learning_rate
    series['batch_size']=config.batch_size

    # Loss definition
    series['similarity']=config.similarity_fct
    for sub_key, sub_val in config.similarity_kwargs.items():
        series[f'similarity_{sub_key}']=sub_val
    series['entropy_weight']=config.regularisations.entropy_weight
    series['vat_weight']=config.regularisations.vat_weight
    for sub_key, sub_val in config.regularisations.vat_kwargs.items():
        series[f'vat_{sub_key}']=sub_val
    series['gemini']=config.gemini.distance
    series['ovo']=config.gemini.ovo
    
    # Dataset
    series['dataset']=config.dataset
    for sub_key, sub_val in config.dataset_kwargs.items():
        series[f'dataset_{sub_key}']=sub_val
    series['train']=train
    
    # Architecture
    series['encoder']=config.model.encoder
    for sub_key, sub_val in config.model.encoder_kwargs.items():
        series[f'encoder_{sub_key}']=sub_val
    series['clustering_head']=config.model.clustering_head
    for sub_key, sub_val in config.model.clustering_head_kwargs.items():
        series[f'clustering_head_{sub_key}']=sub_val
    
    # Results
    series['ARI']=metrics.ari().item()
    series['ACC']=metrics.accuracy().item()
    series['PTY']=metrics.purity().item()
    series['UCL']=metrics.ucl().item()
    series['NCE']=metrics.normalised_conditional_entropy().item()

    # Create the complete series
    series=pd.Series(series)

    # Verify that there is not already an existing csv, if so load it
    csv_name=os.path.join(config.result_path,'results.csv')
    logging.debug(f'CSV will be exported at {csv_name}')
    if os.path.exists(csv_name):
        df=pd.read_csv(csv_name,sep=',',index_col=False)
    else:
        df=pd.DataFrame()
    logging.debug(f'There are currently {len(df)} entries in this csv, appending one')
    # Concat new series to dataframe
    df=pd.concat([df,series.to_frame(1).T])
    # In case of missing columns, replace all NaNs to -1
    df.fillna(-1,inplace=True)

    # Export csv
    df.to_csv(csv_name,sep=',',index=False)