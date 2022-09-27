from utils.config import create_config
from utils.train_utils import train_model
import argparse
import logging
import sys

from utils.utils import build_model, export_csv_results,load_and_build_train_dataset,load_and_build_val_dataset

def parse_configuration():
    parser=argparse.ArgumentParser()

    parser.add_argument('--config',type=str,required=True)
    parser.add_argument('--model_name',type=str,default='')
    parser.add_argument('--seed',type=int,default=-1)
    parser.add_argument('--no_csv',default=False,action='store_true')

    arguments=parser.parse_args()

    return arguments


if __name__=='__main__':
    logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)
    
    args=parse_configuration()

    logging.info('Looking for configuration files')
    config=create_config(args)

    logging.info('Loading training dataset')
    train_ds=load_and_build_train_dataset(config)
    logging.info('Loading validation dataset')
    val_ds=load_and_build_val_dataset(config)

    logging.info('Building the model')
    model=build_model(config)

    logging.info('Training model')
    train_metrics,val_metrics=train_model(config,model,train_ds,val_ds)

    if not config.no_csv:
        logging.info('Exporting csv results')
        logging.debug('Exporting training metrics')
        export_csv_results(config,train_metrics)
        if val_metrics is not None:
            logging.debug('Exporting validation metrics')
            export_csv_results(config,val_metrics,False)
            

    logging.info('Finished')
    sys.exit(0)