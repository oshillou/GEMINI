from losses.vat import VATLoss
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler,SequentialSampler
from utils.metrics import AvgMetric
from utils.utils import get_loss, get_optimiser
from losses import base_loss
from torch.utils.tensorboard import SummaryWriter
import os
import logging

def train_model(config,model,train_ds,val_ds=None):
    logging.debug(f'Fetching optimiser: {config.optimiser} (lr= {config.learning_rate:.3E})')
    optimiser=get_optimiser(config,model.parameters())

    logging.debug(f'Initiating log for tensorboard')
    log_writer=SummaryWriter(log_dir=config.storage_path)
    logging.debug(f'Logging every {config.print_freq_step} steps for training and validation')

    device=torch.device('cuda' if (torch.cuda.is_available() and config.use_cuda) else 'cpu')

    logging.debug(f'Initiating training dataloader')
    train_loader=DataLoader(train_ds,sampler=BatchSampler(RandomSampler(train_ds),batch_size=config.batch_size,drop_last=False),collate_fn=train_ds.collate_fn)
    logging.debug(f'Train: Building metric for {config.model.clustering_head_kwargs.num_clusters} clusters')
    train_metrics=AvgMetric(config.model.clustering_head_kwargs.num_clusters)

    if val_ds is not None:
        logging.debug(f'Initiating validation dataloader')
        val_loader=DataLoader(val_ds,batch_size=config.batch_size,shuffle=False)
        logging.debug(f'Val: Building metric for {config.model.clustering_head_kwargs.num_clusters} clusters')
        val_metrics=AvgMetric(config.model.clustering_head_kwargs.num_clusters)

    loss_fct=get_loss(config)

    train_step=0
    val_step=0
    for epoch in range(config.epochs):
        logging.info(f'Epoch {epoch+1:3d}/{config.epochs}')
        train_metrics.reset()
        for batch in train_loader:

            # Cast the batch into double and set it to device if necessary
            batch=adapt_batch(config,batch,device)
            gemini_loss, entropy, vat=training_step(config,model,batch,loss_fct,train_metrics)

            total_loss=gemini_loss+entropy+vat
            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

            train_step+=1

            if train_step%config.print_freq_step==1:
                for name,value in zip(['GEMINI','Entropy','VAT'],[gemini_loss,entropy,vat]):
                    if value!=0:
                        log_writer.add_scalar(f'Train/{name}',value,train_step)
                log_writer.add_scalar('Train/loss',total_loss,train_step)
                log_writer.add_scalar('Train/epoch',epoch,train_step)
                logging.info(f'\tStep={train_step:4d}\tLoss={total_loss:.3f}\t(GEMINI: {gemini_loss:.3f}\tVAT: {vat:.3f}\tEntropy: {entropy:.3f})')
        
                log_metrics(config,log_writer,train_metrics,epoch,'Train')

        if val_ds is not None:
            val_metrics.reset()
            for batch in val_loader:

                batch=adapt_batch(config,batch,device)
                validation_step(model,batch,val_metrics)
                val_step+=1

            log_metrics(config,log_writer,val_metrics,epoch,'Val',True)
        log_writer.flush()
        checkpoint(config,model,epoch)
    
    log_writer.close()

    logging.info('Finished training')

    logging.debug(f'Final Training confusion matrix:\n{str(train_metrics)}')
    if val_ds is not None:
        logging.debug(f'Final Validation confusion matrix:\n{str(val_metrics)}')
        return train_metrics,val_metrics
    else:
        return train_metrics,None

def adapt_batch(config,batch,device):
    new_batch=tuple()
    for elem in batch:
        if elem.dtype==torch.float32 and 'wasserstein' in config.gemini.distance:
            new_elem=elem.double()
        else:
            new_elem=elem
        new_batch+=(new_elem.to(device),)
    return new_batch

def log_metrics(config,log_writer,m,epoch,mode='Train',verbose=False):
    # Log the metrics
    verbose_str=f'\t({mode}) '
    if config.metrics.ari:
        ari=m.ari()
        log_writer.add_scalar(f'{mode}/ARI',ari,epoch)
        if verbose:
            verbose_str+=f'\tARI: {ari:.3f}'
    if config.metrics.accuracy:
        accuracy=m.accuracy()
        log_writer.add_scalar(f'{mode}/Accuracy',accuracy,epoch)
        if verbose:
            verbose_str+=f'\tACC: {accuracy:.3f}'
    if config.metrics.purity:
        purity=m.purity()
        log_writer.add_scalar(f'{mode}/Purity',purity,epoch)
        if verbose:
            verbose_str+=f'\tPTY: {purity:.3f}'
    if config.metrics.used_clusters:
        ucl=m.ucl()
        log_writer.add_scalar(f'{mode}/UCL',ucl,epoch)
        if verbose:
            verbose_str+=f'\tUCL: {ucl:2d}'
    if config.metrics.stability:
        stability=m.stability()
        log_writer.add_scalar(f'{mode}/Stability',stability,epoch)
        if verbose:
            verbose_str+=f'\tSTB: {stability:3d}'
    if config.metrics.normalised_conditional_entropy:
        nce=m.normalised_conditional_entropy()
        log_writer.add_scalar(f'{mode}/NCE',nce,epoch)
        if verbose:
            verbose_str+=f'\tNCE: {nce:.3f}'
    if verbose:
        logging.debug(verbose_str)
    
def training_step(config,model,batch, loss_fct,train_metrics):
    if len(batch)==3:
        x,y,D=batch
    else:
        x,D=batch
        y=None
    p_yx,x_hat=model(x)

    gemini_loss=loss_fct(p_yx,D)
    
    if config.regularisations.entropy_weight!=0:
        entropy=config.regularisations.entropy_weight*base_loss.entropy_loss(p_yx)
    else:
        entropy=0
    if config.regularisations.vat_weight!=0:
        vat_loss=VATLoss(**config.regularisations.vat_kwargs)
        vat=config.regularisations.vat_weight*vat_loss(model,x,p_yx.detach())
    else:
        vat=0

    if y is not None:
        with torch.no_grad():
            train_metrics(y,p_yx.argmax(1))

    return gemini_loss, entropy, vat

def validation_step(model,batch,val_metrics):
    if len(batch)==3:
        x,y,D=batch
    else:
        x,y=batch
    
    p_yx,x_hat=model(x)

    val_metrics(y,p_yx.argmax(1))

def checkpoint(config,model,epoch):
    model_state_dict=model.state_dict()

    path_to_save=os.path.join(config.storage_path,'model.pt')
    torch.save({'epoch':epoch,'model_state_dict':model_state_dict},path_to_save)
