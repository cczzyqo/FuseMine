import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from model.dti import DTIModel
import torch
import pandas as pd
import os
import argparse
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from prettytable import PrettyTable
from dataset.dataset import preparedataset
from utils import mkdir, get_config
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description='Drug-Target Interaction Prediction (Lightning Version)')
    parser.add_argument('--base_config', type=str, default='./config/config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset name (e.g., biosnap, drugbank, bindingdb etc.)')
    parser.add_argument('--outname', required=True, type=str, help='output folder name')
    parser.add_argument('--num_worker', required=True, type=int, help='number of workers for dataloader')
    parser.add_argument('--gpus', type=str, default='5,6', help='Specify GPUs to use, e.g. 0,1,2,3')
    parser.add_argument('--strategy', type=str, default='ddp', choices=['ddp', 'single'], 
                       help='Training strategy: ddp (distributed) or single (single GPU)')
    parser.add_argument('--type', type=str, default='random', choices=['random', 'cold', 'cluster'], 
                        help='Training type: random, cold, cluster')
    return parser.parse_args()


# python train.py --outname biosnap_model --data biosnap --num_worker 0 --type random
# python train.py --outname bindingdb_model --data bindingdb --num_worker 0 --type cluster
def main():
    # Parse arguments
    args = parse_args()
    
    # Set visible GPUs 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    # Load and setup configuration
    cfg = get_config(args.base_config, args.data)
    # set seed 
    pl.seed_everything(cfg.Global.Seed, workers=True)

    # Setup device and paths
    datafolder = os.path.join(cfg.Data.Path, args.data)
    outfolder = os.path.join(cfg.Result.Output_Dir, args.data, args.type)
    mkdir(outfolder)
    print(f'datafolder: {datafolder} \n output_folder: {outfolder}')
    
    # prepare data
    train_loader, val_loader, test_loader, compounds, proteins = preparedataset(cfg.Global.Batch_Size, args.type, args.data, args.num_worker,cfg.Drug.Max_Nodes, cfg.Protein.Max_Length)
    
    # define model
    model = DTIModel(cfg, learning_rate=cfg.Global.LR)
    
    # Save model configs
    OmegaConf.save(cfg, os.path.join(outfolder, "model_configs.yaml"))
    with open(os.path.join(outfolder, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(outfolder, 'checkpoints'),
        filename='best_model',  # Simplify file name for loading
        save_top_k=1,
        verbose=True,
        monitor='val_auroc',  # Use AUROC for selecting best model
        mode='max',  # AUROC higher is better
        save_weights_only=True,  # Save only weights for easier loading
    )

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=outfolder,
        name='lightning_logs'
    )
    

    ddp_strategy = DDPStrategy(
        find_unused_parameters=True,
        process_group_backend="nccl",
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[5,6],
        strategy=ddp_strategy,
        max_epochs=cfg.Global.Max_Epoch,
        callbacks=[checkpoint_callback],
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=logger,
        default_root_dir=outfolder
    )
    

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test
    # print best model path for test
    if trainer.is_global_zero:
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nbest model path: {best_model_path}")
        
        with open(os.path.join(outfolder, "best_model_path.txt"), "w") as f:
            f.write(best_model_path)
            
        print(f"python test.py --cfg {args.cfg} --model_path {best_model_path} --outname {args.outname} --data {args.data} --gpu 5")

if __name__ == '__main__':
    main() 