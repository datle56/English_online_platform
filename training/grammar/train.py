import argparse
from transformers import AutoTokenizer, get_scheduler
from gector import (
    GECToR,
    GECToRDataset,
    GECToRConfig,
    load_dataset,
    load_vocab
)
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json
import numpy as np
import random
from collections import OrderedDict

def need_add_prefix_space(model_id):
    for m in ['roberta', 'deberta', 'microsoft/deberta-v3-xsmall']:
        if m in model_id:
            return True
    return False

def has_add_pooling_layer(model_id):
        for m in ['xlnet', 'deberta']:
            if m in model_id:
                return False
        return True

def solve_model_id(model_id):
    if model_id == 'deberta-base':
        return 'microsoft/deberta-base'
    elif model_id == 'deberta-large':
        return 'microsoft/deberta-large'
    elif model_id == 'mamba-130m':
        return 'state-spaces/mamba-130m-hf'
    else:
        return model_id

def train(
    model,
    loader,
    optimizer,
    lr_scheduler,
    epoch,
    step_scheduler,
    device
):
    log = {
        'loss': 0,
        'accuracy': 0,
        'accuracy_d': 0
    }
    model.train()
    pbar = tqdm(loader, total=len(loader))
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        if step_scheduler:
            lr_scheduler.step()
        log['loss'] += loss.item()
        log['accuracy'] += outputs.accuracy.item()
        log['accuracy_d'] += outputs.accuracy_d.item()
        
        pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
        pbar.set_postfix(OrderedDict(
            loss=loss.item(),
            accuracy=outputs.accuracy.item(),
            accuracy_d=outputs.accuracy_d.item(),
            lr=optimizer.param_groups[0]['lr']
        ))
    return {k: v/len(loader) for k, v in log.items()}

@torch.no_grad()
def valid(
    model,
    loader,
    epoch,
    device
):
    log = {
        'loss': 0,
        'accuracy': 0,
        'accuracy_d': 0
    }
    model.eval()
    pbar = tqdm(loader, total=len(loader))
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        log['loss'] += outputs.loss.item()
        log['accuracy'] += outputs.accuracy.item()
        log['accuracy_d'] += outputs.accuracy_d.item()
        
        pbar.set_description(f'[Epoch {epoch}] [VALID]')
        pbar.set_postfix(OrderedDict(
            loss=outputs.loss.item(),
            accuracy=outputs.accuracy.item(),
            accuracy_d=outputs.accuracy_d.item(),
        ))
    return {k: v/len(loader) for k, v in log.items()}

def main(args):
    # To easily specify the model_id 
    args.model_id = solve_model_id(args.model_id)
    print('Start ...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    if args.restore_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir+"/tokenizer")
    else:
        add_prefix_space = need_add_prefix_space(args.model_id)
        if add_prefix_space:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                add_prefix_space=add_prefix_space
            )
        else: 
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                add_prefix_space=add_prefix_space
            )
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['$START']}
    )
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    print('Loading datasets...')
    dataset_args = {
        'input_file': args.train_file,
        'tokenizer': tokenizer,
        'delimeter': args.delimeter,
        'additional_delimeter': args.additional_delimeter,
        'max_length': args.max_len
    }
    if args.train_bin_path is not None:
        train_dataset = GECToRDataset(tokenizer=tokenizer,
                                    max_length=args.max_len)
        train_dataset.load_bin(args.train_bin_path)
    else: train_dataset = load_dataset(**dataset_args)
    
    dataset_args['input_file'] = args.valid_file
    if args.valid_bin_path is not None:
        valid_dataset = GECToRDataset(tokenizer=tokenizer,
                                    max_length=args.max_len)
        valid_dataset.load_bin(args.valid_bin_path)
    else: valid_dataset = load_dataset(**dataset_args)

    del dataset_args

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            cuda_device = list(range(torch.cuda.device_count()))
        else:
            cuda_device = 0
    else:
        cuda_device = -1

    if args.restore_dir is not None:
        # If you specify path or id to --restore_dir, the model loads weights and vocab.
        model = torch.load(args.restore_dir+'/last_model.pth')
    else:
        # Otherwise, the model will be trained from scratch.
        label2id, d_label2id = load_vocab(args.restore_vocab)
        
        gector_config = GECToRConfig(
            model_id=args.model_id,
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
            d_label2id=d_label2id,
            p_dropout=args.p_dropout,
            max_length=args.max_len,
            label_smoothing=args.label_smoothing,
            has_add_pooling_layer=has_add_pooling_layer(args.model_id)
        )
        model = GECToR(config=gector_config)

        del gector_config
    # if args.pretrain_model is not None:
    #     state_dict = torch.load(args.pretrain_model)
    #     model.load_state_dict(state_dict)
        
    if args.train_bin_path is None:
        train_dataset.append_vocab(
            model.config.label2id,
            model.config.d_label2id
        )
    if args.valid_bin_path is None:
        valid_dataset.append_vocab(
            model.config.label2id,
            model.config.d_label2id
        )

    print('# instances of train:', len(train_dataset))
    
    # train_dataset(
        # tokenizer
        # srcs
        # d_labels
        # labels
        # word_masks
        # max_length
        # label2id
        # d_label2id
    #)
    print('# instances of valid:', len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    del train_dataset
    del valid_dataset
    # print(train_dataset)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * (args.n_epochs - args.n_cold_epochs) // args.accumulation,
    )

    tokenizer.save_pretrained(args.save_dir+'/tokenizer')
    max_acc = -1
    
    model = model.to(device)

    print('Start training...')

    def set_lr(optimizer, lr):
        for param in optimizer.param_groups:
            param['lr'] = lr

    logs = {'argparse': args.__dict__}
    for e in range(args.n_epochs):
        step_scheduler = e >= args.n_cold_epochs
        if e < args.n_cold_epochs:
            # Only set tune_bert if it exists in the model (avoiding model.module issue)
            if hasattr(model, "module"):
                model.module.tune_bert(False)
            else:
                model.tune_bert(False)
            set_lr(optimizer, args.cold_lr)
        elif e == args.n_cold_epochs:
            if hasattr(model, "module"):
                model.module.tune_bert(True)
            else:
                model.tune_bert(True)
            set_lr(optimizer, args.lr)

        print(f'=== Epoch {e} ===')
        train_log = train(model, train_loader, optimizer, lr_scheduler, e, step_scheduler, device)
        valid_log = valid(model, valid_loader, e, device)

        # Save the best model based on accuracy
        if valid_log['accuracy'] > max_acc:
            # print(model.state_dict())
            torch.save(model, os.path.join(args.save_dir, 'best_model.pth'))
            max_acc = valid_log['accuracy']
            valid_log['message'] = 'The best checkpoint has been updated.'

        torch.save(model, os.path.join(args.save_dir, 'last_model.pth'))
        logs[f'Epoch {e}'] = {'train_log': train_log, 'valid_log': valid_log}
        with open(os.path.join(args.save_dir, 'log.json'), 'w') as f:
            json.dump(logs, f, indent=2)
    print('finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='data_processed/sample.txt')
    parser.add_argument('--valid_file', default='data_processed/sample.txt')
    parser.add_argument('--train_bin_path', default=None)
    parser.add_argument('--valid_bin_path', default=None)
    parser.add_argument('--pretrain_model', default=None)
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--delimeter', default='SEPL|||SEPR')
    parser.add_argument('--additional_delimeter', default='SEPL__SEPR')
    parser.add_argument('--restore_dir')
    parser.add_argument('--restore_vocab', default= 'data/vocabulary')
    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--n_max_labels', type=int, default=5000)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--p_dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cold_lr', type=float, default=1e-3)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--n_cold_epochs', type=int, default=0)
    parser.add_argument('--num_warmup_steps', type=int, default=500)
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
#python train.py --model_id mamba-130m
# bert-base-cased roberta-base deberta-base xlnet-base-cased
# bert-large-cased roberta-large deberta-large xlnet-large-cased
#data train 8865347