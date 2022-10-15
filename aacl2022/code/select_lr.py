import argparse
import logging
import math
import numpy as np
import os
import pandas as pd
import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from lm_utils import add_context

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def tokenize(text, tokenizer):
    return tokenizer(tokenizer.bos_token + text, return_tensors="pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="The path to the folder containing train and eval dialogue-specific datasets."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output directory path for a csv file containing the output dataframe."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=False,
        help="The directory path of a trained language model."
    )
    parser.add_argument(
        "--window_size", default=50, type=int,
        help="The size of the context window."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="For distributed training: local_rank."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    logging.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        raise NotImplementedError('Local rank {}'.format(args.local_rank))
        # torch.cuda.set_device(args.local_rank)
        # device = torch.device('cuda', args.local_rank)
        # torch.distributed.init_process_group(backend='nccl')
        # n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s',
        args.local_rank,
        device,
        n_gpu,
        bool(args.local_rank != -1)
    )

    # Set seeds across modules
    set_seed(args.seed, n_gpu)

    # if args.local_rank not in [-1, 0]:
    #     # Make sure only the first process in distributed training will download model & vocab
    #     torch.distributed.barrier()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

    # if args.local_rank == 0:
    #     # End of barrier to make sure only the first process in distributed training
    #     # download model & vocab
    #     torch.distributed.barrier()

    ppls = []
    i = 0
    for filename in os.listdir(args.data_path):
        if filename.endswith("-eval.txt"):
            continue
        dial_id, _ = filename.split('.txt')

        i += 1
        # if i > 4:
        #     break
        logger.warning('Fold {}'.format(i))

        with open(os.path.join(args.data_path, filename), 'r') as f:
            training_data = [line.strip() for line in f.readlines() if line]

        with open(os.path.join(args.data_path, '{}-eval.txt').format(dial_id), 'r') as f:
            eval_data = [line.strip() for line in f.readlines() if line]

        lm = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, return_dict=True)
        lm.to(device)

        # Self perplexity before
        lm.eval()
        train_losses_before = []
        context = None
        for j, utterance in enumerate(training_data):
            # if j >= 50:
            #     break
            inputs = tokenize(utterance, tokenizer=tokenizer)
            inputs_w_ctx, labels_w_ctx, context = add_context(inputs, context, args.window_size, tokenizer, device)
            with torch.no_grad():
                outputs = lm(**inputs_w_ctx, labels=labels_w_ctx)
            train_losses_before.append(outputs.loss.item())
        try:
            train_perplexity_before = math.exp(np.mean(train_losses_before))
        except OverflowError:
            train_perplexity_before = float("inf")

        # Others perplexity before
        eval_losses_before = []
        for j, utterance in enumerate(eval_data):
            # if j >= 50:
            #     break
            inputs = tokenize(utterance, tokenizer=tokenizer)
            inputs_w_ctx, labels_w_ctx, context = add_context(inputs, context, args.window_size, tokenizer, device)
            with torch.no_grad():
                outputs = lm(**inputs_w_ctx, labels=labels_w_ctx)
            eval_losses_before.append(outputs.loss.item())
        try:
            eval_perplexity_before = math.exp(np.mean(eval_losses_before))
        except OverflowError:
            eval_perplexity_before = float("inf")

        for lr in [0.00001]:  #, 0.0001, 0.001, 0.01, 0.1, 1]:
            logger.warning('lr {}'.format(lr))
            # Self perplexity after
            optimizer = torch.optim.SGD(lm.parameters(), lr=lr)
            train_losses_after = []
            for j, utterance in enumerate(training_data):
                # if j >= 50:
                #     break
                inputs = tokenize(utterance, tokenizer=tokenizer)
                inputs_w_ctx, labels_w_ctx, context = add_context(inputs, context, args.window_size, tokenizer, device)
                lm.eval()
                with torch.no_grad():
                    outputs = lm(**inputs_w_ctx, labels=labels_w_ctx)
                    train_losses_after.append(outputs.loss.item())
                lm.train()
                optimizer.zero_grad()
                _outputs = lm(**inputs_w_ctx, labels=labels_w_ctx)
                _outputs.loss.backward()
                optimizer.step()
            try:
                train_perplexity_after = math.exp(np.mean(train_losses_after))
            except OverflowError:
                train_perplexity_after = float("inf")

            # Others perplexity after
            lm.eval()
            eval_losses_after = []
            for j, utterance in enumerate(eval_data):
                # if j >= 50:
                #     break
                inputs = tokenize(utterance, tokenizer=tokenizer)
                inputs_w_ctx, labels_w_ctx, context = add_context(inputs, context, args.window_size, tokenizer, device)
                with torch.no_grad():
                    outputs = lm(**inputs_w_ctx, labels=labels_w_ctx)
                eval_losses_after.append(outputs.loss.item())
            try:
                eval_perplexity_after = math.exp(np.mean(eval_losses_after))
            except OverflowError:
                eval_perplexity_after = float("inf")

            ppls.append((
                dial_id,
                lr,
                train_perplexity_before,
                train_perplexity_after,
                eval_perplexity_before,
                eval_perplexity_after
            ))

    df = pd.DataFrame(ppls, columns=[
        'Dialogue ID', 'Learning rate', 'Train ppl before', 'Train ppl after', 'Eval ppl before', 'Eval ppl after'
    ])
    df.fillna(0, inplace=True)

    df.to_csv(
        os.path.join(args.out_path, 'ppl_lr_selection.csv'),
        index=False,
    )
    logger.warning('Output path: {}'.format(os.path.join(args.out_path, 'ppl_lr_selection.csv')))
