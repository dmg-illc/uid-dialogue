import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import time
import torch
from torch.nn.functional import log_softmax
from decimal import Decimal
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from lm_utils import add_context

logger = logging.getLogger(__name__)


def equal_length(lists):
    list_lens = [len(l) for l in lists]
    for length in list_lens[1:]:
        if length != list_lens[0]:
            return False
    return True


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="The path to the input pandas dataframe."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output directory path for a csv file containing the output dataframe."
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="The directory path of a trained language model."
    )
    parser.add_argument(
        "--window_size", type=int, required=True,
        help="The size of the Transformer's context window."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0,
        help="The learning rate for Stochastic Gradient Descent."
    )
    parser.add_argument(
        "--n_dial", default=0, type=int,
        help="The maximum number of dialogues to process. Default 0: all dialogues in the input data."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--log_every", type=int, default=1000000000,
        help="Number of turns between each logging event. Default: no logging."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="For distributed training: local_rank."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

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

    # multi-gpu
    # if n_gpu > 1:
    #     lm = torch.nn.DataParallel(lm)

    # Distributed
    # if args.local_rank != -1:
    #     lm = torch.nn.parallel.DistributedDataParallel(
    #         lm, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    #     )

    logger.warning(args)

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    LOG_2 = torch.log(torch.tensor(2.))
    output_data = []

    logger.warning('Compute entropy...')
    for i, dial_id in enumerate(data):
        if args.n_dial and i >= args.n_dial:
            break
        logger.warning('Dialogue {}/{}'.format(i + 1, len(data)))

        lm = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, return_dict=True)
        lm.to(device)
        if args.learning_rate > 0:
            optimizer = torch.optim.SGD(lm.parameters(), lr=args.learning_rate)

        context = None

        for turn_id in data[dial_id]:
            if int(turn_id) > 0 and int(turn_id)% args.log_every == 0:
                logger.warning('Turn {}/{}'.format(int(turn_id) + 1, len(data[dial_id])))

            speaker_label, speaker_id, text = data[dial_id][turn_id]

            text = tokenizer.bos_token + text
            inputs = tokenizer(text, return_tensors="pt")
            inputs_w_ctx, labels_w_ctx, context = add_context(inputs, context, args.window_size, tokenizer, device)

            lm.eval()
            with torch.no_grad():
                try:
                    outputs = lm(**inputs_w_ctx)
                except RuntimeError:
                    logger.warning('RuntimeError: {}'.format(text))
                    logger.warning(inputs_w_ctx)
                    output_data.append((
                        dial_id,
                        int(turn_id),
                        0,
                        [],
                        []
                    ))
                    continue

            if args.learning_rate > 0:
                lm.train()
                optimizer.zero_grad()
                try:
                    _outputs = lm(**inputs_w_ctx, labels=labels_w_ctx)
                except RuntimeError:
                    logger.warning('RuntimeError: {}'.format(text))
                    logger.warning(inputs_w_ctx)
                    output_data.append((
                        dial_id,
                        int(turn_id),
                        0,
                        [],
                        []
                    ))
                    continue

                loss = _outputs.loss  # todo: collect losses
                loss.backward()
                optimizer.step()

            logp_w = log_softmax(outputs.logits, dim=-1)
            logp_w = logp_w[0, :, :]
            logp_w /= LOG_2

            turn_tokens = []
            turn_token_surprisals = []
            for t in range(args.window_size, logp_w.shape[0] - 1):
                w_id = inputs_w_ctx['input_ids'][0, t + 1]
                turn_tokens.append(tokenizer.convert_ids_to_tokens(w_id.item()))
                turn_token_surprisals.append(- logp_w[t, w_id].item())

            output_data.append((
                dial_id,
                int(turn_id),
                len(turn_tokens),
                turn_tokens,
                turn_token_surprisals
            ))

    dataframe = pd.DataFrame(output_data, columns=[
        'Dialogue ID',
        'Turn index',
        'N tokens',
        'Tokens',
        'Surprisal'
    ])

    out_file_name = os.path.join(args.out_path, 'surprisal_SBNC_gpt2_{}_{}.csv'.format(
        args.window_size,
        '{:.0e}'.format(Decimal(args.learning_rate))
    ))

    dataframe.to_csv(
        out_file_name,
        index=False,
    )

    logger.warning('Output path: {}'.format(out_file_name))
