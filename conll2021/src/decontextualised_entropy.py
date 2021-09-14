import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
import torch
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
from lm_utils import pad, SentenceDataset

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


def compute_entropy(args, dataframe):
    """
    Compute entropy values for a set of sentences.

    :param args: the argparse script arguments
    :param dataframe: a pandas dataframe containing the column 'sentence'
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1

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

    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    if not args.model_path:
        if args.model_name == 'dialogpt':
            args.model_path = 'microsoft/DialoGPT-small'
        else:
            args.model_path = args.model_name

    # Load LM and tokenizer
    if args.model_name.lower() == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        lm = GPT2LMHeadModel.from_pretrained(args.model_path, return_dict=True)
    elif args.model_name.lower() == 'dialogpt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        lm = AutoModelForCausalLM.from_pretrained(args.model_path)

    lm.to(device)

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training
        # download model & vocab
        torch.distributed.barrier()

    args.batch_size = args.per_gpu_batch_size * max(1, n_gpu)

    def collate(batch):
        return [
            pad(tokenizer, [item[0] for item in batch]),
            [item[1] for item in batch]
        ]

    data = SentenceDataset(dataframe, tokenizer, args.max_seq_len)
    sampler = SequentialSampler(data) if args.local_rank == -1 else DistributedSampler(data, shuffle=False)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)

    # multi-gpu
    if n_gpu > 1:
        lm = torch.nn.DataParallel(lm)

    # Distributed
    if args.local_rank != -1:
        lm = torch.nn.parallel.DistributedDataParallel(
            lm, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    log_2 = torch.log(torch.tensor(2.))

    sentence_sumlogp = []
    sentence_length = []
    sentence_idx = []
    sentence_tokens_logp = []
    sentence_tokens = []

    unk_id = tokenizer.convert_tokens_to_ids('<unk>')

    logger.warning('Compute entropy...')
    iterator = tqdm(dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(iterator):

        inputs, df_idx = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)

        batch_sumlogp = [0 for _ in df_idx]
        batch_lengths = [0 for _ in df_idx]
        batch_tokens_logp = [[] for _ in df_idx]
        batch_tokens = [[] for _ in df_idx]
        max_sent_len = inputs['input_ids'].shape[1]

        lm.eval()

        # run unidirectional LM to obtain next token probabilities (for every sentence in the batch)
        with torch.no_grad():
            outputs = lm(**inputs)  # n_sentences, max_sent_len, vocab_size

        logp_w = log_softmax(outputs.logits, dim=-1)
        logp_w /= log_2

        # for every token...
        for token_index in range(max_sent_len - 1):

            # get next token id (for every sentence in the batch)
            w_ids = inputs['input_ids'][:, token_index + 1]  # n_sentences

            # for every sentence in the batch...
            for s_id in range(inputs['input_ids'].shape[0]):

                # get next token id (for this sentence)
                w_id = w_ids[s_id]

                # skip special tokens (BOS, EOS, PAD)
                if w_id in tokenizer.all_special_ids and (w_id != unk_id or args.model_name == 'gpt2'):
                    continue

                token_logp = logp_w[s_id, token_index, w_id].item()

                batch_lengths[s_id] += 1
                batch_sumlogp[s_id] += token_logp
                batch_tokens_logp[s_id].append(token_logp)
                batch_tokens[s_id].append(tokenizer.convert_ids_to_tokens(w_id.item()))

        sentence_sumlogp.extend(batch_sumlogp)
        sentence_length.extend(batch_lengths)
        sentence_tokens.extend(batch_tokens)
        sentence_tokens_logp.extend(batch_tokens_logp)
        sentence_idx.extend(df_idx)

    iterator.close()
    logger.warning('--- %s seconds ---' % (time.time() - start_time))

    sentence_sumlogp = - np.array(sentence_sumlogp)
    sentence_length = np.array(sentence_length)
    sentence_avglogp = sentence_sumlogp / sentence_length
    sentence_idx = np.array(sentence_idx)

    assert equal_length(
        [sentence_idx, sentence_sumlogp, sentence_avglogp, sentence_length, sentence_tokens_logp, sentence_tokens])

    dataframe['h'] = sentence_sumlogp
    dataframe['normalised_h'] = sentence_avglogp
    dataframe['length'] = sentence_length
    dataframe['tokens_h'] = sentence_tokens_logp
    dataframe['tokens'] = sentence_tokens

    h_bar = dataframe.groupby('length').agg({"normalised_h": "mean"})
    xu_h = []
    for index, row in dataframe.iterrows():
        try:
            xu_h.append(row['normalised_h'] / h_bar.loc[row['length'], 'normalised_h'])
        except KeyError:
            xu_h.append(np.nan)

    dataframe.loc[:, 'xu_h'] = xu_h

    out_file_name = os.path.join(args.out_path, '{}_{}'.format(
        args.model_path.replace('/', '-'),
        args.max_seq_len
    ))
    dataframe.to_csv(
        '{}.csv'.format(out_file_name),
        index=False
    )

    return out_file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="The path to the input pandas dataframe."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output directory path for a .gzip archive containing the output dataframe."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="The language model name: 'gpt2', 'dialogpt'."
    )
    parser.add_argument(
        "--model_path", type=str, required=False,
        help="The directory path of a trained language model."
    )
    parser.add_argument(
        "--max_seq_len", default=1024, type=int,
        help="The maximum number of input tokens."
    )
    parser.add_argument(
        "--per_gpu_batch_size", default=4, type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="For distributed training: local_rank."
    )
    parser.add_argument(
        "--n_sent", default=-1, type=int,
        help="The maximum number of sentences to process. Default -1: all sentences in the input dataframe."
    )
    parser.add_argument(
        "--max_position", default=-1, type=int,
        help="The maximum sentence position."
    )

    args = parser.parse_args()

    dataframe = pd.read_csv(args.data_path)
    if args.max_position > 0:
        try:
            dataframe = dataframe[dataframe['position'] <= args.max_position]
        except KeyError:
            try:
                dataframe = dataframe[dataframe['position_in_doc'] <= args.max_position]
            except KeyError:
                dataframe = dataframe[dataframe['position_in_dialogue'] <= args.max_position]

    dataframe = dataframe[: args.n_sent]
    out_file_name = compute_entropy(args, dataframe)
    logger.warning('Output: {}.csv'.format(out_file_name))
