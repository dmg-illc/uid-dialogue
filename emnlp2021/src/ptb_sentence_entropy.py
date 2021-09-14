import argparse
import logging
import os
import time
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, GPT2Tokenizer, RobertaForMaskedLM, GPT2LMHeadModel, AutoTokenizer, \
    AutoModelForCausalLM, TransfoXLTokenizer, TransfoXLLMHeadModel
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from lm_utils import pad, MaptaskSentenceDataset

logger = logging.getLogger(__name__)


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
        elif args.model_name.lower() == 'transfo-xl':
            args.model_path = 'transfo-xl-wt103'
        elif args.model_name.lower() == 'roberta':
            args.model_path = 'roberta-base'
        else:
            args.model_path = args.model_name

    # Load LM and tokenizer
    if args.model_name.lower() == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
        lm = RobertaForMaskedLM.from_pretrained(args.model_path, return_dict=True)
    elif args.model_name.lower() == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        lm = GPT2LMHeadModel.from_pretrained(args.model_path, return_dict=True)
    elif args.model_name.lower() == 'dialogpt':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        lm = AutoModelForCausalLM.from_pretrained(args.model_path)
    elif args.model_name.lower() == 'transfo-xl':
        tokenizer = TransfoXLTokenizer.from_pretrained(args.model_path)
        lm = TransfoXLLMHeadModel.from_pretrained(args.model_path)
    # else:
    #     tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    #     lm = RobertaForMaskedLM.from_pretrained(args.model_name)
    #     #raise ValueError('Incorrect model name: {}. Available: gpt2, roberta, and dialogpt'.format(args.model_name))

    lm.to(device)

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training
        # download model & vocab
        torch.distributed.barrier()

    args.batch_size = args.per_gpu_batch_size * max(1, n_gpu)

    if args.model_name == 'transfo-xl':
        def collate(batch):
            return [
                pad(tokenizer, [item[0] for item in batch], attention_mask=False),
                [item[1] for item in batch]
            ]
    else:
        def collate(batch):
            return [
                pad(tokenizer, [item[0] for item in batch]),
                [item[1] for item in batch]
            ]

    data = MaptaskSentenceDataset(dataframe, tokenizer, max_seq_len=args.max_seq_len)
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

    unk_id = tokenizer.convert_tokens_to_ids('<unk>')

    logger.warning('Compute entropy...')
    iterator = tqdm(dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(iterator):

        inputs, idx = batch
        inputs['input_ids'] = inputs['input_ids'].to(device)
        try:
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
        except KeyError:
            pass

        batch_sumlogp = [0 for _ in idx]
        batch_lengths = [0 for _ in idx]
        max_sent_len = inputs['input_ids'].shape[1]

        lm.eval()

        if args.model_name.lower() == 'roberta':
            # for every next token...
            for token_index in range(1, max_sent_len):

                # mask next token (for every sentence in the batch)
                masked_inputs = {
                    'input_ids': inputs['input_ids'].clone(),
                    'attention_mask': inputs['attention_mask'].clone()
                }

                masked_inputs['input_ids'][:, token_index] = tokenizer.mask_token_id
                if args.right_context >= 0:
                    masked_inputs['attention_mask'][:, token_index + 1 + args.right_context:] = torch.tensor(0)

                masked_inputs['input_ids'] = masked_inputs['input_ids'].to(device)
                masked_inputs['attention_mask'] = masked_inputs['attention_mask'].to(device)

                # run LM to obtain next token probabilities (for every sentence in the batch)
                with torch.no_grad():
                    outputs = lm(**masked_inputs)  # n_sentences, max_sent_len, vocab_size

                # get log probability of next token (for every sentence in the batch)
                logp_w = log_softmax(outputs.logits[:, token_index, :], dim=-1)  # n_sentences, vocab_size
                logp_w /= log_2  # change to base 2

                # for every sentence in the batch...
                for s_id in range(inputs['input_ids'].shape[0]):

                    # get next token id (for this sentence)
                    w_id = inputs['input_ids'][s_id, token_index]

                    # skip special tokens (BOS, EOS, PAD)
                    if w_id in tokenizer.all_special_ids and w_id != unk_id:
                        continue

                    # increase sentence length if next token is not special token
                    batch_lengths[s_id] += 1
                    # increase non-normalised log probability of the sentence
                    batch_sumlogp[s_id] += logp_w[s_id, w_id].item()
        else:
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

                    # increase sentence length if next token is not special token
                    batch_lengths[s_id] += 1
                    # increase non-normalised log probability of the sentence
                    batch_sumlogp[s_id] += logp_w[s_id, token_index, w_id].item()

        sentence_sumlogp.extend(batch_sumlogp)
        sentence_length.extend(batch_lengths)
        sentence_idx.extend(idx)

    iterator.close()
    logger.warning('--- %s seconds ---' % (time.time() - start_time))

    sentence_sumlogp = - np.array(sentence_sumlogp)
    sentence_length = np.array(sentence_length)
    sentence_avglogp = sentence_sumlogp / sentence_length
    sentence_idx = np.array(sentence_idx)

    dataframe.loc[:, 'h'] = np.nan
    dataframe.loc[:, 'normalised_h'] = np.nan
    dataframe.loc[:, 'length'] = np.nan

    for idx, h, n_h, len in zip(sentence_idx, sentence_sumlogp, sentence_avglogp, sentence_length):
        dataframe.loc[idx, 'h'] = h
        dataframe.loc[idx, 'normalised_h'] = n_h
        dataframe.loc[idx, 'length'] = len

    h_bar = dataframe.groupby('length').agg({"normalised_h": "mean"})
    xu_h = []
    for index, row in dataframe.iterrows():
        try:
            xu_h.append(row['normalised_h'] / h_bar.loc[row['length'], 'normalised_h'])
        except KeyError:
            xu_h.append(np.nan)

    dataframe.loc[:, 'xu_h'] = xu_h

    out_file_name = os.path.join(args.out_path, '{}_{}_{}_{}'.format(
        args.sections,
        args.model_path.replace('/', '-'),
        args.right_context,
        args.max_seq_len,
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
        help="The input data directory."
    )
    parser.add_argument(
        "--sections", default="dev", type=str, required=True,
        help="The PTB sections to use: 'train', 'test', 'dev', or 'all'."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output data directory."
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="The language model name: 'gpt2', 'roberta', 'dialogpt' or 'transfo-xl'."
    )
    parser.add_argument(
        "--model_path", type=str, required=False,
        help="The directory path of a trained language model."
    )
    parser.add_argument(
        "--right_context", default=-1, type=int,
        help="The size of the right context window for a bidirectional language model. -1 for the entire context."
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
    args = parser.parse_args()

    logger.warning('Load corpus from {}'.format(args.data_path))
    dev_sections = ["01"]
    train_sections = ["{:02d}".format(sec) for sec in range(0, 21)]
    test_sections = ["{:02d}".format(sec) for sec in range(21, 25)]
    all_sections = ["{:02d}".format(sec) for sec in range(0, 25)]

    if args.sections == 'train':
        sections = train_sections
    elif args.sections == 'test':
        sections = test_sections
    elif args.sections == 'dev':
        sections = dev_sections
    elif args.sections == 'all':
        sections = all_sections
    else:
        raise ValueError("Invalid sections identifier '%s'" % args.section)

    sentences = []
    positions_in_doc = []
    positions_in_par = []
    doc_ids = []

    doc_id = 0
    for sec in sections:
        for root, _, files in os.walk(os.path.join(args.data_path, sec)):
            for file in files:
                doc_id += 1
                with open(os.path.join(root, file), 'r', encoding="ISO-8859-1") as f:
                    # reset for each new document
                    position_in_doc = 1
                    position_in_par = 1
                    for line in f:
                        line = line.strip('\n').strip()
                        if line == '.START':
                            continue
                        if line:
                            sentences.append(line)
                            positions_in_doc.append(position_in_doc)
                            positions_in_par.append(position_in_par)
                            doc_ids.append(doc_id)
                            position_in_doc += 1
                            position_in_par += 1
                        else:
                            position_in_par = 1  # reset at each paragraph break

    logger.warning("Number of sentences: {}".format(len(sentences)))
    print("Max position in document: {}".format(max(positions_in_doc)))
    print("Max position in paragraph: {}".format(max(positions_in_par)))

    dataset = list(zip(doc_ids, positions_in_doc, positions_in_par, sentences))
    dataframe = pd.DataFrame(dataset, columns=['doc_id', 'position_in_document', 'position_in_paragraph', 'sentence'])

    out_file_name = compute_entropy(args, dataframe)
    logger.warning('Output: {}.csv'.format(out_file_name))
