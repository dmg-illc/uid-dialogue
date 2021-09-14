import argparse
import dill as pickle
import logging
import os
import pandas as pd
import time
from nltk import ngrams
from transformers import RobertaTokenizer
from lm_utils import tokenize_documents_for_ngram

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="The input data as a preprocessed txt file. Newline after each sentence / utterance; empty line after "
             "each document / conversation."
    )
    parser.add_argument(
        "--lm_path", type=str, required=True,
        help="The path to the pickled language model."
    )
    parser.add_argument(
        "--out_path", type=str, required=True,
        help="The output data directory."
    )
    parser.add_argument(
        "--n", default=3, type=int,
        help="The ngram order."
    )
    parser.add_argument(
        "--max_seq_len", default=None, type=int,
        help="The maximum number of tokens to process in a sentence."
    )
    args = parser.parse_args()

    with open(args.lm_path, 'rb') as f:
        lm = pickle.load(f)
    logger.warning('Loaded language model from {}'.format(args.lm_path))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    docs_tokenized = tokenize_documents_for_ngram(
        tokenizer, args.data_path, split_sentences=True, max_seq_len=args.max_seq_len)

    logger.warning('Compute entropy')
    start_time = time.time()

    token_dataset = []
    sentence_dataset = []

    for doc_tokens in docs_tokenized:
        sentence_position = 0
        token_position = 0

        for j, sent_tokens in enumerate(doc_tokens):
            sentence_position += 1
            sentence_length = 0
            sentence_entropy = 0

            # connect sentences for token-level measurements
            if j > 0:
                sent_tokens = doc_tokens[j - 1][-(args.n - 1):] + sent_tokens

            for ngram in ngrams(sent_tokens, args.n):

                if ngram[-1] in ['<s>', '</s>']:
                    continue

                token_position += 1
                entropy = - lm.logscore(ngram[-1], list(ngram[:-1]))
                token_dataset.append((entropy, token_position))

                # no connection between sentences for sentence-level measurements
                if '</s>' in ngram:
                    continue
                sentence_entropy += entropy
                sentence_length += 1
            sentence_dataset.append((sentence_entropy, sentence_position, sentence_length))

    logger.warning('--- %s seconds ---' % (time.time() - start_time))

    token_df = pd.DataFrame({
        'position': [x[1] for x in token_dataset],
        'entropy': [x[0] for x in token_dataset]
    })
    sentence_df = pd.DataFrame({
        'position': [x[1] for x in sentence_dataset],
        'entropy': [x[0] for x in sentence_dataset],
        'length': [x[2] for x in sentence_dataset]
    })
    out_file_name = os.path.join(args.out_path, 'n{}_{}'.format(
        args.n,
        args.lm_path.split('/')[-1][:-4]
    ))
    token_df.to_csv(
        '{}_token.zip'.format(out_file_name),
        index=False,
        compression=dict(
            method='zip', archive_name='{}.csv'.format(out_file_name))
    )
    sentence_df.to_csv(
        '{}_sentence.zip'.format(out_file_name),
        index=False,
        compression=dict(
            method='zip', archive_name='{}.csv'.format(out_file_name))
    )
