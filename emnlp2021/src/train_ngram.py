import argparse
import dill as pickle
import logging
import os
import time
from nltk import ngrams
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from transformers import RobertaTokenizer
from lm_utils import tokenize_documents_for_ngram

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", type=str,
        help="The training data as a preprocessed txt file. Newline after each sentence / utterance; empty line after "
             "each document / conversation."
    )
    parser.add_argument(
        "--test_data_path", type=str,
        help="The evaluation data as a preprocessed txt file. Newline after each sentence / utterance; empty line after "
             "each document / conversation."
    )
    parser.add_argument(
        "--lm_path", type=str,
        help="The path to the pickled language model."
    )
    parser.add_argument(
        "--out_path", type=str,
        help="The output data directory."
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="The ngram order."
    )
    parser.add_argument(
        "--discount", default=0.1, type=float,
        help="The discounting value for the Kneser-Ney Interpolated LM."
    )
    parser.add_argument(
        "--train", action='store_true',
        help="Whether to train and save a new language model."
    )
    parser.add_argument(
        "--eval", action='store_true',
        help="Whether to evaluate the language model."
    )
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if args.train:
        train_tokenized = tokenize_documents_for_ngram(tokenizer, args.train_data_path)
    if args.eval:
        dev_tokenized = tokenize_documents_for_ngram(tokenizer, args.test_data_path)

    logger.warning('N = {}, discount = {}'.format(args.n, args.discount))
    if args.train:
        lm = KneserNeyInterpolated(order=args.n, discount=args.discount)  # default discount
    if not args.train and args.eval:
        with open(args.lm_path, 'rb') as f:
            lm = pickle.load(f)
        logger.warning('Loaded language model from {}'.format(args.lm_path))

    if args.train:
        logger.warning('Training')
        start_time = time.time()
        train, vocab = padded_everygram_pipeline(args.n, train_tokenized)
        lm.fit(train, vocab)
        logger.warning('--- {:.2f} seconds ---'.format(time.time() - start_time))

        file_name = 'KNI_{}_{}.pkl'.format(args.n, args.discount)
        file_path = os.path.join(args.out_path, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(lm, f)
        logger.warning('Model saved to {}'.format(file_path))

    if args.eval:
        dev = []
        for text in dev_tokenized:
            dev.extend(list(ngrams(text, args.n)))

        logger.warning('Evaluation')
        start_time = time.time()
        logger.warning('ppl: {:.2f}'.format(lm.perplexity(dev)))
        logger.warning('--- {:.2f} seconds ---'.format(time.time() - start_time))
