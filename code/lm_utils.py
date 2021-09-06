import logging
from string import ascii_uppercase
import torch
from collections import Counter
from nltk import ngrams
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizer, GPT2Tokenizer, RobertaForMaskedLM, GPT2LMHeadModel

logger = logging.getLogger(__name__)



# FOR SENTENCE-LEVEL EXPERIMENTS
# ---------------------------------------------------------------

def pad(tokenizer, batch, attention_mask=True):
    max_len = max([x['input_ids'].shape[1] for x in batch])
    if tokenizer.pad_token_id:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = tokenizer.eos_token_id

    padded_input_ids = []
    if attention_mask:
        padded_attention_masks = []

    for x in batch:
        x_len = x['input_ids'].shape[1]
        difference = max_len - x_len
        padded_input_ids.append(torch.cat((
            x['input_ids'],
            torch.tensor(difference * [pad_id], dtype=torch.long).unsqueeze(0)),
            dim=1))
        if attention_mask:
            padded_attention_masks.append(torch.tensor(x_len * [1] + difference * [0], dtype=torch.long).unsqueeze(0))

    if attention_mask:
        return {'input_ids': torch.cat(padded_input_ids, dim=0),
                'attention_mask': torch.cat(padded_attention_masks, dim=0)}
    else:
        return {'input_ids': torch.cat(padded_input_ids, dim=0)}


class MaptaskSentenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, add_speaker_ids=False):
        super(MaptaskSentenceDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_speaker_ids = add_speaker_ids

        logger.warning('Tokenize...')
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            sentence = row['text']
            if self.add_speaker_ids:
                sentence = '{}: {}'.format(row['speaker'], sentence)

            if tokenizer.bos_token:
                sentence = tokenizer.bos_token + sentence + tokenizer.eos_token
            else:
                sentence = tokenizer.eos_token + sentence + tokenizer.eos_token

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )
            self.data.append((inputs, idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



# FOR TOKEN-LEVEL EXPERIMENTS
# ---------------------------------------------------------------

def add_special_tokens(tokenizer, text):
    if tokenizer.eos_token == tokenizer.bos_token:
        return tokenizer.bos_token + text
    else:
        return tokenizer.bos_token + text + tokenizer.eos_token


def count_padding(tokenizer, array, left_only=False):
    left_padding = 0
    for x in array:
        if left_only and x == tokenizer.bos_token_id:
            left_padding += 1
        elif not left_only and x == tokenizer.pad_token_id:
            left_padding += 1
        else:
            break

    if left_only and left_padding:  # pad token == bos token
        left_padding -= 1
    if left_only:
        return left_padding
    if left_padding == len(array) or (left_only and left_padding == len(array) - 1):
        return left_padding, 0

    right_padding = 0
    for x in array[::-1]:
        if x == tokenizer.pad_token_id:
            right_padding += 1
        else:
            break

    return left_padding, right_padding


def tokenize_documents(tokenizer, path, special_tokens=True):
    """
    Each document is stored as a list of token ids.
    Sentence breaks are marked by special token ids.

    :return list of lists of document token ids
    """
    docs_tokenized = []
    with open(path, 'r') as f:
        doc_tokens = []
        for line in f.readlines():
            line = line.strip('\n')
            line = line.strip()
            if line:
                if special_tokens:
                    inputs = tokenizer(add_special_tokens(tokenizer, line), add_special_tokens=False)
                else:
                    inputs = tokenizer(line, add_special_tokens=False)
                doc_tokens.extend(inputs['input_ids'])
            else:
                docs_tokenized.append(doc_tokens)
                doc_tokens = []
        if doc_tokens:
            docs_tokenized.append(doc_tokens)
    return docs_tokenized


def tokenize_documents_for_ngram(tokenizer, path, split_sentences=False, max_seq_len=None):
    docs_tokenized = []
    with open(path, 'r') as f:
        doc_tokens = []
        for line in f.readlines():
            line = line.strip('\n')
            line = line.strip()
            if line:
                inputs = tokenizer(line, add_special_tokens=False)
                token_ids = inputs['input_ids']
                if max_seq_len:
                    tokens = ['<s>'] + tokenizer.convert_ids_to_tokens(token_ids)[:max_seq_len] + ['</s>']
                else:
                    tokens = ['<s>'] + tokenizer.convert_ids_to_tokens(token_ids) + ['</s>']
                if split_sentences:
                    doc_tokens.append(tokens)
                else:
                    doc_tokens.extend(tokens)
            else:
                docs_tokenized.append(doc_tokens)
                doc_tokens = []
        if doc_tokens:
            docs_tokenized.append(doc_tokens)
    return docs_tokenized


class UnidirectionalDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, tokenizer, left_context, special_tokens=True, skip_speaker_ids=False):
        super(UnidirectionalDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        if skip_speaker_ids:
            self.speaker_ids = self.tokenizer.convert_tokens_to_ids([c for c in ascii_uppercase])
            self.colon_id = self.tokenizer.convert_tokens_to_ids([':'])[0]

        logger.warning('Tokenize...')

        # each document is stored as a list of token ids
        # sentence breaks are marked by special token ids
        docs_tokenized = tokenize_documents(self.tokenizer, data_path, special_tokens)

        # slide window over the token ids of each document
        # prepare inputs and collect token positions
        for doc_tokens in docs_tokenized:
            position = 0
            # token window has an extra token at the end, just to check if it's a colon
            for token_window in ngrams(
                    doc_tokens, left_context + 2,
                    pad_left=True, left_pad_symbol=self.tokenizer.bos_token_id):

                # skip special tokens (BOS, PAD)
                if token_window[left_context] in self.tokenizer.all_special_ids:
                    continue
                # skip speaker identifier
                if skip_speaker_ids \
                        and token_window[left_context] in self.speaker_ids \
                        and token_window[left_context - 1] == tokenizer.bos_token_id \
                        and token_window[left_context + 1] == self.colon_id:
                    continue
                # skip colon after speaker identifier
                if skip_speaker_ids \
                        and token_window[left_context] == self.colon_id \
                        and token_window[left_context - 1] in self.speaker_ids \
                        and token_window[left_context - 2] == tokenizer.bos_token_id:
                    continue
                # token window had an extra token at the end, just to check if it's a colon
                token_window = token_window[:-1]

                position += 1
                token_window, target_id = token_window[:-1], token_window[-1]
                n_pads = count_padding(self.tokenizer, token_window, left_only=True)

                inputs = {
                    'input_ids': torch.tensor(
                        token_window, dtype=torch.long).unsqueeze(0),
                    'attention_mask': torch.tensor(
                        n_pads * [0] + (left_context - n_pads) * [1], dtype=torch.long).unsqueeze(0)
                }

                self.data.append((inputs, target_id, position))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs, target_id, position = self.data[index]
        return inputs, target_id, position

    def apply_cutoff(self, cutoff):
        if cutoff <= 1:
            return
        pos_counter = Counter([item[2] for item in self.data])
        tmp_data = []
        for inputs, target_id, position in self.data:
            if pos_counter[position] >= cutoff:
                tmp_data.append((inputs, target_id, position))
        self.data = tmp_data


class BidirectionalDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, tokenizer, left_context, right_context, skip_speaker_ids=False):
        super(BidirectionalDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        if skip_speaker_ids:
            self.speaker_ids = self.tokenizer.convert_tokens_to_ids([c for c in ascii_uppercase])
            self.colon_id = self.tokenizer.convert_tokens_to_ids([':'])[0]

        logger.warning('Tokenize...')

        # each document is stored as a list of token ids
        # sentence breaks are marked by special token ids
        docs_tokenized = tokenize_documents(self.tokenizer, data_path)

        # slide window over the token ids of each document
        # prepare inputs and collect token positions
        for doc_tokens in docs_tokenized:
            position = 0
            for token_window in ngrams(
                    doc_tokens, left_context + 1 + right_context,
                    pad_left=True, left_pad_symbol=self.tokenizer.pad_token_id,
                    pad_right=True, right_pad_symbol=self.tokenizer.pad_token_id):

                # skip special tokens (BOS, EOS, PAD)
                if token_window[left_context] in self.tokenizer.all_special_ids:
                    continue
                # skip speaker identifier
                if skip_speaker_ids \
                        and token_window[left_context] in self.speaker_ids \
                        and token_window[left_context - 1] == tokenizer.bos_token_id \
                        and token_window[left_context + 1] == self.colon_id:
                    continue
                # skip colon after speaker identifier
                if skip_speaker_ids \
                        and token_window[left_context] == self.colon_id \
                        and token_window[left_context - 1] in self.speaker_ids \
                        and token_window[left_context - 2] == tokenizer.bos_token_id:
                    continue

                position += 1
                token_window = list(token_window)
                target_id = token_window[left_context]
                token_window[left_context] = self.tokenizer.mask_token_id
                left_pads, right_pads = count_padding(self.tokenizer, token_window)
                no_pads = left_context + 1 + right_context - left_pads - right_pads

                inputs = {
                    'input_ids': torch.tensor(
                        token_window, dtype=torch.long).unsqueeze(0),
                    'attention_mask': torch.tensor(
                        left_pads * [0] + no_pads * [1] + right_pads * [0], dtype=torch.long).unsqueeze(0)
                }

                self.data.append((inputs, target_id, position))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs, target_id, position = self.data[index]
        return inputs, target_id, position

    def apply_cutoff(self, cutoff):
        if cutoff <= 1:
            return
        pos_counter = Counter([item[2] for item in self.data])
        tmp_data = []
        for inputs, target_id, position in self.data:
            if pos_counter[position] >= cutoff:
                tmp_data.append((inputs, target_id, position))
        self.data = tmp_data


if __name__ == '__main__':

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # lm = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict=True)
    # d = BidirectionalDataset('../data/test.txt', tokenizer, 3, 3, skip_speaker_ids=True)

    tokenizer = RobertaTokenizer.from_pretrained('gpt2')
    lm = RobertaForMaskedLM.from_pretrained('gpt2', return_dict=True)
    d = UnidirectionalDataset('../data/test.txt', tokenizer, 3, skip_speaker_ids=False)

    for (inputs, target_id, position) in d:
        print(position, target_id, tokenizer.convert_ids_to_tokens([target_id]))
        print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

