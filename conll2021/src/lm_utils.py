import json
import logging
from string import ascii_uppercase
import torch
import pandas as pd
from collections import Counter
from nltk import ngrams
import random
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


# class SentenceDataset(torch.utils.data.Dataset):
#
#     def __init__(self, dataset, tokenizer, max_seq_len):
#         super(SentenceDataset).__init__()
#         self.data = []
#         self.tokenizer = tokenizer
#
#         logger.warning('Tokenize...')
#         for sentence, position in tqdm(dataset, total=len(dataset)):
#             sentence = tokenizer.bos_token + sentence + tokenizer.eos_token
#             inputs = self.tokenizer(
#                 sentence,
#                 return_tensors='pt',
#                 truncation=True,
#                 add_special_tokens=False,
#                 max_length=max_seq_len + 2
#             )
#             self.data.append((inputs, position))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index]


class TreebankSentenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer, max_seq_len):
        super(TreebankSentenceDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        logger.warning('Tokenize...')
        for sentence, idx_in_doc, idx_in_par, doc_id in tqdm(dataset, total=len(dataset)):
            sentence = tokenizer.bos_token + sentence + tokenizer.eos_token
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )
            self.data.append((inputs, idx_in_doc, idx_in_par, doc_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class PhotobookSentenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer, max_seq_len):
        super(PhotobookSentenceDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        logger.warning('Tokenize...')
        for sentence, idx_in_game, idx_in_round, round_nr, game_id in tqdm(dataset, total=len(dataset)):
            sentence = tokenizer.bos_token + sentence + tokenizer.eos_token
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )
            self.data.append((inputs, idx_in_game, idx_in_round, round_nr, game_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ContextualisedDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, context_field, add_special_tokens=True):
        super(ContextualisedDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_seq_len = max_seq_len
        self.current_document = None

        logger.warning('Tokenize...')

        self.reset_current_document_context()

        current_context = ''

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            # if current_context == '':
            #     current_context = row[context_field]

            try:
                sentence = row['text'] + self.tokenizer.eos_token
            except TypeError:
                sentence = self.tokenizer.eos_token

            # print('SENT', sentence)

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                # truncation=True,
                add_special_tokens=False,
                # max_length=max_seq_len + 2
            )

            # print('TOK', tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

            if row[context_field] != current_context:
                self.reset_current_document_context()
                current_context = row[context_field]

            contextualised_inputs = {
                    'input_ids': torch.cat((self.current_document['input_ids'], inputs['input_ids']), 1),
                    'attention_mask': torch.cat((self.current_document['attention_mask'], inputs['attention_mask']), 1),
                    'end_index': self.current_document['end_index'] + inputs['input_ids'].shape[1]
            }

            # cut off context if longer than max_seq_len
            if contextualised_inputs['input_ids'].shape[1] > self.max_seq_len:
                contextualised_inputs = {
                    'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                    'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                    'end_index': self.max_seq_len - inputs['input_ids'].shape[1]
                }

            # print('dialogue')
            # print('+CTX', tokenizer.convert_ids_to_tokens(contextualised_inputs['input_ids'][0]))
            # print('>>>', tokenizer.convert_ids_to_tokens(contextualised_inputs['input_ids'][0, self.current_document['end_index']:]))
            # print()

            # data_in_doc = (contextualised_inputs.copy(), self.current_document['end_index'])
            self.data.append((
                contextualised_inputs.copy(),
                self.current_document['end_index'],
                idx
            ))
            self.current_document = contextualised_inputs.copy()

    def reset_current_document_context(self):
        if self.add_special_tokens and (self.tokenizer.bos_token == self.tokenizer.eos_token or ((not self.tokenizer.bos_token) and self.tokenizer.eos_token)):
            self.current_document = self.tokenizer(
                self.tokenizer.eos_token, add_special_tokens=False, return_tensors='pt')
            self.current_document['end_index'] = 0
        else:
            self.current_document = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ContextualisedDatasetControl(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, context_field, seed, size2sents_path, dataframe2_path, add_special_tokens=True):
        super(ContextualisedDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_seq_len = max_seq_len
        self.current_document = None
        self.dataframe2 = pd.read_csv(dataframe2_path)

        random.seed(seed)


        with open(size2sents_path, 'r') as f:
            self.size2sents = json.load(f)

        logger.warning('Tokenize...')

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):

            try:
                sentence = row['text'] + self.tokenizer.eos_token
            except TypeError:
                sentence = self.tokenizer.eos_token

            # print('SENT', sentence)

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                # truncation=True,
                add_special_tokens=False,
                # max_length=max_seq_len + 2
            )

            c_size = row['context_size_{}'.format(context_field)]

            # logger.warning('c_size {}'.format(c_size))

            control_context = self.tokenizer.eos_token
            if c_size != 0:

                # if c_size < 1024:
                try:
                    possible_control_contexts = self.size2sents[str(c_size)]
                except KeyError:
                    # if c_size > 1024 or cize not in size2sents
                    possible_control_contexts = self.size2sents["1024"]

                # logger.warning('possible_control_contexts {}'.format(possible_control_contexts))

                control_sent_ids = random.sample(possible_control_contexts, 1)[0]
                control_sent_ids = json.loads(control_sent_ids)

                # logger.warning(control_sent_ids)

                for sent_id in control_sent_ids:
                    # logger.warning(sent_id)
                    sent = self.dataframe2.loc[sent_id, 'text']
                    control_context = control_context + sent + self.tokenizer.eos_token

            control_context_inputs = self.tokenizer(
                control_context,
                return_tensors='pt',
                add_special_tokens=False,
            )

            contextualised_inputs = {
                    'input_ids': torch.cat((control_context_inputs['input_ids'], inputs['input_ids']), 1),
                    'attention_mask': torch.cat((control_context_inputs['attention_mask'], inputs['attention_mask']), 1),
                    'end_index': control_context_inputs['input_ids'].shape[1] - 1 # + inputs['input_ids'].shape[1]
            }

            # cut off context if longer than max_seq_len
            if contextualised_inputs['input_ids'].shape[1] > self.max_seq_len:
                contextualised_inputs = {
                    'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                    'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                    'end_index': self.max_seq_len - inputs['input_ids'].shape[1]
                }

            self.data.append((
                contextualised_inputs.copy(),
                contextualised_inputs['end_index'],
                idx
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ContextualisedDataset2(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, context_field, add_special_tokens=True):
        super(ContextualisedDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

        logger.warning('Tokenize...')

        current_context = ''

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            # if current_context == '':
            #     current_context = row[context_field]

            sentence = row['text']
            try:
                sentence = self.tokenizer.eos_token + sentence
            except TypeError:
                continue
            # print('SENT', sentence)

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                # truncation=True,
                add_special_tokens=False,
                # max_length=max_seq_len + 2
            )

            # print('TOK', tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

            if row[context_field] != current_context:
                is_new_context = True
                current_context = row[context_field]
            else:
                is_new_context = False

            self.data.append((
                inputs.copy(),
                is_new_context,
                idx
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class ContextualisedMaptaskDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, add_special_tokens=False):
        super(ContextualisedMaptaskDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_seq_len = max_seq_len
        self.current_dialogue = None
        self.current_transaction = None

        logger.warning('Tokenize...')

        self.reset_current_dialogue_context()
        self.reset_current_transaction_context()

        current_dialogue_id = ''
        current_transaction_num = ''

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            if current_dialogue_id == '':
                current_dialogue_id = row['dialogue_id']
            if current_transaction_num == '':
                current_transaction_num = row['transaction_number']

            sentence = row['text']

            if self.add_special_tokens and self.tokenizer.bos_token != self.tokenizer.eos_token:
                sentence = self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
            elif self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
                sentence = sentence + self.tokenizer.eos_token

            # print(sentence)

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )

            # print(inputs)

            if row['dialogue_id'] != current_dialogue_id:
                self.reset_current_dialogue_context()
                current_dialogue_id = row['dialogue_id']

            contextualised_inputs = {
                    'input_ids': torch.cat((self.current_dialogue['input_ids'], inputs['input_ids']), 1),
                    'attention_mask': torch.cat((self.current_dialogue['attention_mask'], inputs['attention_mask']), 1),
                    'end_index': self.current_dialogue['end_index'] + inputs['input_ids'].shape[1]
            }
            # cut off context if longer than max_seq_len
            contextualised_inputs = {
                'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                'end_index': contextualised_inputs['end_index']
            }
            # print('dialogue')
            # print(contextualised_inputs)
            data_in_dialogue = (contextualised_inputs.copy(), self.current_dialogue['end_index'])
            self.current_dialogue = contextualised_inputs.copy()

            if row['transaction_number'] != current_transaction_num:
                self.reset_current_transaction_context()
                current_transaction_num = row['transaction_number']

            contextualised_inputs = {
                'input_ids': torch.cat((self.current_transaction['input_ids'], inputs['input_ids']), 1),
                'attention_mask': torch.cat((self.current_transaction['attention_mask'], inputs['attention_mask']), 1),
                'end_index': self.current_transaction['end_index'] + inputs['input_ids'].shape[1]
            }
            # cut off context if longer than max_seq_len
            contextualised_inputs = {
                'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                'end_index': contextualised_inputs['end_index']
            }
            # print('transaction')
            # print(contextualised_inputs)

            self.data.append((
                contextualised_inputs.copy(),           # inputs in transaction
                data_in_dialogue[0],                    # inputs in dialogue
                self.current_transaction['end_index'],  # start_idx in transaction
                data_in_dialogue[1],                    # start_idx in dialogue
                idx                                     # dataframe index
            ))
            self.current_transaction = contextualised_inputs.copy()

    def reset_current_dialogue_context(self):
        if self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
            self.current_dialogue = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            self.current_dialogue['end_index'] = 0
        else:
            self.current_dialogue = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def reset_current_transaction_context(self):
        if self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
            self.current_transaction = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            self.current_transaction['end_index'] = 0
        else:
            self.current_transaction = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class ContextualisedPhotoBookDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, add_special_tokens=False):
        super(ContextualisedMaptaskDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_seq_len = max_seq_len
        self.current_dialogue = None
        self.current_round = None

        logger.warning('Tokenize...')

        self.reset_current_dialogue_context()
        self.reset_current_round_context()

        current_dialogue_id = ''
        current_round_num = ''

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            if current_dialogue_id == '':
                current_dialogue_id = row['dialogue_id']
            if current_round_num == '':
                current_round_num = row['round_number']

            sentence = row['text']

            if self.add_special_tokens and self.tokenizer.bos_token != self.tokenizer.eos_token:
                sentence = self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
            elif self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
                sentence = sentence + self.tokenizer.eos_token

            # print(sentence)

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )

            # print(inputs)

            if row['dialogue_id'] != current_dialogue_id:
                self.reset_current_dialogue_context()
                current_dialogue_id = row['dialogue_id']

            contextualised_inputs = {
                    'input_ids': torch.cat((self.current_dialogue['input_ids'], inputs['input_ids']), 1),
                    'attention_mask': torch.cat((self.current_dialogue['attention_mask'], inputs['attention_mask']), 1),
                    'end_index': self.current_dialogue['end_index'] + inputs['input_ids'].shape[1]
            }
            # cut off context if longer than max_seq_len
            contextualised_inputs = {
                'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                'end_index': contextualised_inputs['end_index']
            }
            # print('dialogue')
            # print(contextualised_inputs)
            data_in_dialogue = (contextualised_inputs.copy(), self.current_dialogue['end_index'])
            self.current_dialogue = contextualised_inputs.copy()

            if row['round_number'] != current_round_num:
                self.reset_current_round_context()
                current_round_num = row['round_number']

            contextualised_inputs = {
                'input_ids': torch.cat((self.current_round['input_ids'], inputs['input_ids']), 1),
                'attention_mask': torch.cat((self.current_round['attention_mask'], inputs['attention_mask']), 1),
                'end_index': self.current_round['end_index'] + inputs['input_ids'].shape[1]
            }
            # cut off context if longer than max_seq_len
            contextualised_inputs = {
                'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                'end_index': contextualised_inputs['end_index']
            }
            # print('transaction')
            # print(contextualised_inputs)

            self.data.append((
                contextualised_inputs.copy(),     # inputs in round
                data_in_dialogue[0],              # inputs in dialogue
                self.current_round['end_index'],  # start_idx in round
                data_in_dialogue[1],              # start_idx in dialogue
                idx                               # dataframe index
            ))
            self.current_round = contextualised_inputs.copy()

    def reset_current_dialogue_context(self):
        if self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
            self.current_dialogue = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            self.current_dialogue['end_index'] = 0
        else:
            self.current_dialogue = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def reset_current_round_context(self):
        if self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
            self.current_round = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            self.current_round['end_index'] = 0
        else:
            self.current_round = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class ContextualisedPhotoBookChainsDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, add_special_tokens=False):
        super(ContextualisedMaptaskDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.max_seq_len = max_seq_len
        self.current_chain = None
        self.current_transaction = None

        logger.warning('Tokenize...')

        self.reset_current_chain_context()

        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):

            sentence = row['text']

            if self.add_special_tokens and self.tokenizer.bos_token != self.tokenizer.eos_token:
                sentence = self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
            elif self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
                sentence = sentence + self.tokenizer.eos_token

            # print(sentence)

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )

            # print(inputs)

            if row['position_in_chain'] == 1:
                self.reset_current_chain_context()

            contextualised_inputs = {
                    'input_ids': torch.cat((self.current_chain['input_ids'], inputs['input_ids']), 1),
                    'attention_mask': torch.cat((self.current_chain['attention_mask'], inputs['attention_mask']), 1),
                    'end_index': self.current_chain['end_index'] + inputs['input_ids'].shape[1]
            }
            # cut off context if longer than max_seq_len
            contextualised_inputs = {
                'input_ids': contextualised_inputs['input_ids'][:, -self.max_seq_len:],
                'attention_mask': contextualised_inputs['attention_mask'][:, -self.max_seq_len:],
                'end_index': contextualised_inputs['end_index']
            }
            # print(contextualised_inputs)

            self.data.append((
                contextualised_inputs.copy(),     # inputs in chain
                self.current_chain['end_index'],  # start_idx in chain
                idx                               # dataframe index
            ))
            self.current_chain = contextualised_inputs.copy()

    def reset_current_chain_context(self):
        if self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
            self.current_chain = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            self.current_chain['end_index'] = 0
        else:
            self.current_chain = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def reset_current_transaction_context(self):
        if self.add_special_tokens and self.tokenizer.bos_token == self.tokenizer.eos_token:
            self.current_transaction = self.tokenizer(
                self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt')
            self.current_transaction['end_index'] = 0
        else:
            self.current_transaction = {
                'input_ids': torch.empty([1, 0], dtype=torch.long),
                'attention_mask': torch.empty([1, 0], dtype=torch.long),
                'end_index': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SentenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len):
        super(SentenceDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        logger.warning('Tokenize...')
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):

            try:
                sentence = self.tokenizer.eos_token + row['text']
            except TypeError:
                sentence = self.tokenizer.eos_token

            # if tokenizer.bos_token:
            #     sentence = tokenizer.bos_token + sentence
            # else:
            #     sentence = tokenizer.eos_token + sentence

            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len
            )
            self.data.append((inputs, idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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


class MaptaskContributionDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len, add_speaker_ids=False, add_turn_splits=False):
        super(MaptaskSentenceDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.add_speaker_ids = add_speaker_ids

        logger.warning('Tokenize...')
        current_game_number = 1
        df_indices = []
        contribution = tokenizer.bos_token
        for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):

            sentence = row['text']

            if self.add_speaker_ids:
                sentence = '{}: {}'.format(row['speaker'], sentence)

            if row['game_number'] != current_game_number:

                if not add_turn_splits:
                    contribution = contribution + ' ' + tokenizer.eos_token

                if add_turn_splits and tokenizer.bos_token != tokenizer.eos_token:
                    contribution = contribution[len(tokenizer.bos_token) + 1:]  # skip extra "<s> "

                inputs = self.tokenizer(
                    contribution,
                    return_tensors='pt',
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_seq_len + 2
                )
                self.data.append((inputs, df_indices))

                current_game_number = row['game_number']
                df_indices = [idx]

                if add_turn_splits:
                    contribution = tokenizer.bos_token + ' ' + sentence + ' ' + tokenizer.eos_token
                else:
                    contribution = tokenizer.bos_token + ' ' + sentence
            else:
                df_indices.append(idx)

                if add_turn_splits:
                    if tokenizer.bos_token != tokenizer.eos_token:
                        contribution = contribution + ' ' + tokenizer.bos_token + ' ' + sentence + ' ' + tokenizer.eos_token
                    else:
                        contribution = contribution + ' ' + sentence + ' ' + tokenizer.eos_token
                else:
                    contribution = contribution + ' ' + sentence

        # finally
        if not add_turn_splits:
            contribution = contribution + ' ' + tokenizer.eos_token

        if add_turn_splits and tokenizer.bos_token != tokenizer.eos_token:
            contribution = contribution[len(tokenizer.bos_token) + 1:]  # skip extra "<s> "

        inputs = self.tokenizer(
            contribution,
            return_tensors='pt',
            truncation=True,
            add_special_tokens=False,
            max_length=max_seq_len + 2
        )
        self.data.append((inputs, df_indices))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class PhotobookChainsSentenceDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer, max_seq_len):
        super(PhotobookChainsSentenceDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer

        logger.warning('Tokenize...')
        for sentence, idx_in_chain, idx_in_round, game_id in tqdm(dataset, total=len(dataset)):
            sentence = tokenizer.bos_token + sentence + tokenizer.eos_token
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False,
                max_length=max_seq_len + 2
            )
            self.data.append((inputs, idx_in_chain, idx_in_round, game_id))

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

