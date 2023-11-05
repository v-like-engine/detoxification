import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from make_dataset import unzip_tsv, bring_toxic_to_one_col, FILEPATH


# code for torch-based training experiments. Include Dataset class, creation and dataloader loading


def extract_tsv(filepath):
    # load the .tsv file and return the data as a pandas DataFrame
    data = pd.read_csv(filepath, sep='\t')
    return data


class ParaNMTDataset(Dataset):
    def __init__(self, tokenizer, dataframe=None, filepath=None):
        if filepath is not None:
            self.data = bring_toxic_to_one_col(extract_tsv(filepath))
        elif dataframe is not None:
            self.data = bring_toxic_to_one_col(dataframe)
        else:
            print('No file or dataframe were provided to Dataset constructor')
            self.data = None
        self.tokenizer = tokenizer

        # example tokenizer: BertTokenizer.from_pretrained('bert-base-uncased') for Bert model

    def __len__(self):
        # len is obligatory method for the torch-formatted dataset
        return len(self.data)

    def __getitem__(self, index):
        source_text = self.data.iloc[index]['toxic']
        target_text = self.data.iloc[index]['neutral']
        ref_tox = self.data.iloc[index]['toxic_tox']
        trn_tox = self.data.iloc[index]['neutral_tox']

        source_tokens = self.tokenizer.tokenize(source_text)
        target_tokens = self.tokenizer.tokenize(target_text)
        source_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(source_tokens))
        target_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(target_tokens))

        # Add the missing inputs
        input_ids = source_ids
        attention_mask = torch.ones_like(input_ids)
        decoder_input_ids = target_ids
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, ref_tox, trn_tox


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    decoder_input_ids = [item[2] for item in batch]
    decoder_attention_mask = [item[3] for item in batch]
    toxic_tox = [item[4] for item in batch]
    neutral_tox = [item[5] for item in batch]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True)
    decoder_input_ids_padded = pad_sequence(decoder_input_ids, batch_first=True)
    decoder_attention_mask_padded = pad_sequence(decoder_attention_mask, batch_first=True)
    toxic_tox = torch.tensor(toxic_tox)
    neutral_tox = torch.tensor(neutral_tox)

    return input_ids_padded, attention_mask_padded, decoder_input_ids_padded, decoder_attention_mask_padded, \
        toxic_tox, neutral_tox


def get_dataloader(dataframe=None, train_test_split=True, test_size=0.2, filepath=None) -> (DataLoader, DataLoader):
    if dataframe is not None:
        dataset_ = ParaNMTDataset(dataframe)
    elif filepath is not None:
        dataset_ = ParaNMTDataset(filepath)
    else:
        print('No data was provided to get_dataloader')
        return None

    if train_test_split:
        # split dataset into train and test sets
        dataset_size = len(dataset_)
        train_size = int((1 - test_size) * dataset_size)
        test_size = dataset_size - train_size
        train_data, test_data = random_split(dataset_, [train_size, test_size])

        # create DataLoaders for train and test sets
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

        return train_dataloader, test_dataloader
    else:
        # create a DataLoader for the entire dataset
        dataloader_ = DataLoader(dataset_, batch_size=32, shuffle=True, collate_fn=collate_fn)
        return dataloader_


def get_dataloader_from_zip(filepath=FILEPATH):
    return get_dataloader(unzip_tsv(filepath))

