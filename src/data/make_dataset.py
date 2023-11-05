import warnings

from datasets import load_metric, DatasetDict, Dataset
import transformers
import datasets
import random
import pandas as pd
import zipfile
import io
import numpy as np

warnings.filterwarnings('ignore')


FILEPATH = 'data/raw/filtered_paranmt.zip'


def unzip_tsv(filepath=FILEPATH):
    """
    GitHub does not allow to store raw tsv file due to its large size. Thus, zip is stored and it is unzipped here
    :param filepath: where zip file is stored
    :return: pandas DataFrame
    """
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        file_content = zip_ref.read("filtered.tsv").decode("utf-8")
    df = pd.read_csv(io.StringIO(file_content), sep="\t")
    return df


def bring_toxic_to_one_col(df):
    """
    Make shuffled dataset more structured by division of toxic and neutral sentences.
    For more details chech Final solution report
    :param df: pandas DataFrame of the dataset
    :return: pandas DataFrame, not shuffled
    """
    df['toxic'] = df.apply(lambda row: row['reference'] if row['ref_tox'] > row['trn_tox'] else row['translation'], axis=1)
    df['toxic_tox'] = df[['ref_tox', 'trn_tox']].max(axis=1)
    df['neutral'] = df.apply(lambda row: row['reference'] if row['ref_tox'] <= row['trn_tox'] else row['translation'], axis=1)
    df['neutral_tox'] = df[['ref_tox', 'trn_tox']].min(axis=1)

    # Drop the old columns
    df = df.drop(['reference', 'translation', 'ref_tox', 'trn_tox'], axis=1)
    return df


def formalize_dataset(df):
    """
    Make datasets library-formal dataset dict
    :param df: pandas DataFrame of the dataset
    :return:
    """
    dataset = Dataset.from_pandas(df)

    # Split the dataset into train, validation, and test sets
    train_val_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()
    print(train_val_dataset, test_dataset)
    train_dataset, val_dataset = train_val_dataset.train_test_split(test_size=0.2).values()

    dataset_dict = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
    return dataset_dict


def get_dataset_dict():
    """
    Summary dataset dict creation function
    :return:
    """
    return formalize_dataset(bring_toxic_to_one_col(unzip_tsv()))
