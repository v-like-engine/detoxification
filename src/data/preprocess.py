from transformers import AutoTokenizer

from models.model_checkpoint_config import PROPHET
from src.data import REFERENCE, TRANSLATION, PREFIX, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
from src.data.make_dataset import get_dataset_dict


def set_up_tokenizer(model_checkpoint=PROPHET):
    """
    Function to get the tokenizer instance
    :param model_checkpoint: str value of the model used
    :return: AutoTokenizer instance
    """
    return AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize(examples, model_checkpoint=PROPHET):
    """
    Function to tokenize the dataset. Uses tokenizer setter from above.
    :param examples: dataset to tokenize
    :param model_checkpoint: str value of the model used
    :return:
    """
    tokenizer = set_up_tokenizer(model_checkpoint)
    inputs = [PREFIX + ex for ex in examples[REFERENCE]]
    targets = [ex for ex in examples[TRANSLATION]]

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def crop_dataset(raw_datasets, train_crop_size=5000, val_crop_size=1000, test_crop_size=1000):
    """
    For the sake of time economy in case of fine-tuning the pre-trained model,
    it is recommended to make dataset cropped
    :param raw_datasets: datasets library formatted dataset
    :param train_crop_size: number of instances in the train part of dataset
    :param val_crop_size:
    :param test_crop_size:
    :return:
    """
    cropped_datasets = raw_datasets
    cropped_datasets['train'] = raw_datasets['train'].select(range(train_crop_size))
    cropped_datasets['validation'] = raw_datasets['validation'].select(range(val_crop_size))
    cropped_datasets['test'] = raw_datasets['test'].select(range(test_crop_size))
    tokenized_datasets = cropped_datasets.map(tokenize, batched=True)
    return tokenized_datasets


def get_tokenized_dataset():
    return crop_dataset(get_dataset_dict())
