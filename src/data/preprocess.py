from transformers import AutoTokenizer

from models.model_checkpoint_config import PROPHET
from src.data import REFERENCE, TRANSLATION, PREFIX, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
from src.data.make_dataset import get_dataset_dict


def set_up_tokenizer(model_checkpoint=PROPHET):
    return AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize(examples, model_checkpoint=PROPHET):
    tokenizer = set_up_tokenizer(model_checkpoint)
    inputs = [PREFIX + ex for ex in examples[REFERENCE]]
    targets = [ex for ex in examples[TRANSLATION]]

    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def crop_dataset(raw_datasets, train_crop_size=10000, val_crop_size=2000, test_crop_size=2000):
    cropped_datasets = raw_datasets
    cropped_datasets['train'] = raw_datasets['train'].select(range(train_crop_size))
    cropped_datasets['validation'] = raw_datasets['validation'].select(range(val_crop_size))
    cropped_datasets['test'] = raw_datasets['test'].select(range(test_crop_size))
    tokenized_datasets = cropped_datasets.map(tokenize, batched=True)
    return tokenized_datasets


def get_tokenized_dataset():
    return crop_dataset(get_dataset_dict())
