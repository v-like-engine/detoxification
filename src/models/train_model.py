from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from models.model_checkpoint_config import PROPHET
from src.data.config import TRAIN, VAL
from src.data.make_dataset import *
from src.data.compute_metrics import compute_metrics
from src.data.preprocess import set_up_tokenizer


def create_model(model_checkpoint=PROPHET, from_best=False):
    """
    Create model. If from_best=True, creates model from saved parameters. Default directory is 'models/best'
    :param model_checkpoint: str value of the model used
    :param from_best: if the best model was already saved, one can load it from the '/models/best' directory
    :return:
    """
    if from_best:
        model = AutoModelForSeq2SeqLM.from_pretrained(f'models/best')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    return model


def set_up_training(model_checkpoint=PROPHET, batch_size=32, learning_rate=2e-5, num_epochs=10):
    """
    Set up all the preferences and parameters for training
    :param model_checkpoint: str value of the model used
    :param batch_size:
    :param learning_rate:
    :param num_epochs:
    :return:
    """
    transformers.set_seed(42)
    model = create_model(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"data/interim/{model_name}-finetuned-detox",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=True,
        report_to='tensorboard',
    )
    return model, model_name, args


def train(tokenized_dataset, model_checkpoint=PROPHET, batch_size=4, learning_rate=2e-5, num_epochs=10, save=True):
    """
    Set up trainer and start training loop
    :param tokenized_dataset: dataset dict tokenized
    :param model_checkpoint: str value of the model used
    :param batch_size:
    :param learning_rate:
    :param num_epochs:
    :param save: whether model should be saved as best after training or not
    :return:
    """
    model, model_name, args = set_up_training(model_checkpoint, batch_size, learning_rate, num_epochs)
    tokenizer = set_up_tokenizer(model_checkpoint)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_dataset[TRAIN],
        eval_dataset=tokenized_dataset[VAL],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    if save:
        trainer.save_model(f'models/best')
