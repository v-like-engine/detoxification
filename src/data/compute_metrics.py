import numpy as np
from datasets import load_metric

from models import CURRENT
from src.data import set_up_tokenizer


def postprocess_text(preds, labels):
    """
    Auxiliary function for postprocessing of preds and labels (formatting them as lists)
    :param preds:
    :param labels:
    :return:
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# compute metrics function from the Lab5
def compute_metrics(eval_preds):
    """
    This function is to compute sacrebleu metric.
    Code is reused from lab5.
    Goes as parameter to the Seq2SeqTrainer
    :param eval_preds:
    :return:
    """
    tokenizer = set_up_tokenizer(CURRENT)
    metric = load_metric("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
