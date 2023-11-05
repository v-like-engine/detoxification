from models import *
from src.models import train, predict
from src.data import get_tokenized_dataset, set_up_tokenizer

if __name__ == '__main__':
    model_name = T5_S
    print(f"Initializing dataset...")
    tokenized_dataset = get_tokenized_dataset()
    print(f"Initializing model {model_name} and training loop...")
    train(tokenized_dataset, model_name)
    prediction = predict(sentence="Damn that test makes me swear hardly, man. Bruv I am losin' patience",
                         tokenizer=set_up_tokenizer(model_name))
