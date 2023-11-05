from src.data.config import PREFIX
from src.models.train_model import create_model


def detoxify(model, inference_request, tokenizer, print_res=True):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)
    if print_res:
        print(f'Initial message: {inference_request.strip(PREFIX)}')
        print(f'Detoxified message: {decoded}')
    return decoded


def predict(sentence, tokenizer, model=None, print_res=True):
    if model is None:
        try:
            model = create_model(from_best=True)
        except:
            print('No model found. Please, either save best model or provide it as an argument')
            return
    model.eval()
    model.config.use_cache = False

    inference_request = PREFIX + sentence
    return detoxify(model, inference_request, tokenizer, print_res)
