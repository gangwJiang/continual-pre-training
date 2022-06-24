from transformers import (
    BartForConditionalGeneration, BartTokenizerFast
  )
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer():
    return BartTokenizerFast.from_pretrained("facebook/bart-large", max_length=64)

TOKENIZER = load_tokenizer()

def load_model(model_path:str) -> BartForConditionalGeneration:
    bart = BartForConditionalGeneration.from_pretrained(model_path)

    bart.to(DEVICE)

    return bart

def generate(model, input_concepts, min_length, max_length, iterations):
    dct = TOKENIZER([input_concepts], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    hyp = model.generate(dct["input_ids"], max_length=max_length, min_length=min_length, num_beams=iterations, num_return_sequences=iterations)
    decoded_sents = TOKENIZER.batch_decode(hyp, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for decoded_sent in decoded_sents:
        yield decoded_sent

concept = "girl sit red clothes"
bart = load_model("facebook/bart-base")
out = generate(bart, concept, min_length=2, max_length=40, iterations=3)
print(str(out))