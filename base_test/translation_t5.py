from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("t5-small")

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

tokenized_books = books.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)