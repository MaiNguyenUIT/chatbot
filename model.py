from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer
from config import training_args
from data_loader import load_and_preprocess_dataset

def get_tokenizer_and_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

def get_trainer(model, tokenizer):
    train_dataset = load_and_preprocess_dataset(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    return trainer
