from model import get_tokenizer_and_model, get_trainer

tokenizer, model = get_tokenizer_and_model()
trainer = get_trainer(model, tokenizer)

trainer.train()

# Lưu model
model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")
