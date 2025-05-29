from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./chatbot_model",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
