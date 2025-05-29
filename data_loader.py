from datasets import load_dataset

def load_and_preprocess_dataset(tokenizer):
    dataset = load_dataset("dltdojo/ecommerce-faq-chatbot-dataset")
    train_dataset = dataset['train']

    def preprocess_data(examples):
        inputs = ["question: " + q for q in examples['question']]
        targets = examples['answer']
        return {'input_text': inputs, 'target_text': targets}

    train_dataset = train_dataset.map(preprocess_data, batched=True)

    def tokenize_data(examples):
        inputs = tokenizer(examples['input_text'], max_length=128, truncation=True, padding="max_length")
        targets = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': targets['input_ids']
        }

    train_dataset = train_dataset.map(tokenize_data, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_dataset
