from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

tokenizer = T5Tokenizer.from_pretrained("./chatbot_model")
model = T5ForConditionalGeneration.from_pretrained("./chatbot_model")

def answer_question(question):
    input_text = "question: " + question
    inputs = tokenizer(input_text, return_tensors="pt", max_length=200, truncation=True, padding=True)
    outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=1000,
    num_beams=6,
    early_stopping=True,
    no_repeat_ngram_size=2,   
    num_return_sequences=1,
)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load dataset gốc để test
dataset = load_dataset("dltdojo/ecommerce-faq-chatbot-dataset")

for i in range(5):
    q = dataset['train'][i]['question']
    a = dataset['train'][i]['answer']
    pred = answer_question(q)
    print(f"\nQ: {q}\nExpected: {a}\nPredicted: {pred}")
