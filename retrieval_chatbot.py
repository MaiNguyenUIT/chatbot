# retrieval_chatbot.py
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch
import json

class FAQChatbot:
    def __init__(self, dataset_path="data.json", model_name="all-MiniLM-L6-v2"):
        # Load dataset từ file JSON
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.faq_questions = [item['question'] for item in data['questions']]
        self.faq_answers = [item['answer'] for item in data['questions']]

        # Load embedding model
        self.model_emb = SentenceTransformer(model_name)

        # Encode FAQ questions, lưu trên GPU nếu có
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_emb = self.model_emb.to(device)
        self.faq_emb = self.model_emb.encode(
            self.faq_questions,
            convert_to_tensor=True,
            device=device
        )

    def answer_question(self, query):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        query_emb = self.model_emb.encode(query, convert_to_tensor=True, device=device)
        hits = util.semantic_search(query_emb, self.faq_emb, top_k=1)
        best_idx = hits[0][0]['corpus_id']
        return self.faq_answers[best_idx]

if __name__ == "__main__":
    chatbot = FAQChatbot()
    while True:
        q = input("Your question: ")
        if q.lower() in ["exit", "quit"]:
            break
        a = chatbot.answer_question(q)
        print(f"Answer: {a}")
