from transformers import AutoTokenizer, AutoModel
import torch

# Test sentences
test_sentences = [
    "I have two cats, a black one named Tom and a white one named Jerry.",
    "Tom likes to chase Jerry around the house.",
    "The white house is a big house.",
    "NYU is a university in New York City.",
    "Jerry is a professor at NYU.",
    "Tom is twelve years old.",
    "Jerry is a cat."
]

class SentenceTransformer:
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        
    def encode(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output.last_hidden_state.mean(dim=1)

    def cosine_similarity(self, sent1, sent2):
        emb1, emb2 = self.encode([sent1, sent2])
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()

class TestSentenceTransformers:
    def __init__(self):
        self.models = {
            'DistilBERT': SentenceTransformer('distilbert-base-uncased'),
            'BERT': SentenceTransformer('bert-base-uncased'),
            'RoBERTa': SentenceTransformer('roberta-base'),
            'XLM-RoBERTa': SentenceTransformer('xlm-roberta-base'),
            'Jina v3': SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
        }

    def test_encoding(self):
        for name, model in self.models.items():
            print(f"Testing encoding for {name}:")
            embeddings = model.encode(test_sentences)
            print(f"Shape of embeddings: {embeddings.shape}")
            print(f"Sample embedding (first 5 dimensions): {embeddings[0][:5]}\n")

    def test_similarity(self):
        for name, model in self.models.items():
            print(f"Testing similarity for {name}:")
            sim = model.cosine_similarity(test_sentences[0], test_sentences[1])
            print(f"Similarity between '{test_sentences[0]}' and '{test_sentences[1]}': {sim:.4f}\n")

# Run tests
tester = TestSentenceTransformers()
print("Testing encoding:")
tester.test_encoding()
print("\nTesting similarity:")
tester.test_similarity()