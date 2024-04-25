import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import os

script_directory = os.path.dirname(__file__)
file_path = os.path.join(script_directory, 'fnn_train.csv')
dataset = pd.read_csv(file_path, delimiter=',', engine='python', on_bad_lines=lambda bad_lines: [line for line in bad_lines if '"' in line and line.endswith('"')])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset['statement'])

#model.load_state_dict(torch.load('semantic_matching_model.pth'))
#model.eval()

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.text for p in paragraphs])
    return text

def find_matching_statement_from_dataset(query_sentence):
    query_vector = vectorizer.transform([query_sentence])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    max_similarity_index = similarity_scores.argmax()
    matching_statement = dataset.iloc[max_similarity_index]['paragraph_based_content']
    speaker = dataset.iloc[max_similarity_index]['speaker']
    label = dataset.iloc[max_similarity_index]['label_fnn']
    return matching_statement, speaker, label, similarity_scores[0, max_similarity_index]

# Function to find matching statement from website
def find_matching_statement_from_website(website_text, query_sentence):
    website_sentences = website_text.split('.')  # Split text into sentences
    max_similarity = 0
    matching_sentence = ""
    for sentence in website_sentences:
        if len(sentence.strip()) > 0:
            website_embedding = get_bert_embeddings(sentence)
            query_embedding = get_bert_embeddings(query_sentence)
            similarity_score = cosine_similarity(website_embedding, query_embedding)[0][0]
            if similarity_score > max_similarity:
                max_similarity = similarity_score
                matching_sentence = sentence
    return matching_sentence, max_similarity

# Function to encode text with BERT and get embeddings
def get_bert_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    embeddings = output.last_hidden_state[:, 0, :]
    return embeddings

if __name__ == '__main__':
    website_url = input("Enter website URL: ")
    sentence_to_match = input("Enter sentence to match: ")
    website_text = scrape_website(website_url)
    # Find matching statement from dataset
    matching_statement_from_dataset, speaker_from_dataset, label_from_dataset, similarity_score_from_dataset = find_matching_statement_from_dataset(sentence_to_match)
    # Find matching statement from website
    matching_sentence_from_website, similarity_score_from_website = find_matching_statement_from_website(website_text, sentence_to_match)
    # print("Matching Statement from Dataset:", matching_statement_from_dataset)
    print("Speaker from Dataset:", speaker_from_dataset)
    print("Label from Dataset (Real/Fake):", label_from_dataset)
    print("Similarity Score from Dataset:", similarity_score_from_dataset)

    print("Matching Statement from Website:", matching_sentence_from_website)
    print("Similarity Score from Website:", similarity_score_from_website)



