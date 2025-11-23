import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# 1. Base simples de treinamento
mensagens = [
    "promoção imperdível compre agora",
    "você ganhou um prêmio clique aqui",
    "oferta exclusiva desconto limitado",
    "vamos almoçar hoje?",
    "pode me ligar quando puder",
    "estou chegando em casa"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = spam, 0 = normal


# 2. Criar TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(mensagens)

# 3. Treinar modelo
modelo = MultinomialNB()
modelo.fit(X, labels)

# 4. Salvar modelo + vectorizer
os.makedirs("model/model", exist_ok=True)

with open("model/modelo_spam.pkl", "wb") as f:
    pickle.dump((modelo, vectorizer), f)

print("✔️ Modelo salvo em model/modelo_spam.pkl")
