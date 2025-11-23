import pickle

# Carregar modelo salvo
with open("model/modelo_spam.pkl", "rb") as f:
    modelo, vectorizer = pickle.load(f)

print("Modelo carregado! Vamos testar...")

while True:
    texto = input("\nDigite uma mensagem (ou 'sair'): ")

    if texto.lower() == "sair":
        break

    vetor = vectorizer.transform([texto])
    resultado = modelo.predict(vetor)[0]

    if resultado == 1:
        print("⚠️ Isso parece SPAM!")
    else:
        print("✔️ Mensagem normal.")
