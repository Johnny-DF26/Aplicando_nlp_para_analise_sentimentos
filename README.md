# 🚀 Análise de Sentimentos em Avaliações de Produtos com NLP

## 📝 Resumo do Projeto

Este projeto demonstra a aplicação de técnicas de Processamento de Linguagem Natural (NLP) para desenvolver um sistema de **Análise de Sentimentos** em avaliações de clientes. O objetivo é fornecer uma solução escalável e precisa para empresas monitorarem a satisfação de seus clientes, identificando rapidamente avaliações positivas e negativas. Para isso, são exploradas e comparadas diversas metodologias de NLP, desde abordagens clássicas até modelos de última geração baseados em Transformers.

## 🎯 Objetivo

Desenvolver um sistema robusto de análise de sentimentos que possa processar e classificar automaticamente avaliações de produtos como `positivas` ou `negativas`, permitindo às empresas tomar decisões mais rápidas e informadas sobre a experiência do cliente.

## 🛠️ Metodologia

O projeto segue um pipeline completo de Ciência de Dados e NLP:

### 1. Limpeza e Normalização de Dados

*   **Caixa Baixa:** Conversão de todo o texto para letras minúsculas para padronização.
*   **Remoção de Acentuação:** Utilização da biblioteca `unidecode` para remover acentos.
*   **Remoção de Pontuação:** Tokenização e filtragem de caracteres não alfabéticos.
*   **Remoção de Stop Words:** Exclusão de palavras irrelevantes (artigos, preposições, etc.) usando `nltk` para o português.
*   **Stemming (Simplificação de Palavras):** Redução das palavras às suas raízes morfológicas (e.g., "correndo" -> "corr"), usando `nltk.RSLPStemmer`.

### 2. Vetorização de Texto

A etapa de vetorização é crucial para transformar o texto em representações numéricas que podem ser utilizadas por modelos de Machine Learning. Foram exploradas e comparadas as seguintes técnicas:

*   **Bag of Words (BoW):** Representação simples que conta a frequência das palavras, ignorando a ordem e o contexto. Implementado com `CountVectorizer`.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** Uma evolução do BoW que pondera a frequência de uma palavra pela sua raridade no corpus, destacando termos mais relevantes. Implementado com `TfidfVectorizer`.
*   **Word Embeddings (Word2Vec / GloVe):** Mapeia palavras para vetores densos em um espaço semântico, onde palavras com significados semelhantes têm vetores próximos. Utilizado `spaCy` para extração de vetores médios das frases.
*   **Transformers (BERTimbau):** Modelo de linguagem avançado que gera embeddings contextuais dinâmicos, capturando nuances e dependências complexas no texto através do mecanismo de `Self-Attention`. Utilizado `transformers` para carregar o modelo BERTimbau (português).

### 3. Modelagem e Avaliação

Após a vetorização, modelos de `Regressão Logística` foram treinados e avaliados usando as seguintes métricas e técnicas:

*   **Acurácia:** Medida geral de desempenho do classificador.
*   **`classification_report`:** Relatório detalhado com precisão, recall e F1-score para cada classe.
*   **Matriz de Confusão:** Visualização do desempenho do modelo em relação a acertos e erros por classe.
*   **Validação Cruzada (K-Fold):** Para garantir a robustez e generalização do modelo, evitando *overfitting*.

## 📈 Resultados e Conclusão

| Técnica             | Modelo              | Acurácia (Aprox.) | Observação Principal                                                                      |
| :------------------ | :------------------ | :----------------: | :---------------------------------------------------------------------------------------- |
| **Bag of Words**    | Logistic Regression | ~88%               | Simples, rápido, mas ignora o contexto.                                                   |
| **TF-IDF**          | Logistic Regression | **~93%**           | Melhor equilíbrio entre custo computacional e performance, capturando relevância de termos. |
| **Word Embeddings** | Logistic Regression | ~88%               | Captura similaridade semântica, mas a média pode perder nuances.                           |
| **BERTimbau**       | Logistic Regression | ~88%               | Estado da arte, entende contexto profundo, mas com alto custo computacional.               |

**Conclusão:** O **TF-IDF** demonstrou ser uma excelente alternativa, com um desempenho muito competitivo e menor custo computacional, sendo uma escolha pragmática para a maioria dos cenários com avaliações curtas.

## 🚀 Como Usar o Modelo (TF-IDF)

O modelo treinado com TF-IDF foi salvo e pode ser carregado para prever o sentimento de novas avaliações.

### 1. Carregar o Modelo e o Vetorizador

```python
import joblib

modelo_ml = joblib.load('/content/drive/MyDrive/models-pkl/model_logistic_regressor_nlp_sentimentos.pkl')
tfidf_vectorizer = joblib.load('/content/drive/MyDrive/models-pkl/tfidf_vectorizer_nlp_sentimentos.pkl')
```

### 2. Funções de Limpeza e Previsão

Certifique-se de ter as funções de limpeza e previsão definidas no notebook:

```python
import nltk
from nltk import tokenize
import unidecode
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

palavras_irrelevantes = stopwords.words('portuguese')
token = tokenize.WordPunctTokenizer()
stemmer = RSLPStemmer()

def limpeza_dados(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = token.tokenize(texto)
    texto = [palavra for palavra in texto if palavra.isalpha()]
    texto = [palavra for palavra in texto if palavra not in palavras_irrelevantes]
    texto = [stemmer.stem(palavra) for palavra in texto]
    texto = ' '.join(texto)
    return texto

def previsao_sentimento(text):
    text_vectorized = tfidf_vectorizer.transform([text])
    result = modelo_ml.predict(text_vectorized)
    return result

def formatacao_texto(texto_original):
    print("Fazendo previsão do sentimento do cliente...")
    print('---'*20)
    print(f"Texto Original: {texto_original}")

    texto_limpo = limpeza_dados(texto_original)
    print(f"Texto tratado: {texto_limpo}")

    previsao = previsao_sentimento(texto_limpo)

    print('==='*20)
    print("Previsão:")
    if previsao == 1:
        print('O Sentimento do cliente: POSITIVO!')
    else:
        print('O Sentimento do cliente: NEGATIVO !')
```

### 3. Exemplo de Uso

```python
texto_novo = 'O produto chegou antes do prazo e funciona perfeitamente!'
formatacao_texto(texto_novo)

texto_ruim = 'Produto de péssima qualidade. Não funciona como esperado.'
formatacao_texto(texto_ruim)
```

## 👥 Autor

* Nome: Johnny
* Curso: Alura
* GitHub: https://github.com/Johnny-DF26
* LinkedIn: https://www.linkedin.com/in/datasciencejohnny/
