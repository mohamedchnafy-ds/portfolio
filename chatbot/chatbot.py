import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
# Corpus de questions-réponses
corpus = [
    {
        "question": "Quelles sont les conditions d'admission en Master 2 Mathématiques Appliquées ?",
        "answer": "Les candidats doivent être titulaires d'un Master 1 en Mathématiques ou équivalent. Un solide bagage en analyse, probabilités et statistiques est requis. L'admission se fait sur dossier et éventuellement entretien.",
        "context": "Conditions d'admission pour le Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Quels sont les principaux domaines d'étude dans ce Master 2 ?",
        "answer": "Le programme couvre principalement l'analyse numérique, les équations aux dérivées partielles, les probabilités avancées, les statistiques, l'optimisation et la modélisation mathématique.",
        "context": "Programme d'études du Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Y a-t-il des stages obligatoires dans le cursus ?",
        "answer": "Oui, un stage de fin d'études de 4 à 6 mois est obligatoire. Il peut être effectué en entreprise ou dans un laboratoire de recherche.",
        "context": "Stages dans le Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Quelles sont les débouchés professionnels après ce Master ?",
        "answer": "Les diplômés peuvent travailler comme ingénieurs R&D, data scientists, quants en finance, ou poursuivre en doctorat pour une carrière dans la recherche ou l'enseignement supérieur.",
        "context": "Débouchés professionnels du Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Le Master propose-t-il des parcours spécialisés ?",
        "answer": "Oui, le Master offre trois parcours : 'Modélisation et Calcul Scientifique', 'Statistiques et Science des Données', et 'Mathématiques Financières'.",
        "context": "Parcours spécialisés du Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Quels logiciels ou langages de programmation sont enseignés ?",
        "answer": "Les étudiants apprennent à utiliser Python, R, MATLAB, et des bibliothèques spécialisées comme NumPy, SciPy, et TensorFlow.",
        "context": "Logiciels et langages de programmation enseignés"
    },
    {
        "question": "Y a-t-il des possibilités d'échanges internationaux ?",
        "answer": "Oui, le Master a des partenariats avec plusieurs universités européennes et nord-américaines pour des semestres d'échange au S3 ou des doubles diplômes.",
        "context": "Échanges internationaux dans le Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Quel est le volume horaire des cours ?",
        "answer": "Le volume horaire total est d'environ 400 heures sur l'année, réparties entre cours magistraux, TD, TP et projets.",
        "context": "Volume horaire du Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Comment se déroule l'évaluation dans ce Master ?",
        "answer": "L'évaluation combine contrôle continu, examens écrits, projets en groupe, et soutenance de stage. Chaque UE (Unité d'Enseignement) a ses propres modalités d'évaluation.",
        "context": "Modalités d'évaluation du Master 2 Mathématiques Appliquées"
    },
    {
        "question": "Existe-t-il des aides financières pour les étudiants de ce Master ?",
        "answer": "Oui, des bourses sur critères sociaux sont disponibles, ainsi que des bourses d'excellence pour les meilleurs dossiers. Certaines entreprises partenaires proposent également des contrats de professionnalisation.",
        "context": "Aides financières pour les étudiants du Master 2 Mathématiques Appliquées"
    }
]

# Charger le modèle BERT
@st.cache_resource
def load_model():
    bert_model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
    bert_layer = hub.KerasLayer(bert_model_url, trainable=True)
    
    input_word_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_type_ids")
    
    pooled_output, _ = bert_layer([input_word_ids, input_mask, input_type_ids])
    
    output = tf.keras.layers.Dense(1, activation="sigmoid")(pooled_output)
    
    model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids], outputs=output
    )
    
    return model

model = load_model()

# Fonction pour tokenizer le texte
def bert_encode(texts, tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# Charger le tokenizer BERT
vocab_file = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=False
).resolved_object.vocab_file.asset_path.numpy()
tokenizer = bert.tokenization.FullTokenizer(vocab_file)

# Fonction pour obtenir une réponse
def get_answer(question, context):
    input_text = f"question: {question} context: {context}"
    tokens, masks, segments = bert_encode([input_text], tokenizer, max_len=128)
    predictions = model.predict([tokens, masks, segments])
    return context[int(predictions[0][0] * len(context)):]
# Application Streamlit
st.title("Chatbot Master 2 Mathématiques Appliquées")

question = st.text_input("Posez votre question sur le Master 2 Mathématiques Appliquées :")

if question:
    for item in corpus:
        if question.lower() in item["question"].lower():
            context = item["context"] + " " + item["answer"]
            answer = get_answer(question, context)
            st.write(f"Réponse : {answer}")
            break
    else:
        st.write("Désolé, je n'ai pas trouvé de réponse à cette question spécifique. Voici les questions disponibles :")
        for item in corpus:
            st.write(f"- {item['question']}")