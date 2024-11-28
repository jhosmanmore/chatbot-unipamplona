from sentence_transformers import SentenceTransformer, util
import numpy as np
from ChatBot import ChatBot

# Se inicializa el modelo de embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Se inicializa el bot
bot = ChatBot()

# Preguntas de prueba con respuestas de referencia
test_questions = [
    {"question": "¿Cómo solicito una transferencia interna?", 
     "expected": "Para el trámite de transferencia interna se debe cumplir con los siguientes requerimientos: a. Presentar solicitud a la Oficina de Admisiones, Registro y Control Académico, dentro de las fechas estipuladas en el calendario académico. b. Que la Universidad tenga disponibilidad y oferta de cupos en el programa y nivel solicitados. c. Que el aspirante que solicita la transferencia cumpla con los requisitos de ingreso. d. Tramitar estudio de homologación. e. En los casos de cambio de sede, no se co nsidera como transferencia interna y aplican las condiciones establecidas en el programa del que se transfiere. (art. 27)."},
    {"question": "Dame un ejemplo de un programa 'Hola Mundo' en Python", 
     "expected": "La pregunta está fuera del contexto académico disponible. Por favor, contacta directamente con un asesor de la Universidad de Pamplona para más información."},
]

# Evaluación de similitud y métricas
def evaluar_similitud(test_questions, bot, model, similarity_threshold=0.8):
    similarity_scores = []
    results = []

    for item in test_questions:
        question = item["question"]
        expected_answer = item["expected"]
        
        # Obtener respuesta del sistema
        generated_answer = bot.ask(question)
        
        # Generar embeddings de la respuesta esperada y la generada
        expected_embedding = model.encode(expected_answer, convert_to_tensor=True)
        generated_embedding = model.encode(generated_answer, convert_to_tensor=True)
        
        # Calcular la similitud coseno entre los embeddings
        similarity_score = util.pytorch_cos_sim(expected_embedding, generated_embedding).item()
        similarity_scores.append(similarity_score)
        
        # Verifica si la similitud supera el umbral establecido
        result = "Correcta" if similarity_score >= similarity_threshold else "Incorrecta"
        results.append({
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "similarity_score": similarity_score,
            "result": result
        })

    # Mostrar resultados
    for res in results:
        print(f"Pregunta: {res['question']}")
        print(f"Respuesta Generada: {res['generated_answer']}")
        print(f"Respuesta Esperada: {res['expected_answer']}")
        print(f"Similitud: {res['similarity_score']:.2f} - {res['result']}")
        print("-" * 50)

    # Calcular y mostrar el puntaje promedio de similitud
    average_similarity = np.mean(similarity_scores)
    print(f"\nPuntaje promedio de similitud: {average_similarity:.2f}")
    return average_similarity

if __name__ == "__main__":
    average_score = evaluar_similitud(test_questions, bot, model)


