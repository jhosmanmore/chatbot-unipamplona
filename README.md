![image](https://github.com/user-attachments/assets/41acd360-5a82-4dad-8393-b83e83c7fe9b)


# 🎓 RuleUP@Bot - Chatbot Académico Basado en RAG
RuleUP@Bot es un sistema conversacional inteligente desarrollado como trabajo de grado de Ingeniería de Sistemas. Está diseñado para asistir a estudiantes y usuarios en la consulta de procesos académicos basados en el Reglamento Académico Estudiantil de la Universidad de Pamplona.

Este chatbot utiliza una arquitectura de Generación Aumentada por Recuperación (RAG) combinando modelos de lenguaje (LLMs), vectores semánticos y búsqueda en bases de datos vectoriales para ofrecer respuestas precisas, rápidas y contextualizadas.

# 🧠 Tecnologías utilizadas
- Python

- LangChain – para la orquestación de cadenas RAG

- Hugging Face Inference API – (Modelo: meta-llama/Llama-3.2-3B-Instruct)

- Pinecone – para la base de datos vectorial

- SentenceTransformers – para embeddings (all-MiniLM-L6-v2)

- Flask – como framework web para la interfaz del chatbot

- HTML, CSS, JS – para la interfaz de usuario

- Render / Railway – para despliegue en la nube

# ⚙️ ¿Cómo funciona?
- Carga de documentos: Se procesan archivos PDF o .txt del reglamento académico.

- Segmentación y Embedding: El texto es dividido en fragmentos y transformado en vectores usando sentence-transformers.

- Indexación en Pinecone: Se almacenan los vectores en una base de datos para su búsqueda eficiente.

- Consulta del usuario: El usuario realiza una pregunta en lenguaje natural.

### RAG Chain:

- Se recuperan los fragmentos más relevantes (Retriever).

- Se construye un prompt con contexto y pregunta.

- El modelo LLM genera la respuesta usando el contexto.

# 🖥️ Interfaz de Usuario
La aplicación cuenta con una interfaz web amigable en la que los usuarios pueden:

- Realizar preguntas sobre procedimientos académicos

- Ver el historial de preguntas y respuestas en forma de chat

# 🧪 Evaluación
Se implementaron dos métodos de evaluación:

- Automática: mediante similitud coseno entre respuestas esperadas y generadas.

- Manual / subjetiva: con usuarios reales que valoraron usabilidad, coherencia y satisfacción general.

# 📦 Instalación y uso local
git clone [https://github.com/usuario/ruleup-bot.git](https://github.com/jhosmanmore/chatbot-unipamplona.git)
cd ruleup-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

- Colocar el documento reglamento académico en documento.txt o en la carpeta Documentos/ como PDF.

- Ejecutar el script de extracción si es necesario (extractor.py).

- Configurar el archivo .env con las claves de API: PINECONE_API_KEY=tu_api_key, HUGGINGFACE_API_KEY=tu_token

- Inicia el servidor: python app.py

# 🌐 Despliegue
El proyecto se desplegó incialmente en las siguientes plataformas (actualmente sin despliegue activo): 

- Render

- Railway

## 📚 Créditos
Desarrollado por Jhosman Moreno como parte del Trabajo de Grado para la carrera de Ingeniería de Sistemas en la Universidad de Pamplona.

El actual proyecto es netamente académico y de uso personal, y no tiene ninguna implicación legal o relación con la Universidad de Pamplona.
