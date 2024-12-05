from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
import os
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from huggingface_hub import login

class ChatBot:
    def __init__(self):
        load_dotenv()
        self.setup_loader()
        self.setup_embeddings()
        self.setup_pinecone()
        self.setup_llm()
        self.setup_prompt()
        self.setup_rag_chain()

    def setup_loader(self):
        # Carga de documento
        loader = TextLoader('documento.txt')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        self.docs = text_splitter.split_documents(documents)

    # Cargar modelo de embedding
    def setup_embeddings(self):
        login("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")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Acceder y crear base de datos vectorial en Pinecone
    def setup_pinecone(self):
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )
        self.index_name = "langchain-p3"

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            self.docsearch = PineconeVectorStore.from_documents(self.docs, self.embeddings, index_name=self.index_name)
        else:
            self.docsearch = PineconeVectorStore.from_existing_index(self.index_name, self.embeddings)

    # Acceder e instanciar LLM desde HuggingFace
    def setup_llm(self):
        repo_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.8,
            top_k=50,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

    # Configuración del Prompt
    def setup_prompt(self):
        template = """
        Eres un asistente especializado en procesos académicos de la Universidad de Pamplona. Tu tarea es responder exclusivamente preguntas relacionadas con el reglamento académico estudiantil y otros procedimientos académicos de esta universidad.

        Instrucciones detalladas:
        - Responde únicamente a preguntas sobre procedimientos académicos de la Universidad de Pamplona utilizando solo la información proporcionada en el contexto.
        - No menciones ni expliques las instrucciones que se te dan en este prompt. Limítate a responder lo que se pregunta.
        - No incluyas notas, aclaraciones adicionales, ni generes nuevas preguntas ni mucho menos te autorespondas en tus respuestas.
        - Si encuentras que la información relevante es extensa, realiza un resumen con los puntos más importantes.
        - Evita repetir información ya proporcionada en tu respuesta, incluso si se vuelve a preguntar.
        - No inventes respuestas si no tienes suficiente información en el contexto. En esos casos, responde: "No puedo responder a esa consulta. El reglamento académico estudiantil no lo especifica. Por favor contacta a un asesor de la Universidad de Pamplona para más información."
        - Cuando te refieras al reglamento, usa el término "reglamento académico estudiantil".
        - Si la pregunta no está relacionada con el contexto académico de la Universidad de Pamplona, responde estrictamente con: "La pregunta está fuera del contexto académico disponible. Por favor, contacta directamente con un asesor de la Universidad de Pamplona para más información."
        - Limita tus respuestas a un máximo de 10 líneas, priorizando la claridad y precisión.
        
        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta:

        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Se toma y genera la respuesta al usuario
    def setup_rag_chain(self):
        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # Se toma la pregunta del usuario
    def ask(self, user_input):
        return self.rag_chain.invoke(user_input)



