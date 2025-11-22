import streamlit as st
import requests
import os  
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnableMap, RunnableSequence

load_dotenv()

PAGES = [
    "Intelligence_artificielle",
    "Apprentissage_automatique",
    "Réseaux_de_neurones_artificiels",
    "GPT-3",
    "GPT-4",
    "Modèle_de_langage",
    "Vision_par_ordinateur",
    "Reconnaissance_vocale"
]

def get_wikipedia_page(title: str):
    URL = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAG_project/0.0.1"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None



# Chemin pour la base de données persistante
PERSIST_DIR = "./chroma_db_wikipedia"

@st.cache_resource
def get_retriever(PAGES):
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(PERSIST_DIR):
        print(f"Chargement de la base de données depuis {PERSIST_DIR}...")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, 
                             embedding_function=embeddings)
    else:
        print(f"Création de la base de données dans {PERSIST_DIR}...")
        with st.spinner("Construction du 'retriever' (téléchargement et indexation)..."):
            all_docs = []
            for page in PAGES:
                page_content = get_wikipedia_page(page)
                if page_content:
                    docs = [Document(page_content=page_content, metadata={"title": page, "url": f"https://fr.wikipedia.org/wiki/{page.replace(' ', '_')}"})]
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)                    
                    all_docs.extend(text_splitter.split_documents(docs))

            # Créer la base de données ET la sauvegarder sur le disque
            vectorstore = Chroma.from_documents(documents=all_docs, 
                                                embedding=embeddings, 
                                                persist_directory=PERSIST_DIR)
    
    return vectorstore.as_retriever()


# Instantiate the retriever globally (utilisera le cache)
retriever = get_retriever(PAGES)

# Define custom Runnable classes if necessary
class RetrieverRunnable(Runnable):
    def invoke(self, input: str, config: Any = None) -> List[Document]:
        return retriever.invoke(input)

class ProcessDocsRunnable(Runnable):
    def invoke(self, inputs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        retrieved_docs = inputs['retrieved_docs']
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        source = "\n".join(
            f"Title: {doc.metadata['title']}, URL: {doc.metadata['url']}"
            for doc in retrieved_docs
        )
        return {
            "context": context,
            "source": source,
            "query": inputs['query']
        }

def rag_answer(query):
   
    template = """Answer the question based only on the following context:
{context}

Question: {query}

Source: {source}
"""
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever_runnable = RetrieverRunnable()
    process_docs_runnable = ProcessDocsRunnable()

    rag_chain = RunnableSequence(
            RunnableMap({
                "retrieved_docs": retriever_runnable,
                "query": lambda x: x
            }),
            process_docs_runnable,
            prompt,
            model,
            StrOutputParser()
    )
    return rag_chain.invoke(query)


st.title("TP sur un Chatbot RAG")
st.caption("Basé sur un ensemble de pages Wikipédia.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez une question :"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Cherchant une réponse..."):
            response = rag_answer(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})