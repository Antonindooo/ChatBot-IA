
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnableMap, RunnableSequence
from typing import List, Dict, Any


import requests
import os
from dotenv import load_dotenv

load_dotenv()

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Gemini_(IA)"
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion"
]

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.
    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://fr.wikipedia.org/w/api.php"
    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAG_project/0.0.1"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

def get_retriever(PAGES):
    all_docs = []
    for page in PAGES:
        page_content = get_wikipedia_page(page)
        if page_content:
            docs = [Document(page_content=page_content, metadata={"title": page, "url": f"https://fr.wikipedia.org/wiki/{page.replace(' ', '_')}"})]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_docs.extend(text_splitter.split_documents(docs))
    # Embed
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings)
    return vectorstore.as_retriever()

# Instantiate the retriever globally
retriever = get_retriever(PAGES)

# Define custom Runnable classes if necessary
class RetrieverRunnable(Runnable):
    def invoke(self, input: str, config: Any = None) -> List[Document]:
        # Use the retriever to get relevant documents
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
    # Prompt
    template = """Answer the question based only on the following context:
{context}

Question: {query}

Source: {source}
"""

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Create instances of custom Runnables
    retriever_runnable = RetrieverRunnable()
    process_docs_runnable = ProcessDocsRunnable() 

    # Build the chain using RunnableSequence and RunnableMap
    rag_chain = RunnableSequence(
            RunnableMap({
                "retrieved_docs": retriever_runnable,
                "query": lambda x: x  # Identity function
            }),
            process_docs_runnable,
            prompt,
            model,
            StrOutputParser()
        
    )

    # Invoke the chain with the query
    return rag_chain.invoke(query)

if __name__ == "__main__":
    print('Building retriever...')
    # The retriever is already built globally

    try:
        while True:
            print('-' * 50)
            print('Posez une question :')
            question = input('> ')
            print()
            
            print(rag_answer(question))
            print('\n')

    except KeyboardInterrupt:
        print("\nExiting...")
