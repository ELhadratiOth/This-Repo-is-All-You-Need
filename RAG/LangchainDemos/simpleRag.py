import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["QDRANT_API_KEY"] = os.getenv("QDRANT_API_KEY")
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL")

llm = GoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(  model="models/text-embedding-004",

)


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=True
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="ensah_data",
    embedding=embeddings
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI Agent specialized in answering questions about ENSAH.
    Use the provided context and chat history to create your response:
    - Refer to both the context and chat history to provide complete answers
    - If a question refers to previous context, use that information
    - If the information cannot be found in context or history, respond with 'thella'
    - Always respond in the same language as the question
    -at the  end u should  tell him if  he had more question or not (be more creative on this)
    
    Context:
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
)

retriever=vector_store.as_retriever()
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=chain,
)


store = {}
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

from langchain_core.runnables.history import RunnableWithMessageHistory

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=retrieval_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


def ask_question(question, session_id="my_session"):
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    return response["answer"]

if __name__ == "__main__":
    while True:
        question = input("Ask a question: ")
        print(ask_question(question))








