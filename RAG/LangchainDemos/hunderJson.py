from operator import itemgetter
from dotenv import load_dotenv
import os
from langchain_core import embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
import re
from pathlib import Path
from typing import Callable, Union
from fastapi import FastAPI, HTTPException
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_core.messages import trim_messages
from fastapi.middleware.cors import CORSMiddleware
import json

load_dotenv(override=True)

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"    
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = "ensah-monitoring"
required_vars = ['QDRANT_URL', 'QDRANT_API_KEY', 'GOOGLE_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")


def _is_valid_identifier(value: str) -> bool:
    if not value or len(value) > 100:  
        return False
    valid_characters = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    return bool(valid_characters.match(value))

def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        session_id = session_id.strip()
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid session ID: '{session_id}'. Session ID must start with a letter or number "
                    "and contain only alphanumeric characters, hyphens, or underscores."
                ),
            )
        
        try:
            file_path = base_dir_ / f"{session_id}.json"
            print(file_path)
            return FileChatMessageHistory(str(file_path))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create chat history file: {str(e)}"
            )

    return get_chat_history


qdrant_client =  QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
)


embeddings  = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="semantic_similarity",
    google_api_key=os.getenv("GOOGLE_API_KEY"),)


llm  =  GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

vector_store =  QdrantVectorStore(
    client=qdrant_client,
    collection_name="ensah_data",
    embedding=embeddings,
)
retriever=vector_store.as_retriever(search_kwargs={"k": 6} )


message_trimmer = trim_messages(
    max_tokens=3,
    strategy="last",
    include_system=True,
    start_on="human",
    token_counter=len,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """you are an AI Agent specialized in answering questions about ENSAH.
Your role is to provide accurate and helpful answers about the programs, professors, and training offered at these institutions.

Guidelines:
- Use the provided context to create your response , also use the chat history
- If the question is related to ENSAH, use only relevant context information
- Explain using simple and clear words
- Match the language of the question
- Respond with 'I don't have that information in my knowledge base' if unable to answer
- Your name is: ENSA AL Hociema Chatbot
- Your Response should  not  include somthing like this  in the  beginning : "AI :"
- Try to  be gentil and friendly
Context: {context}
    """),
        MessagesPlaceholder(variable_name="chat_history"),
    
    ("human", "{input}"),
])

def log_prompt(messages):
    print("\n=== Complete Prompt Being Sent to LLM ===")
    for msg in messages:
        print(f"\nRole: {msg}")
        print(f"Content: {msg}")
    print("=====================================\n")
    return messages

from langchain_core.runnables import RunnablePassthrough
chain = (
    # {
    #     "context": lambda x: x["context"], 
    #     "chat_history": message_trimmer,  
    #     "input": lambda x: x["input"]
    # } 
    # RunnablePassthrough.assign(messages=itemgetter("chat_history") | message_trimmer)
    # |
    prompt 
    # | log_prompt  
    | llm
)
# print("this is prompt :")
# print(prompt)

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(
    retriever,
    chain,
)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    # The field extra defines a chat widget.
    # As of 2024-02-05, this chat widget is not fully supported.
    # It's included in documentation to show how it should be specified, but
    # will not work until the widget is fully supported for history persistence
    # on the backend.
    input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "input"}},
    )

CHAT_HISTORY_DIR = Path("chat_histories")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Ensure we have a valid session ID
    session_id = session_id.strip()
    if not _is_valid_identifier(session_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid session ID: '{session_id}'. Session ID must start with a letter or number "
                "and contain only alphanumeric characters, hyphens, or underscores."
            ),
        )
    if not session_id or not session_id.startswith('streamlit_'):
        raise ValueError("Invalid session ID format")
        
    history_file = CHAT_HISTORY_DIR / f"{session_id}.json"
    # Always start with a fresh history object
    history = FileChatMessageHistory(str(history_file))

    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            try:
                saved_messages = json.load(f)

                # Only keep the last two messages
                last_messages = saved_messages[-2:] if len(saved_messages) > 2 else saved_messages

                # # Reset history and add only the last two messages
                history.clear()  # <--- Ensures only the last two messages are stored

                for msg in last_messages:
                    if msg['type'] == 'human':
                        history.add_user_message(msg['data']['content'])
                    elif msg['type'] == 'ai':
                        history.add_ai_message(msg['data']['content'])

            except json.JSONDecodeError:
                pass  # If file is corrupted, start with an empty history
    print("here is  the history :")
    print(history)
    return history


from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
).with_types(input_type=InputChat)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(
    app,
    chain_with_history,
    path="/chat",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
