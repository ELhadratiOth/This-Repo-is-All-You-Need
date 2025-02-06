import os
from typing import Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_qdrant import QdrantVectorStore
import google.generativeai as genai
from langchain_core.messages import trim_messages

class ENSAHChatbot:
    def __init__(self):
        load_dotenv()
        
        self.initialize_qdrant()
        self.initialize_llm()
        self.initialize_vector_store()
        self.initialize_chat_components()
        
    def initialize_qdrant(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        
    def initialize_llm(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="semantic_similarity",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            generation_config=genai.GenerationConfig(
                temperature=0.5,
            ),
        )
        
    def initialize_vector_store(self):
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="ensah_data",
            embedding=self.embeddings
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
    def initialize_chat_components(self):
        self.chat_histories: Dict[str, InMemoryChatMessageHistory] = {}
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI Agent specialized in answering questions about ENSAH.
Your role is to provide accurate and helpful answers about the programs, professors, and training offered at these institutions.

Guidelines:
- Use the provided context to create your response
- If the question is related to ENSAH, use only relevant context information
- Explain using simple and clear words
- Match the language of the question
- Respond with 'I don't have that information in my knowledge base' if unable to answer
- Your name is: ENSAH Chatbot

Context: {context}
History: {history}
"""),
            ("human", "{question}")
        ])
        
        self.message_trimmer = trim_messages(
            max_tokens=30,
            strategy="last",
            token_counter=self.llm,
            include_system=True,
            start_on="human",
        )
        
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = InMemoryChatMessageHistory()
        return self.chat_histories[session_id]
        
    async def handle_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        try:
            
            docs = await self.retriever.ainvoke(query)
            context = "\n".join(doc.page_content for doc in docs)
            
            chain = (
                {"context": lambda _: context, "history": self.message_trimmer, "question": lambda x: x} 
                | self.qa_prompt 
                | self.llm
            )
            
            chain_with_history = RunnableWithMessageHistory(
                chain,
                self.get_session_history
            )
            
            response = await chain_with_history.ainvoke(
                query,
                config={"configurable": {"session_id": session_id}},
            )
            
            return {
                "answer": response.content,
                "documents": [{
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs],
                "source": "knowledge_base"
            }
            
        except Exception as e:
            print(f"Error in handle_query: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error. Please try rephrasing your question.",
                "documents": [],
                "source": "error"
            }

async def main():
    chatbot = ENSAHChatbot()
    session_id = "test_session"
    
    print("ENSAH Chatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("User: ").strip()
        if query.lower() == 'quit':
            break
            
        response = await chatbot.handle_query(query, session_id)
        print("\nAssistant:", response["answer"])
        
        if response["source"] != "error" and response["documents"]:
            print("\nRetrieved Documents:")
            for i, doc in enumerate(response["documents"], 1):
                print(f"\nDocument {i}:")
                print("Content:", doc["content"])
                print("Metadata:", doc["metadata"])
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())