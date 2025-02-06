import requests
import uuid
import json

def chat_with_bot(message: str, session_id: str = None):
    """Send a message to the chatbot and get the response"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    payload = {
        "input": {
            "input": message
        },
        "config": {
            "configurable": {
                "session_id": "test"
            }
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/chat/invoke",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        
        # Extract and print the response
        if "output" in result:
            answer = result["output"].get("answer", "No answer received")
            print("\nAssistant:", answer)
            
            # Print source documents if available
            if "source_documents" in result["output"]:
                docs = result["output"]["source_documents"]
                if docs:
                    print("\nSource Documents:")
                    for i, doc in enumerate(docs, 1):
                        print(f"\nDocument {i}:")
                        print(doc.get("page_content", ""))
        
        return result
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    """Run an interactive chat session"""
    print("Starting chat session...")
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input:
            chat_with_bot(user_input, session_id)

if __name__ == "__main__":
    main()