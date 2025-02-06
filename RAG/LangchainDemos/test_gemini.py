from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_gemini_api():
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key="AIzaSyBy8sLcMwHrhRTlpmGyBAGvisuvxX5WGF8", 
            temperature=0.7,
        )
        
        test_prompt = "Write a short poem about artificial intelligence"
        
        response = llm.invoke(test_prompt)
        
        print("\nTest Results:")
        print("=" * 50)
        print("Status: Success ")
        print("Response from Gemini:")
        print("-" * 50)
        print(response.content)
        print("=" * 50)
        
    except Exception as e:
        print("\nTest Results:")
        print("=" * 50)
        print("Status: Failed ")
        print("Error Message:")
        print("-" * 50)
        print(f"Error: {str(e)}")
        print("=" * 50)

if __name__ == "__main__":
    test_gemini_api()
