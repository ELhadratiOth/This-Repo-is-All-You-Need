from dotenv import load_dotenv
import os

def show_gemini_api_key():
    load_dotenv(override=True)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if api_key:
        print("\nYour Gemini API Key is:")
        print("-" * 50)
        print(api_key)
        print("-" * 50)
    else:
        print("No Gemini API key found in .env file")

if __name__ == "__main__":
    show_gemini_api_key()
