from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.6,
)

if __name__ == '__main__':
    response = llm.invoke("hi")
    print(response.content)
