import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.5
)


def process_calendar_query(query: str) -> str:
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        image_path = os.path.join(os.path.dirname(__file__), "cal.png")
        
        model = genai.GenerativeModel('gemini-pro-vision')
        
        prompt = f"""You are an assistant for a student.
        Please answer this question about the calendar: {query}
        If the answer is not in the image, say "I don't have that specific calendar information."
        """
        
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
            response = model.generate_content([prompt, image_data])
        
        return response.text
        
    except Exception as e:
        print(f"Error processing calendar query: {e}")
        return "I apologize, but I encountered an error processing the calendar query."
    
if __name__ == "__main__":
    query = "When is the next exam?"
    response = process_calendar_query(query, None)
    print(response)
