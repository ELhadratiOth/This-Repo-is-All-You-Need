superheroes = [
    {"Superman": "An alien from Krypton with super strength, flight, and heat vision. Fights for truth, justice, and the American way."},
    {"Batman": "A wealthy detective and martial artist who uses his intellect, gadgets, and fear tactics to fight crime in Gotham City."},
    {"Wonder Woman": "An Amazonian warrior princess with superhuman strength, combat skills, and a lasso of truth."},
    {"Spider-Man": "A young man with spider-like abilities who uses his powers to protect New York City while juggling life as a student."},
    {"Iron Man": "A genius billionaire who builds a powerful suit of armor to fight evil and protect the world."},
    {"Captain America": "A super soldier from World War II with enhanced abilities and an indestructible shield, symbolizing justice and honor."},
    {"Black Panther": "The king of Wakanda who possesses enhanced strength and senses, and wears a suit made of vibranium."},
    {"Hulk": "A scientist who transforms into a giant green brute with immense strength when angry."},
    {"Thor": "The Norse god of thunder who wields the enchanted hammer Mjolnir and protects both Asgard and Earth."},
    {"Flash": "A speedster who can move at superhuman speeds, often using his powers to manipulate time and fight crime."}
]

from dotenv import load_dotenv


load_dotenv(override=True)

from langchain.docstore.document import Document
docs = [
    Document(
        page_content="\n".join([
            f"superheroName: {list(superheroe.keys())[0]}",
            f"superhero Description: {list(superheroe.values())[0]}"
        ]),
        metadata={"superhero_name": list(superheroe.keys())[0]}
    )
    for superheroe in superheroes
]

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="superheros_collection",
    embedding_function=embeddings,
)
vector_store.add_documents(documents=docs)


def get_superheroes(query: str) -> list[str]:
    """
    Retrieves a list of superheroes based on a given query.

    Args:
        query (str): The query to search for superheroes.

    Returns:
        list[str]: A list of superheroes that match the query.
    """
    docs_and_scores = vector_store.similarity_search(query , k=1)
    # print(docs_and_scores)
    if len(docs_and_scores) == 0:
        return ["No superheroes found"]
    return docs_and_scores
from langchain.tools import Tool

# get_superheroes("Hulk")
guest_info_tool = Tool(
    name="superheroes_info_retriever",
    func=get_superheroes,
    description="Retrieves detailed information about get_superheroes based on their name or info."
)