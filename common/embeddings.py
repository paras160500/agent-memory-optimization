# =========================================================================================================
#                                           Import/Init Statements
# =========================================================================================================

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os  


def get_embeddings():
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
        dimensions=1536
    )