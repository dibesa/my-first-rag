import openai
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod

class VectorDatabaseInterface(ABC):
    """
    Abstract interface for vector database implementations.
    
    Expected constructor signature:
        __init__(self) -> None
            Should initialize:
            - self.client: openai.OpenAI client
            - self.embeddings: List for storing embeddings
            - self.chunks: List for storing text chunks
    """
    
    @abstractmethod
    def load_document(self, markdown_text: str) -> None:
        """Load and process a document into embeddings."""
        pass
    
    @abstractmethod
    def print_number_of_embeddings(self) -> None:
        """Print the number of stored embeddings."""
        pass

class VectorDDBB(VectorDatabaseInterface):
    
    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("API_KEY")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://llmproxy.ai.orange"
        )
        self.embeddings = list()
        self.chunks = list()

    def load_document(self, markdown_text: str) -> None:
        """
        Splits text into array (chunks) by the particle "##" (Sections).
        For each chunk get the corresponding embedding and save it in self.embeddings list

        Args:
          markdown_text: input text to split and load into the database.
        """
        self.chunks = markdown_text.split("##")
        for chunk in self.chunks:
            response = self.client.embeddings.create(
                model="openai/text-embedding-3-small",
                input=chunk
            )
            self.embeddings.append(response.data[0].embedding)

    def print_number_of_embeddings(self) -> None:
        """Print the number of stored embeddings"""
        print(len(self.embeddings))
    