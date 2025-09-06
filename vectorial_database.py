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
        

    def _nearest_chunks(self, embedding: [float], top_n: int = 3) -> list[str]:
        """Return the nearest chunks to the given embedding with the dot product similarity"""
        # Calculate dot product similarity for each stored embedding
        similarities = []
        for i, stored_embedding in enumerate(self.embeddings):
            dot_product = sum(a * b for a, b in zip(embedding, stored_embedding))
            similarities.append((dot_product, i))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        return [self.chunks[i] for _, i in similarities[:top_n]]

    def nearest_chunks(self, text: str) -> list[str]:
        """
        Return the nearest chunks to the given text by getting embeddings from the input and
        calculating similarity with _nearest_chunks function.
        Args:
          text: input text to calculate similarity.
        """
        response = self.client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        return self._nearest_chunks(embedding)