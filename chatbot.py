from abc import ABC, abstractmethod
from vectorial_database import VectorDDBB
import sys

class chatbotInterface(ABC):
    
    @abstractmethod
    def ask_question(question: str) -> str:
        """
        Receives a question from the user related to the document and returns the best possible repsonse based on
        OpenAI LLMs and the augmented prompt
        Args:
            question: user input asking things about the document.
        """
        pass

class chatbot(chatbotInterface):
    def __init__(self) -> None:
        """
        Read the document and calculate the embeddings to create the vectorial database
        """
        # Create instance
        self.vector_db = VectorDDBB()
        
        # Load document and create embeddings
        print("Cargando BBDD....")
        self.vector_db.load_document_from_path("./ai-engineer-evaluation-test.md")

        self.client = self.vector_db.client

    def ask_question(self, question: str) -> str:
        nearest_chunks = self.vector_db.nearest_chunks(question)

        system_prompt = """
            Eres un asistente que responde preguntas basadas en un contexto que el usuario te va a proporcionar.
            El usuario te va a proporcionar el contexto en primer lugar y después la pregunta.
            Responde solo con la información que se pueda extraer del contexto y si no sabes la respuesta indica que no puedes
            resolver la petición del usuario y que trate de mejorar la pregunta
        """

        user_prompt = f"""
            Contexto:
            ---
            {"\n".join(nearest_chunks)}
            ---
            
            Pregunta:
            {question}
        """

        response = self.client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    print("Bienvenido al chat de la guía del training AI Engineer")
    chat = chatbot()
    while True:
        string = input("Indique la pregunta que desea realizar sobre el documento, si desea salir escriba 'exit': ")
        if string == "exit":
            sys.exit(0)
        
        print(f"Respuesta:\n{chat.ask_question(string)}\n-------------------------------------------------------------")