# -*- coding: utf-8 -*-
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from knowledge_base.cognitive_biases import COGNITIVE_BIASES
import os

class CognitiveBiasStore:
    """
    Manages a ChromaDB vector store for cognitive biases.
    """
    def __init__(self, collection_name="cognitive_biases", persist_directory="chroma_db/biases"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Ensure the persist directory exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        # Use the default, built-in embedding function for simplicity and stability
        self.embedding_function = DefaultEmbeddingFunction()

        # Initialize the ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function # Pass the function instance here
        )
        self._populate_if_empty()

    def _populate_if_empty(self):
        """
        Populates the vector store with cognitive biases if it's empty.
        This is a one-time operation.
        """
        if self.collection.count() == 0:
            print("Populating cognitive biases vector store...")
            documents = []
            metadatas = []
            ids = []
            for i, bias in enumerate(COGNITIVE_BIASES):
                content = f"Название: {bias['name']}. Описание: {bias['description']}"
                documents.append(content)
                metadatas.append({"name": bias['name'], "description": bias['description']})
                ids.append(str(i + 1))

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Vector store populated successfully.")

    def query_biases(self, query_text: str, n_results: int = 5):
        """
        Queries the vector store to find the most relevant cognitive biases.

        Args:
            query_text: The user's input text.
            n_results: The number of similar biases to return.

        Returns:
            A list of dictionaries, where each dictionary represents a cognitive bias.
        """
        if not query_text:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        # The query returns a list of lists for metadatas, so we take the first element.
        return results['metadatas'][0] if results['metadatas'] else []

# Example usage (can be run for testing)
if __name__ == '__main__':
    bias_store = CognitiveBiasStore()

    # Test population
    print(f"Collection count: {bias_store.collection.count()}")

    # Test query
    test_query = "Я думаю, что мой новый проект точно будет успешным, все знаки на это указывают."
    relevant_biases = bias_store.query_biases(test_query)

    print(f"\nQuery: '{test_query}'")
    print("\nMost relevant biases found:")
    for bias in relevant_biases:
        print(f"- {bias['name']}: {bias['description']}")
