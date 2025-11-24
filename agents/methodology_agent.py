import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from database.db_connector import chroma_client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Оставляем локальные эмбеддинги (они бесплатные и быстрые)
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

class MethodologyAgent:
    """
    Агент-Партнер (использует 'умную' модель с Reasoning, если доступна).
    """
    def __init__(self, user_id: str = "default_user", model_name: str = "deepseek/deepseek-r1:free"):
        self.chat = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API_KEY'),
            model=model_name,
            temperature=0.6,
            default_headers={"HTTP-Referer": "https://github.com/ai-thinker"}
        )

        collection_name = f"methodology_memory_{user_id}"
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        self.message_history = []
        print(f"MethodologyAgent инициализирован ({model_name}).")

    def execute(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # RAG (поиск контекста)
            results = self.collection.query(query_texts=[user_prompt], n_results=3)
            relevant_contexts = results["documents"][0] if results["documents"] else []

            context_block = ""
            if relevant_contexts:
                context_block = "📌 Контекст из прошлого:\n" + "\n".join([f"- {ctx}" for ctx in relevant_contexts])

            messages = [SystemMessage(content=system_prompt + "\n" + context_block)]

            # Краткая история текущей сессии
            for role, content in self.message_history[-4:]:
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(content=content)) # Или AIMessage

            messages.append(HumanMessage(content=user_prompt))

            response = self.chat.invoke(messages)

            # Сохраняем в память
            self.collection.add(
                ids=[f"turn_{len(self.message_history)}"],
                documents=[user_prompt + " -> " + response.content],
                metadatas=[{"type": "interaction"}]
            )
            self.message_history.append(("user", user_prompt))
            self.message_history.append(("ai", response.content))

            return response.content

        except Exception as e:
            print(f"MethodologyAgent Error: {e}")
            return "Ошибка методологического ядра."

    def clear_memory(self):
        self.message_history = []
