from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="all-minilm:l6-v2"
    )
    return embeddings
