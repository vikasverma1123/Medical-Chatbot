from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "medical-chatbot"


def _get_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise RuntimeError(
        "Missing Pinecone API key! Create a .env file in the project root with:\n"
        "PINECONE_API_KEY=your-pinecone-key"
    )


def build_index() -> None:
    pinecone_api_key = _get_required_env("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    extracted_data = load_pdf_file(data="data/")
    filter_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filter_data)

    try:
        embeddings = download_hugging_face_embeddings()
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Hugging Face embedding model. "
            "Make sure the machine can reach huggingface.co or that the model is cached locally."
        ) from exc

    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


if __name__ == "__main__":
    build_index()
    print(f"Index '{INDEX_NAME}' is ready.")
