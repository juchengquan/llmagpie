import chromadb
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal, Union


class ChromaDocument(BaseModel):
    class Config:
        extra = "forbid"

    id: str
    document: str
    embedding: List[float]
    metadata: Dict


class ChromaDBStore:
    client: chromadb.ClientAPI
    mode: Literal["local", "client"]
    host: Optional[str] = None
    port: Optional[str] = None
    persistent_path: Optional[str] = None

    def __init__(
        self,
        persistent_path: Optional[str] = None,
        host: str = "localhost",
        port: int = "9999",
    ):
        if persistent_path is not None:
            self.client = chromadb.PersistentClient(
                path=persistent_path,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True),
            )
            self.persistent_path = persistent_path
            self.mode = "local"

        elif (host is not None) or (port is not None):
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True),
            )
            self.host = host
            self.port = port
            self.mode = "client"

    def get_collection(self, collection_name: str, create_if_not_exist: bool = False):
        """"""
        if create_if_not_exist:
            return self.client.get_or_create_collection(collection_name)
        else:
            return self.client.get_collection(collection_name)

    def cleanup_collection(self, collection_name: str):
        """"""
        collection = self.get_collection(collection_name, create_if_not_exist=False)
        collection.delete(collection.get()["ids"])

    def remove_collection(self, collection_name: str, safe_remove: bool = True):
        """"""
        collection = self.get_collection(collection_name, create_if_not_exist=False)
        if safe_remove:
            assert collection.count() == 0, "The collection is not empty"
        self.client.delete_collection(collection_name)

    def add_document(self, collection_name: str, document: ChromaDocument):
        """"""
        self.get_collection(collection_name, create_if_not_exist=False) \
            .add(
                ids=document.id,
                documents=document.document,
                embeddings=document.embedding,
                metadatas=document.metadata)

    def add_documents(self, collection_name: str, documents: List[ChromaDocument]):
        """"""
        ids, documents, embeddings, metadatas = \
            ([getattr(d, k) for d in documents] for k in ("id", "document", "embedding", "metadata"))

        self.get_collection(collection_name, create_if_not_exist=False) \
            .add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,)

    def query_embeddings(
        self,
        collection_name: str,
        query_embeddings: List,
        top_k: int = 10,
        include: List[Union[Literal['documents'], Literal['embeddings'], Literal['metadatas'], Literal['distances'], Literal['uris'], Literal['data']]] = ['metadatas', 'documents', 'distances'],
    ):
        """"""
        return self.get_collection(collection_name, create_if_not_exist=False) \
            .query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=include,)

    def purge_database(self):
        """"""
        if self.mode == "local":
            self.client.reset()
            self.client = chromadb.PersistentClient(
                path=self.persistent_path,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True),
            )
        elif self.mode == "client":
            raise ValueError("Purge is not allowed in remote mode.")
            # self.client = chromadb.HttpClient(
            #     host=host,
            #     port=port,
            #     settings=chromadb.Settings(
            #         anonymized_telemetry=False,
            #         allow_reset=True),
            # )
        else:
            raise ValueError
