import logging
import uuid
from pathlib import Path
from typing import List, Optional

import requests
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        self.splitter_cache: dict = {}

    def load_pdf(self, filepath: str) -> List[Document]:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        logger.info("Loaded %d pages from PDF: %s", len(docs), filepath)
        for i, doc in enumerate(docs):
            doc.metadata.setdefault("source", filepath)
            doc.metadata.setdefault("page", i)
        return docs

    def load_directory(
        self, path: str, glob: str = "**/*.pdf"
    ) -> List[Document]:
        loader = DirectoryLoader(
            path,
            glob=glob,
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        logger.info(
            "Loaded %d documents from directory: %s (glob=%s)",
            len(docs),
            path,
            glob,
        )
        for doc in docs:
            doc.metadata.setdefault("source", str(Path(doc.metadata.get("source", path)).resolve()))
        return docs

    def load_web(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        docs = loader.load()
        logger.info("Loaded %d documents from URL: %s", len(docs), url)
        for doc in docs:
            doc.metadata["source"] = url
        return docs

    def chunk_documents(
        self,
        docs: List[Document],
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[Document]:
        key = (chunk_size, overlap)
        if key not in self.splitter_cache:
            self.splitter_cache[key] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
        splitter = self.splitter_cache[key]
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata.setdefault("chunk_index", i)

        logger.info(
            "Split %d documents into %d chunks (size=%d, overlap=%d)",
            len(docs),
            len(chunks),
            chunk_size,
            overlap,
        )
        return chunks

    def load_confluence(
        self,
        base_url: str,
        space_key: str,
        token: str,
        max_pages: int = 500,
    ) -> List[Document]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        session = requests.Session()
        session.headers.update(headers)

        documents: List[Document] = []
        start = 0
        limit = 50

        while True:
            url = (
                f"{base_url}/rest/api/content"
                f"?spaceKey={space_key}&expand=body.storage,version,ancestors"
                f"&start={start}&limit={limit}&type=page"
            )
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                page_id = page["id"]
                title = page["title"]
                body_html = page.get("body", {}).get("storage", {}).get("value", "")
                version = page.get("version", {}).get("number", 1)
                page_url = f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"

                # Strip basic HTML tags for plain text
                import re
                plain_text = re.sub(r"<[^>]+>", " ", body_html)
                plain_text = re.sub(r"\s+", " ", plain_text).strip()

                doc = Document(
                    page_content=plain_text,
                    metadata={
                        "source": page_url,
                        "title": title,
                        "page_id": page_id,
                        "version": version,
                        "space_key": space_key,
                        "type": "confluence",
                    },
                )
                documents.append(doc)

            start += limit
            if start >= data.get("size", 0) or len(documents) >= max_pages:
                break

        logger.info(
            "Loaded %d pages from Confluence space %s", len(documents), space_key
        )
        return documents

    def load_and_chunk(
        self,
        source: str,
        source_type: str = "pdf",
        chunk_size: int = 1000,
        overlap: int = 200,
        **kwargs,
    ) -> List[Document]:
        if source_type == "pdf":
            docs = self.load_pdf(source)
        elif source_type == "directory":
            docs = self.load_directory(source, **kwargs)
        elif source_type == "web":
            docs = self.load_web(source)
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
        return self.chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)

# _r 20260528133208-a31a2026
