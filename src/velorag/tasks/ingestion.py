from __future__ import annotations

import time
from pathlib import Path

from velorag.tasks.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_document(self, doc_id: str, kb_id: str, file_path: str):
    from velorag.db.base import get_session
    from velorag.db.models import Document, Chunk
    from velorag.document.loader import DocumentLoader
    from velorag.pipeline import VeloRAG

    session = get_session()
    try:
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return {"status": "error", "message": "Document not found"}

        doc.status = "processing"
        session.commit()

        rag = VeloRAG()
        path = Path(file_path)
        chunks = rag.ingest_file(str(path))

        for chunk in chunks:
            db_chunk = Chunk(
                id=chunk.id,
                doc_id=doc_id,
                kb_id=kb_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                token_count=len(chunk.text) // 4,
                metadata_=chunk.metadata,
            )
            session.add(db_chunk)

        doc.status = "completed"
        doc.chunk_count = len(chunks)
        session.commit()

        return {"status": "completed", "chunk_count": len(chunks)}

    except Exception as exc:
        doc = session.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = "failed"
            doc.error_message = str(exc)
            session.commit()
        raise self.retry(exc=exc)

    finally:
        session.close()


@celery_app.task(bind=True, max_retries=2)
def delete_document_chunks(self, doc_id: str, kb_id: str):
    from velorag.db.base import get_session
    from velorag.db.models import Chunk

    session = get_session()
    try:
        session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()
        session.commit()
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as exc:
        raise self.retry(exc=exc)
    finally:
        session.close()
