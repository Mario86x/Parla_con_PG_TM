import json
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
from llama_index.core.node_parser import SentenceSplitter


def build_chunks_with_outline(pdf_path: str,
                              chunk_size: int = 2000,
                              chunk_overlap: int = 100) -> List[Dict]:
    """Estrae testo dal PDF, lo chunkizza con LlamaIndex e aggiunge i metadati dai segnalibri."""
    doc = fitz.open(pdf_path)
    outline = doc.get_toc()  # lista di [livello, titolo, pagina]

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    results = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue

        # Trova il contesto (capitolo, sottocapitolo, sezione) dai segnalibri
        context = {"capitolo": "", "sezione": "", "paragrafo": ""}
        for level, title, pg in outline:
            if pg <= page_num:
                if level == 1:
                    context["capitolo"] = title
                elif level == 2:
                    context["sezione"] = title
                elif level == 3:
                    context["paragrafo"] = title

        # Split in chunk con llamaindex
        chunks = splitter.split_text(text)

        for chunk in chunks:
            results.append({
                "content": chunk,
                "metadata": {
                    "page": page_num,
                    **context
                }
            })

    return results


if __name__ == "__main__":
    pdf_path = "docs/ardania.pdf"
    chunks = build_chunks_with_outline(pdf_path)

    # Salvataggio JSON
    out_path = Path("docs/ardania_chunks.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Estratti {len(chunks)} chunk, salvati in {out_path}")
