import fitz  # PyMuPDF
import re
import json
import hashlib

def text_pdf_to_markdown(pdf_path: str, md_path: str):
    doc = fitz.open(pdf_path)

    # Recupera outline (lista di tuple: livello, titolo, pagina)
    outline = doc.get_toc(simple=True)

    # Prepara lista con i titoli già scritti per evitare duplicati
    written_titles = set()

    with open(md_path, "w", encoding="utf-8") as f:
        for level, title, page_num in outline:
            # Normalizza titolo (togli spazi doppi)
            clean_title = re.sub(r"\s+", " ", title.strip())

            if clean_title not in written_titles:
                # Genera il prefisso di heading in base al livello
                heading = "#" * level
                f.write(f"{heading} {clean_title}\n\n")
                written_titles.add(clean_title)

            # Estrae testo dalla pagina associata al titolo
            if 0 <= page_num - 1 < len(doc):
                page = doc[page_num - 1]
                text = page.get_text("text")
                if text.strip():
                    f.write(text.strip() + "\n\n")

    print(f"✅ Conversione completata: {md_path}")


if __name__ == "__main__":
    text_pdf_to_markdown("docs/ardania.pdf", "lore_md/ardania.md")