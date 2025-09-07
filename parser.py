from llama_parse import LlamaParse
import os
from dotenv import load_dotenv
load_dotenv()

def parse_pdf_to_markdown(pdf_path: str, output_path: str = "output.md") -> None:
    """
    Converte un file PDF in markdown utilizzando LlamaParse e salva l'output su file.

    Args:
        pdf_path (str): percorso del file PDF da convertire.
        output_path (str): percorso del file markdown in cui salvare il risultato.
    """
    parser = LlamaParse(
        api_key=os.getenv("LLAMAPARSE_API_KEY"),
        result_type="json",
        verbose=True,
    )

    extra_info = {"file_name": pdf_path}
    # creo il file output_path, sovrarscrivendolo se esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # with open(pdf_path, "rb") as f:
    #     # È necessario fornire extra_info con il file_name quando si passa un file object
    #     documents = parser.load_data(f, extra_info=extra_info)
    documents = parser.get_json_result(f, extra_info=extra_info)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text)

if __name__ == "__main__":

    parse_pdf_to_markdown("docs/ardania.pdf", "knowledge/ardania.json")