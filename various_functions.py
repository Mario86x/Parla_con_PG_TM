import hashlib

def chunk_id_from_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


import numpy as np

def normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


from datetime import date

# Mappa mesi reali -> mesi di Ardania
MONTHS_MAP = {
    1: "Postapritore",
    2: "Forense",
    3: "Macinale",
    4: "Adulain",
    5: "Madrigale",
    6: "Granaio",
    7: "Lithe",
    8: "Antedain",
    9: "Solfeggiante",
    10: "Orifoglia",
    11: "Nembonume",
    12: "Dodecabrullo"
}

def real_to_ardania(real_date: date, year_offset: int = 1736):
    """
    Converte una data reale in data imperiale di Ardania.
    
    Args:
        real_date (date): data del calendario gregoriano.
        year_offset (int): numero da sottrarre all'anno reale.
    
    Returns:
        (giorno, nome_mese, anno_ai)
    """
    day = real_date.day
    month_name = MONTHS_MAP[real_date.month]
    year_ai = real_date.year - year_offset
    
    return day, month_name, year_ai


# 🔹 Esempio d'uso:
if __name__ == "__main__":
    # Esempio per real_to_ardania
    oggi = date.today()
    giorno, mese, anno_ai = real_to_ardania(oggi)
    print(f"Data reale: {oggi}")
    print(f"Data di gioco: {giorno} {mese} {anno_ai} A.I.")