Eseguire Milvus Lite a livello locale
Questa pagina illustra come eseguire Milvus localmente con Milvus Lite. Milvus Lite è la versione leggera di Milvus, un database vettoriale open source che alimenta le applicazioni di intelligenza artificiale con embeddings vettoriali e ricerca di similarità.

Panoramica
Milvus Lite può essere importato nella vostra applicazione Python, fornendo la funzionalità di ricerca vettoriale di base di Milvus. Milvus Lite è già incluso nell'SDK Python di Milvus. Può essere distribuito semplicemente con pip install pymilvus.

Con Milvus Lite, è possibile iniziare a costruire un'applicazione di intelligenza artificiale con ricerca di similarità vettoriale in pochi minuti! Milvus Lite può essere eseguito nei seguenti ambienti:

Notebook Jupyter / Google Colab
Computer portatili
Dispositivi Edge
Milvus Lite condivide le stesse API di Milvus Standalone e Distributed e copre la maggior parte delle funzionalità, come la persistenza e la gestione dei dati vettoriali, le operazioni CRUD sui vettori, la ricerca vettoriale rada e densa, il filtraggio dei metadati, la ricerca multivettoriale e ibrida. Insieme, forniscono un'esperienza coerente in diversi tipi di ambienti, dai dispositivi edge ai cluster nel cloud, adattandosi a casi d'uso di dimensioni diverse. Con lo stesso codice lato client, è possibile eseguire applicazioni GenAI con Milvus Lite su un laptop o un Jupyter Notebook, o Milvus Standalone su un container Docker, o Milvus Distributed su un cluster Kubernetes su scala massiccia che serve miliardi di vettori in produzione.

Prerequisiti
Milvus Lite supporta attualmente i seguenti ambienti:

Ubuntu >= 20.04 (x86_64 e arm64)
MacOS >= 11.0 (Apple Silicon M1/M2 e x86_64)
Si noti che Milvus Lite è adatto solo per casi di ricerca vettoriale su piccola scala. Per i casi di utilizzo su larga scala, si consiglia di utilizzare Milvus Standalone o Milvus Distributed. Si può anche prendere in considerazione Milvus completamente gestito su Zilliz Cloud.

Configurazione di Milvus Lite
pip install -U pymilvus

Si consiglia di utilizzare pymilvus. Dal momento che milvus-lite è incluso in pymilvus versione 2.4.2 o superiore, è possibile pip install con -U per forzare l'aggiornamento all'ultima versione e milvus-lite viene installato automaticamente.

Se si desidera installare esplicitamente il pacchetto milvus-lite, o se si è installata una versione precedente di milvus-lite e si desidera aggiornarla, si può usare pip install -U milvus-lite.

Connettersi a Milvus Lite
In pymilvus, specificando un nome di file locale come parametro uri di MilvusClient, si utilizzerà Milvus Lite.

from pymilvus import MilvusClient
client = MilvusClient("./milvus_demo.db")

Dopo aver eseguito il frammento di codice di cui sopra, nella cartella corrente verrà generato un file di database denominato milvus_demo.db.

NOTA: Si noti che la stessa API si applica anche a Milvus Standalone, Milvus Distributed e Zilliz Cloud, con l'unica differenza di sostituire il nome del file locale con l'endpoint del server remoto e le credenziali, ad esempioclient = MilvusClient(uri="http://localhost:19530", token="username:password").

Esempi
Di seguito è riportata una semplice demo che mostra come utilizzare Milvus Lite per la ricerca di testo. Esistono esempi più completi per l'utilizzo di Milvus Lite per la creazione di applicazioni come RAG, la ricerca di immagini e l'utilizzo di Milvus Lite in framework RAG popolari come LangChain e LlamaIndex!

from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("./milvus_demo.db")
client.create_collection(
    collection_name="demo_collection",
    dimension=384  # The vectors we will use in this demo has 384 dimensions
)

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
# For illustration, here we use fake vectors with random numbers (384 dimension).

vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
res = client.insert(
    collection_name="demo_collection",
    data=data
)

# This will exclude any text in "history" subject despite close to the query vector.
res = client.search(
    collection_name="demo_collection",
    data=[vectors[0]],
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)
print(res)

# a query that retrieves all entities matching filter expressions.
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print(res)

# delete
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'history'",
)
print(res)

Limiti
Quando si utilizza Milvus Lite, si noti che alcune funzioni non sono supportate. Le tabelle seguenti riassumono i limiti di utilizzo di Milvus Lite.

Raccolta
Metodo / Parametro	Supportato in Milvus Lite
create_collection()	Supporto con parametri limitati
collection_name	Y
dimension	Y
primary_field_name	Y
id_type	Y
vector_field_name	Y
metric_type	Y
auto_id	Y
schema	Y
index_params	Y
enable_dynamic_field	Y
num_shards	N
partition_key_field	N
num_partitions	N
consistency_level	N (supporta solo Strong; qualsiasi configurazione sarà trattata come Strong).
get_collection_stats()	Supporta l'ottenimento delle statistiche della raccolta.
collection_name	Y
timeout	Y
describe_collection()	num_shards, consistency_level e collection_id nella risposta non sono validi.
timeout	Y
has_collection()	Supporta la verifica dell'esistenza di una collezione.
collection_name	Y
timeout	Y
list_collections()	Supporta l'elenco di tutte le raccolte.
drop_collection()	Supporta l'eliminazione di una collezione.
collection_name	Y
timeout	Y
rinominare_raccolta()	La ridenominazione di un insieme non è supportata.
Campo e schema
Metodo / Parametro	Supportato in Milvus Lite
crea_schema()	Supporto con parametri limitati
auto_id	Y
enable_dynamic_field	Y
primary_field	Y
partition_key_field	N
add_field()	Supporto con parametri limitati
field_name	Y
datatype	Y
is_primary	Y
max_length	Y
element_type	Y
max_capacity	Y
dim	Y
is_partition_key	N
Inserisci e cerca
Metodo / Parametro	Supportato in Milvus Lite
ricerca()	Supporto con parametri limitati
collection_name	Y
data	Y
filter	Y
limit	Y
output_fields	Y
search_params	Y
timeout	Y
partition_names	N
anns_field	Y
query()	Supporto con parametri limitati
collection_name	Y
filter	Y
output_fields	Y
timeout	Y
ids	Y
partition_names	N
get()	Supporto con parametri limitati
collection_name	Y
ids	Y
output_fields	Y
timeout	Y
partition_names	N
cancellare()	Supporto con parametri limitati
collection_name	Y
ids	Y
timeout	Y
filter	Y
partition_name	N
insert()	Supporto con parametri limitati
collection_name	Y
data	Y
timeout	Y
partition_name	N
upsert()	Supporto con parametri limitati
collection_name	Y
data	Y
timeout	Y
partition_name	N
Carico e rilascio
Metodo / Parametro	Supportato in Milvus Lite
load_collection()	Y
collection_name	Y
timeout	Y
rilasciare_collezione()	Y
collection_name	Y
timeout	Y
get_load_state()	L'ottenimento dello stato di carico non è supportato.
refresh_load()	Il caricamento dei dati non caricati di una collezione caricata non è supportato.
close()	Y
Indice
Metodo / Parametro	Supportato in Milvus Lite
list_indexes()	È supportato l'elenco degli indici.
collection_name	Y
field_name	Y
creare_indice()	Supporta solo il tipo di indice FLAT.
index_params	Y
timeout	Y
drop_index()	È supportata l'eliminazione degli indici.
collection_name	Y
index_name	Y
timeout	Y
descrivere_indice()	È supportata la descrizione degli indici.
collection_name	Y
index_name	Y
timeout	Y
Tipi di indici vettoriali
Milvus Lite supporta solo il tipo di indice FLAT. Utilizza il tipo FLAT indipendentemente dal tipo di indice specificato nella raccolta.

Caratteristiche della ricerca
Milvus Lite supporta le funzioni Sparse Vector, Multi-vector e Hybrid Search.

Partizione
Milvus Lite non supporta le partizioni e i metodi relativi alle partizioni.

Utenti e ruoli
Milvus Lite non supporta utenti e ruoli e i relativi metodi.

Alias
Milvus Lite non supporta gli alias e i metodi relativi agli alias.

Migrazione dei dati da Milvus Lite
Tutti i dati memorizzati in Milvus Lite possono essere facilmente esportati e caricati in altri tipi di distribuzione Milvus, come Milvus Standalone su Docker, Milvus Distributed su K8s o Milvus completamente gestito su Zilliz Cloud.

Milvus Lite fornisce uno strumento a riga di comando che consente di scaricare i dati in un file json, che può essere importato in Milvus e Zilliz Cloud(il servizio cloud completamente gestito per Milvus). Il comando milvus-lite sarà installato insieme al pacchetto python milvus-lite.

Install
pip install -U "pymilvus[bulk_writer]"

milvus-lite dump -h

usage: milvus-lite dump [-h] [-d DB_FILE] [-c COLLECTION] [-p PATH]

optional arguments:
  -h, --help            show this help message and exit
  -d DB_FILE, --db-file DB_FILE
                        milvus lite db file
  -c COLLECTION, --collection COLLECTION
                        collection that need to be dumped
  -p PATH, --path PATH  dump file storage dir

L'esempio seguente esporta tutti i dati della collezione demo_collection memorizzati in ./milvus_demo.db (file del database Milvus Lite).

Per esportare i dati:

milvus-lite dump -d ./milvus_demo.db -c demo_collection -p ./data_dir
./milvus_demo.db: milvus lite db file
demo_collection: collection that need to be dumped
./data_dir : dump file storage dir

Con il file di dump, è possibile caricare i dati su Zilliz Cloud tramite Data Import o caricare i dati sui server Milvus tramite Bulk Insert.