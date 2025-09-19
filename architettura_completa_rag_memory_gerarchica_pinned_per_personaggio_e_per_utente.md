# Architettura completa — RAG + Memory gerarchica + Pinned

Documento: architettura dettagliata e flow dell'algoritmo per un chatbot RAG che usa LlamaIndex + vectorstore (Chroma/FAISS), con memoria gerarchica, pinned per sessione/storico e pinned per interlocutore.

---

## 1. Obiettivo
Fornire un'architettura end-to-end per un chatbot di gioco (personaggio) che:

- conserva la chat history (multiruolo),
- mantiene pinned memory sia short-term (sessione) che long-term (storico),
- separa retrieval su due DB (lore statico e chat memory dinamica),
- permette controllo e tracciabilità: vedere i retrieved nodes, prompt finale, log del reasoning.

---

## 2. Diagramma ad alto livello (flow)

```mermaid
flowchart TD
  A[User Input] --> B[Preprocessor]
  B --> C{Routing}
  C -->|Identifica interlocutore/contesto| D[Load Context]
  D --> D1[Load Pinned: session + global + interlocutor]
  D --> D2[Load Short-term Chat Memory]
  D --> D3[Load Persona + System Instructions]

  B --> E[Parallel Retrieval]
  E --> E1[Lore Retriever (chroma_lore)]
  E --> E2[Chat Retriever (chroma_chat)]

  E1 --> F[Retrieved Lore Nodes]
  E2 --> G[Retrieved Chat Nodes]

  F & G & D1 & D2 & D3 --> H[Memory Manager]
  H --> H1[Decide Promoted Candidates]
  H --> H2[Update Pinned: session & historic]

  H & UserInput --> I[Prompt Assembler]
  I --> J[LLM (local HF model)]
  J --> K[Response Postprocessor]
  K --> L[Response to User]

  K --> M[Telemetry & Logs]
  M --> N[Debug UI / Traces]

  style E1 stroke:#2b6cb0,stroke-width:2px
  style E2 stroke:#2b6cb0,stroke-width:2px
  style H2 stroke:#dd6b20,stroke-width:2px
```

---

## 3. Componenti e responsabilità

### 3.1 Preprocessor
- Normalizza la query (tokenizzazione leggera, estrazione entità, rilevamento lingua).  
- Determina interlocutore e contesto (es. `speaker_id`, `session_id`, `quest_id`).

### 3.2 Routing / Load Context
- Carica pinned globali, pinned per interlocutore (se esiste), pinned sessione, short-term chat memory.
- Limita la dimensione (token budget per sezione).

### 3.3 Parallel Retrieval
- **Lore Retriever**: query su `chroma_lore` (statico), `top_k_lore` risultati.
- **Chat Retriever**: query su `chroma_chat` (dinamico), `top_k_chat` risultati.
- Entrambi ritornano `NodeWithScore` includendo metadati (speaker, timestamp, source_type).
- *Output visibile per debug*.

### 3.4 Memory Manager
- Riceve retrieved nodes + pinned attuali + user query.
- **Candidate detection**: identifica frasi/fatti da promuovere a pinned (regole + LLM scorer):
  - presenza ripetuta,
  - rilevanza alta rispetto alla query/missione,
  - esplicita richiesta di "ricorda"
- **Normalization**: genera statement atomici sintetici (1–3 frasi).
- **Promotion**: inserisce in pinned_session o pinned_historic (con metadata: source, confidence, created_at, score).
- **Compression**: se pinned_historic supera token_limit -> rank & summarize i meno usati.

### 3.5 Prompt Assembler
- Costruisce il prompt in sezioni chiaramente delimitate:
  1. System instructions (stile personaggio, rules)
  2. Pinned long-term (historic)
  3. Pinned session (short-term pinned)
  4. Retrieved chat nodes (dettagli recenti)
  5. Retrieved lore nodes (facts rilevanti)
  6. Chat short-term raw (ultimi turni, se spazio)
  7. User query
- Ogni sezione ha header espliciti e può includere metadati inline (es. `[from: Yennefer | t:2025-09-14]`).

### 3.6 LLM Engine
- Modello locale HF (GPU).  
- Puoi usare diversi modelli per *generation* e per *reranking* (cross-encoder).  
- Regole per il controllo: max_tokens_output, temperatura, safety filters.

### 3.7 Postprocessor & Action
- Eventuale parsing della risposta per estrarre azioni (es. aggiornare lo stato di una missione).
- Logging dettagliato dei retrieved nodes, prompt build, token usage, risposta.

### 3.8 Telemetry & Debug UI
- Salva: prompt finale, retrieved node ids + text + scores, LLM output, decisioni del Memory Manager.
- Fornisce UI per ispezionare ogni step: quali documenti sono stati retrieval, perché un fatto è stato pinned.

---

## 4. Dati / Schema (esempi JSON)

### 4.1 Node stored in Chroma (lore / chat)
```json
{
  "id": "uuid-1234",
  "text": "Geralt ha incontrato Yennefer a Novigrad.",
  "embedding": [0.12, -0.04, ...],
  "metadata": {
    "type": "lore",        
    "source": "compendium",
    "chapter": "Incontri",
    "language": "it"
  }
}
```

### 4.2 Pinned fact (historic)
```json
{
  "id": "pin-987",
  "text": "Geralt è stato incaricato da re Radovid di trovare un condottiero.",
  "source_refs": ["uuid-1234", "uuid-5678"],
  "created_at": "2025-09-15T22:15:00Z",
  "score": 0.92,
  "tags": ["mission", "royalty"],
  "scope": "global"   
}
```

### 4.3 Pinned per interlocutore
```json
{
  "user_id": "Yennefer",
  "pinned": [ {..}, {..} ]
}
```

---

## 5. Algoritmi chiave (pseudocodice)

### 5.1 Candidate detection & promotion (schematic)
```python
def detect_and_promote(retrieved_nodes, pinned, query):
    candidates = []
    for node in retrieved_nodes:
        if not semantically_duplicate(node.text, pinned):
            score = importance_score(node, query)
            if score > PROMOTE_THRESHOLD:
                fact = normalize_to_fact(node.text)
                candidates.append((fact, score, node.id))
    # rank and promote
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    for fact, score, ref in candidates[:MAX_PROMOTE_PER_TURN]:
        pinned.add(make_pin(fact, refs=[ref], score=score))
    return pinned
```

### 5.2 Compression
```python
def compress_pinned(pinned_list):
    if token_count(pinned_list) <= PINNED_TOKEN_LIMIT:
        return pinned_list
    ranked = rank_by_usage_and_score(pinned_list)
    keep = ranked[:TOP_K]
    rest = ranked[TOP_K:]
    summary = summarize_with_llm([r.text for r in rest])
    return keep + [ {"id":"summary-..","text":summary, "tags":["archived_summary"]} ]
```

---

## 6. Prompt template (esempio semplificato)

```jinja
SYSTEM: Sei Geralt di Rivia. Mantieni tono secco, diretto.

PINNED LONG-TERM:
{{pinned_historic}}

PINNED SESSION:
{{pinned_session}}

RETRIEVED CHAT (dettagli recenti):
{{retrieved_chat}}

RETRIEVED LORE (facts):
{{retrieved_lore}}

RECENT RAW TURNS:
{{short_term_raw}}

USER QUERY:
{{query}}

RESPONDI:
```

---

## 7. Decisioni di progetto & trade-offs
- **Separare DB (lore vs chat)** → migliore scalabilità e manutenzione.  
- **Pinned per interlocutore + global** → coerenza narrativa, ma più complessità di storage.  
- **Promote-on-retrieve** → robustezza della continuity ma aumento della dimensione del pinned: serve compressione.  
- **Tracciabilità** → salva i motivi (scores, refs) per ogni promozione.

---

## 8. Monitoring, privacy, sicurezza
- Logga token usage per sessione (costo performances).  
- Proteggi Pinned storico (contiene dati persistenti) con ACL e cifratura.  
- Rate-limit retrievers per evitare costi I/O e throttling.

---

## 9. Esempi di estensioni
- Multi-LLM: usare modello leggero per reranking e modello più grande per generation.  
- Tooling: UI di inspection (che mostri retrieved nodes e perché), editor per pinned facts.  
- Plugin di fact-checking per evitare promozione di informazioni contraddittorie.

---

## 10. Prossimi passi suggeriti
1. Implementare i retriever separati e il prompt assembler (PoC).  
2. Implementare il Memory Manager con rule-based promotion + LLM scorer.  
3. Aggiungere sistema di compressione periodica.  
4. Costruire UI per debug e ispezione.

---

*Fine del documento.*

