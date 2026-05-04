from fastapi import APIRouter
from models import Chunk, MessageRequest, MessageResponse
from dependencies import client, get_db_connection
from utils import cosine_similarity

router = APIRouter()

SIMILARITY_THRESHOLD = 0.5
CONTEXT_MAX_CHARS = 25000


QUERY_EXPANSION_PROMPT = (
    "Ești un expert în legislația medicală din România. Sarcina ta este să extinzi o întrebare "
    "a utilizatorului pentru a îmbunătăți căutarea semantică într-o bază de date cu documente legislative.\n\n"
    "Pe baza întrebării utilizatorului, generează un text îmbogățit care include:\n"
    "- Întrebarea originală reformulată clar\n"
    "- Cuvinte cheie și termeni juridici/medicali relevanți (ex: lege, ordin, hotărâre, normă, regulament, CNAS, CASMB, Ministerul Sănătății)\n"
    "- Sinonime și formulări alternative pentru termenii principali\n"
    "- Posibile interpretări sau aspecte conexe ale întrebării\n"
    "- Domenii legislative relevante (ex: asigurări de sănătate, malpraxis, drepturile pacientului, "
    "exercitarea profesiei medicale, medicamente, dispozitive medicale, etc.)\n\n"
    "Răspunde DOAR cu textul îmbogățit, fără explicații suplimentare, fără liste numerotate și fără titluri. "
    "Textul trebuie să fie un paragraf compact, optimizat pentru căutare semantică în limba română.\n\n"
    "Întrebarea utilizatorului: {message}"
)

ANSWER_SYSTEM_PROMPT = (
    "Ești EPIAssist, un asistent virtual prietenos și profesionist, specializat exclusiv în legislația "
    "medicală din România. Comunici întotdeauna în limba română, pe un ton cald, accesibil, dar riguros "
    "din punct de vedere juridic.\n\n"
    "REGULI DE COMPORTAMENT:\n\n"
    "1. RĂSPUNSURI BAZATE PE CONTEXT: Răspunzi la întrebări despre legislația medicală românească folosind "
    "STRICT informațiile din contextul furnizat. Nu inventezi informații și nu folosești cunoștințe externe "
    "contextului primit.\n\n"
    "2. CONTEXT INSUFICIENT: Dacă informațiile din context nu sunt suficiente pentru a răspunde complet și "
    "corect la întrebare, recunoaște acest lucru cu eleganță. Spune politicos că nu ai informațiile necesare "
    "în baza de date pentru a oferi un răspuns sigur și sugerează utilizatorului să reformuleze întrebarea "
    "sau să consulte direct sursele oficiale. Nu încerca să ghicești sau să improvizezi.\n\n"
    "3. ÎNTREBĂRI ÎN AFARA SUBIECTULUI: Dacă utilizatorul îți pune o întrebare care nu are legătură cu "
    "legislația medicală din România (ex: rețete culinare, sport, vreme, programare, etc.), răspunde cu "
    "umor delicat și prietenos, printr-o glumă scurtă și subtilă, apoi precizează politicos că rolul tău "
    "este să discuți doar despre legislația medicală românească. Invită utilizatorul să îți pună o "
    "întrebare pe această temă.\n\n"
    "4. CITĂRI: Când răspunzi pe baza contextului, menționează sursele relevante (URL-urile sau actele "
    "normative) atunci când sunt disponibile, pentru ca utilizatorul să poată verifica informația.\n\n"
    "5. LIMBAJ: Folosește un limbaj clar, evită jargonul juridic excesiv, dar păstrează acuratețea termenilor "
    "tehnici esențiali. Explică termenii complicați atunci când e cazul.\n\n"
    "6. STRUCTURĂ: Pentru răspunsuri lungi, folosește paragrafe scurte și, dacă ajută claritatea, liste cu "
    "marcaje. Pentru întrebări simple, răspunde concis.\n\n"
    "Nu răspunde niciodată în altă limbă decât româna, indiferent de limba în care îți este adresată întrebarea."
)

ANSWER_USER_PROMPT = (
    "Întrebarea utilizatorului:\n{message}\n\n"
    "Context din baza de date legislativă:\n{context}\n\n"
    "Te rog să răspunzi conform regulilor stabilite."
)


@router.post("/ai", response_model=MessageResponse)
def ai(body: MessageRequest):

    # 1. enhance prompt
    enhanced_prompt = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(message=body.message)}],
        temperature=0.3,
    ).choices[0].message.content

    # 2. convert to embedding
    embedding = client.embeddings.create(
        input=enhanced_prompt,
        model="text-embedding-3-large"
    ).data[0].embedding

    # 3. retrieve top similar chunks
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, url, content, embedding FROM chunks WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row_id, url, content, embedding_json in rows:
        chunk_embedding = embedding_json
        sim = cosine_similarity(embedding, chunk_embedding)
        if sim >= SIMILARITY_THRESHOLD:
            results.append(Chunk(id=row_id, url=url, content=content, similarity=sim))

    results.sort(key=lambda x: x.similarity, reverse=True)

    # 3.5. build context
    context = ""
    for chunk in results:
        entry = f"{chunk.url}\n{chunk.content}\n\n"
        if len(context) + len(entry) > CONTEXT_MAX_CHARS:
            break
        context += entry
    context = context.rstrip("\n")

    # 4. generate answer
    message = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": ANSWER_USER_PROMPT.format(message=body.message, context=context)},
        ],
        temperature=0.5,
    ).choices[0].message.content

    return MessageResponse(message=message)
