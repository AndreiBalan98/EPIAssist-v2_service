import logging

from fastapi import APIRouter
from psycopg2.extras import Json

from models import Chunk, MessageRequest, MessageResponse
from dependencies import client, get_db_connection
from utils import cosine_similarity

router = APIRouter()
logger = logging.getLogger(__name__)

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
    "sau să consulte direct sursele oficiale. Nu încerca să ghicești sau să improvizezi. În acest caz NU "
    "adăuga secțiunea \"Surse:\" la final.\n\n"
    "3. ÎNTREBĂRI ÎN AFARA SUBIECTULUI: Dacă utilizatorul îți pune o întrebare care nu are legătură cu "
    "legislația medicală din România (ex: rețete culinare, sport, vreme, programare, etc.), răspunde cu "
    "umor delicat și prietenos, printr-o glumă scurtă și subtilă, apoi precizează politicos că rolul tău "
    "este să discuți doar despre legislația medicală românească. Invită utilizatorul să îți pună o "
    "întrebare pe această temă. În acest caz NU adăuga secțiunea \"Surse:\" la final.\n\n"
    "4. FORMAT MARKDOWN: Răspunsul tău trebuie să fie întotdeauna formatat în Markdown valid. Folosește:\n"
    "   - Titluri cu `##` sau `###` pentru a structura răspunsurile lungi\n"
    "   - **bold** pentru termenii cheie și informațiile importante\n"
    "   - *italic* pentru nuanțări, citate scurte sau termeni tehnici\n"
    "   - Liste cu marcaje folosind formatul `- a)`, `- b)`, `- c)` pentru enumerări neordonate\n"
    "   - Liste ordonate folosind formatul `- 1.`, `- 2.`, `- 3.` atunci când ordinea contează\n"
    "   - Fiecare element de listă trebuie să fie pe propria linie, începând cu `- ` urmat de litera) sau numărul.\n"
    "   - Paragrafe scurte separate prin linii goale\n\n"
    "5. CITĂRI - REGULĂ STRICTĂ: Atunci când răspunzi pe baza contextului, TREBUIE să incluzi la sfârșitul "
    "răspunsului o secțiune de surse, respectând EXACT următorul format:\n"
    "   - Lasă o linie goală după conținutul răspunsului\n"
    "   - Pe linia următoare scrie exact: `Surse:`\n"
    "   - Pe liniile următoare, listează URL-urile complete folosite, câte UNUL pe linie, fără marcaje, "
    "fără numerotare, fără text suplimentar (ex: `https://legislatie.just.ro/Public/.../ORDIN.../section/sub-section`)\n"
    "   - NU duplica niciodată un URL — fiecare sursă apare o singură dată\n"
    "   - Include DOAR URL-urile care susțin efectiv afirmațiile din răspuns; orice afirmație factuală din "
    "răspunsul tău trebuie să poată fi regăsită într-una din sursele listate\n"
    "   - NU pune secțiunea \"Surse:\" în interiorul răspunsului — doar la final\n"
    "   - NU adăuga text după lista de surse\n\n"
    "Exemplu de format final corect:\n"
    "```\n"
    "## Titlu răspuns\n\n"
    "Conținutul răspunsului cu **termeni cheie** și *nuanțări*.\n\n"
    "- a) Primul aspect\n"
    "- b) Al doilea aspect\n\n"
    "Surse:\n"
    "https://legislatie.just.ro/.../ORDIN-123/section/art-5\n"
    "https://legislatie.just.ro/.../LEGE-95/section/cap-2\n"
    "```\n\n"
    "6. LIMBAJ: Folosește un limbaj clar, evită jargonul juridic excesiv, dar păstrează acuratețea termenilor "
    "tehnici esențiali. Explică termenii complicați atunci când e cazul.\n\n"
    "Nu răspunde niciodată în altă limbă decât româna, indiferent de limba în care îți este adresată întrebarea."
)

ANSWER_USER_PROMPT = (
    "Întrebarea utilizatorului:\n{message}\n\n"
    "Context din baza de date legislativă (fiecare chunk începe cu URL-ul său pe prima linie):\n{context}\n\n"
    "Te rog să răspunzi conform regulilor stabilite."
)


def _log_conversation(
    user_prompt: str,
    enhanced_prompt: str,
    candidate_chunks: list[dict],
    context_chunks: list[dict],
    final_answer: str,
) -> None:
    """Persist a single conversation row. Failure is swallowed so the API call still returns successfully."""
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations "
                    "(user_prompt, enhanced_prompt, candidate_chunks, context_chunks, final_answer) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        user_prompt,
                        enhanced_prompt,
                        Json(candidate_chunks),
                        Json(context_chunks),
                        final_answer,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to log conversation")


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

    candidates: list[Chunk] = []
    for row_id, url, content, embedding_json in rows:
        sim = cosine_similarity(embedding, embedding_json)
        if sim >= SIMILARITY_THRESHOLD:
            candidates.append(Chunk(id=row_id, url=url, content=content, similarity=sim))

    candidates.sort(key=lambda x: x.similarity, reverse=True)

    # 3.5. build context
    context = ""
    included: list[Chunk] = []
    for chunk in candidates:
        entry = f"{chunk.url}\n{chunk.content}\n\n"
        if len(context) + len(entry) > CONTEXT_MAX_CHARS:
            break
        context += entry
        included.append(chunk)
    context = context.rstrip("\n")

    # 4. generate answer
    final_answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": ANSWER_USER_PROMPT.format(message=body.message, context=context)},
        ],
        temperature=0.5,
    ).choices[0].message.content

    # 5. log conversation
    candidate_log = [{"url": c.url, "similarity": c.similarity} for c in candidates]
    context_log = [{"url": c.url, "similarity": c.similarity} for c in included]
    _log_conversation(
        user_prompt=body.message,
        enhanced_prompt=enhanced_prompt,
        candidate_chunks=candidate_log,
        context_chunks=context_log,
        final_answer=final_answer,
    )

    return MessageResponse(message=final_answer)
