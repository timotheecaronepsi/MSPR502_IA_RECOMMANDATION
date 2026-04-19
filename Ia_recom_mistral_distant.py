import json
import re # re pour Regex
import os
import time
import urllib.request
import urllib.error

# Racine du projet: sert a construire des chemins absolus vers les fichiers txt et .env.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def charger_env(path):
    """Charge un fichier .env simple (KEY=VALUE) sans dependance externe."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#") or "=" not in ligne:
                continue

            key, value = ligne.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Ne remplace pas une variable deja definie dans l'environnement du shell.
            if key and key not in os.environ:
                os.environ[key] = value


# Charge automatiquement .env au lancement du script.
charger_env(os.path.join(BASE_DIR, ".env"))

# CONFIG HUGGING FACE DISTANT
# Modele principal (modifiable via .env).
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
# Timeout HTTP par tentative (secondes).
HF_TIMEOUT_SEC = float(os.getenv("HF_TIMEOUT_SEC", "12"))
# Nombre de tentatives par modele avant de passer au suivant.
HF_MAX_RETRIES = int(os.getenv("HF_MAX_RETRIES", "2"))
# Budget max de generation pour eviter la troncature du JSON.
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "520"))
# Fallbacks modeles: essayes dans l'ordre si le modele principal echoue.
HF_MODEL_FALLBACKS = [
    HF_MODEL_ID,  # Modele provenant du .env: tente en premier.
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]
# Metriques simples de run pour la soutenance (perf + qualite de sortie).
METRICS = {"calls": 0, "errors": 0, "json_ok": 0, "lat_ms_total": 0.0}


def metrics_snapshot():
    # Calcul des indicateurs globaux a partir du cumul des appels.
    calls = METRICS["calls"]
    avg_lat = round((METRICS["lat_ms_total"] / calls), 2) if calls else 0.0
    err_rate = round((METRICS["errors"] / calls) * 100, 2) if calls else 0.0
    json_rate = round((METRICS["json_ok"] / calls) * 100, 2) if calls else 0.0
    return {
        "calls": calls,
        "latence_moy_ms": avg_lat,
        "taux_erreur_pct": err_rate,
        "taux_json_valide_pct": json_rate,
    }

def call_hf_model(prompt, max_new_tokens, temperature, top_p, repetition_penalty):
    """Appelle Hugging Face Router (OpenAI-compatible) via HTTP."""
    # Debut mesure latence d'un appel de generation.
    METRICS["calls"] += 1
    t0 = time.perf_counter()

    if not HF_API_TOKEN:
        # Fin mesure immediatement si token absent.
        METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
        return json.dumps({"error": "HF_API_TOKEN manquant"}, ensure_ascii=False)

    # Endpoint unique Router compatible format OpenAI Chat Completions.
    endpoint = "https://router.huggingface.co/v1/chat/completions"
    tried_models = []
    last_error = ""

    # Remove duplicates while preserving order
    models_to_try = list(dict.fromkeys(HF_MODEL_FALLBACKS))

    # 1) boucle modeles, 2) boucle retries par modele.
    for model_id in models_to_try:
        tried_models.append(model_id)
        for _ in range(HF_MAX_RETRIES):
            # Prompting minimaliste pour obtenir strictement un objet JSON.
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "Tu es une IA qui repond strictement en JSON valide, sans texte additionnel."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "response_format": {"type": "json_object"}
            }

            # Construction de la requete HTTP POST vers HF Router.
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {HF_API_TOKEN}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            try:
                # Appel distant: peut lever timeout/HTTPError.
                with urllib.request.urlopen(req, timeout=HF_TIMEOUT_SEC) as response:
                    data = json.loads(response.read().decode("utf-8"))

                # Extraction du contenu texte standard OpenAI-compatible.
                if isinstance(data, dict):
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content", "")
                        if content:
                            # Succès: enregistre la latence et retourne brut.
                            METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
                            return content

                last_error = f"HF response invalid pour {model_id}"
                #Gestion des erreurs HTTP: 400 (modele non supporte), 403 (provider bloque), 401/403 (auth invalide).
            except urllib.error.HTTPError as e:
                # Capture du body erreur (utile pour debug).
                raw = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
                last_error = f"HTTP {e.code} pour {model_id}: {raw or str(e)}"
                # Modele non supporte -> inutile de retenter ce modele.
                if e.code == 400 and "model_not_supported" in raw:
                    break
                raw_lower = raw.lower()
                # Cas provider bloque (Cloudflare/Together): passe au modele suivant.
                if e.code == 403 and ("cloudflare" in raw_lower or "access denied" in raw_lower or "api.together.xyz" in raw_lower):
                    break
                # Auth invalide -> on stoppe tout de suite et remonte l'erreur.
                if e.code in (401, 403):
                    METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
                    return json.dumps({"error": f"HF request failed: {last_error}"}, ensure_ascii=False)
            except Exception as e:
                # Timeout DNS/reseau/autres erreurs runtime.
                last_error = f"{model_id}: {str(e)}"

    # Si aucun modele n'a reussi: retourne un JSON d'erreur detaille.
    METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
    return json.dumps(
        {
            "error": "HF request failed",
            "details": last_error,
            "models_testes": tried_models,
            "hint": "Definis HF_MODEL_ID dans .env avec un modele supporte par tes providers HF Router."
        },
        ensure_ascii=False
    )

# ==============================
### CONSTANTES
# ==============================

NIVEAUX_AUTORISES = {"facile", "normal", "intensif"}
OBJECTIFS_AUTORISES = {
    "perte_de_poids",
    "prise_de_masse"
}

# ==============================
### CHEMIN BASE ET FONCTIONS LIRES FICHIERS
# ==============================

def lire_fichier(path):
    # Lecture utilitaire centralisee (evite de repeter la logique d'ouverture).
    chemin = os.path.join(BASE_DIR, path)
    with open(chemin, "r", encoding="utf-8") as f:
        return f.read()

# ==============================
### CHARGEMENT DES DONNEES
# ==============================

def charger_exercices(path):
    # Parse une liste tuple-like depuis exercices.txt via regex.
    texte = lire_fichier(path)
    pattern = re.findall(
        r"\((\d+),\s*'([^']+)',\s*'([^']+)'(?:,\s*'([^']+)')?\)",
        texte
    )
    return [
        {
            "id": int(i),
            "nom": nom,
            "muscle": muscle,
            "niveau": niveau.lower().strip() if niveau else "normal"
        }
        for i, nom, muscle, niveau in pattern
    ]

def charger_materiels(path):
    # Convertit chaque ligne non vide en objet materiel avec id auto-incremente.
    lignes = lire_fichier(path).splitlines()
    return [
        {"id": idx + 1, "nom": l.strip()}
        for idx, l in enumerate(lignes)
        if l.strip()
    ]

def charger_liaisons(path):
    # Parse les couples (id_exercice, id_materiel) depuis le fichier de liaisons.
    texte = lire_fichier(path)
    pattern = re.findall(r"\((\d+),\s*(\d+)\)", texte)
    return [
        {"id_exercice": int(a), "id_materiel": int(b)}
        for a, b in pattern
    ]

# ==============================
### CONTEXTE
# ==============================

def construire_contexte(exercices):
    # Construit le contexte textualise envoye au LLM.
    contexte = "Liste des exercices disponibles :\n"
    for ex in exercices:
        contexte += f"- {ex['nom']} ({ex['muscle']}) — niveau : {ex['niveau']}\n"
    return contexte

# ==============================
### EXTRACTION JSON
# ==============================

def extract_json(text):    # Extrait le premier JSON valide trouvé dans la sortie de l'IA. Empêche l'IA de rajouter du texte
    # Parse robuste: retrouve le premier objet JSON meme si du texte parasite est present.
    stack = []
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if start is None:
                start = i
            stack.append("{")
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        return {"error": "invalid json", "raw": chunk}

    return {"error": "no json found", "raw": text}

# ==============================
### VALIDATION DES ENTREES
# ==============================

def valider_requete(data):     # Verif des données d'entréé 
    # Validation minimale du contrat d'entree API.
    if not data:
        return False, "Aucune donnée reçue"

    if "niveau" not in data:
        return False, "Le niveau est obligatoire"

    if "objectif" not in data:
        return False, "L'objectif est obligatoire"

    if "materiels" not in data:
        return False, "Les matériels sont obligatoires"

    niveau = data["niveau"].lower().strip()
    objectif = data["objectif"].lower().strip()

    if niveau not in NIVEAUX_AUTORISES:
        return False, "Niveau invalide"

    if objectif not in OBJECTIFS_AUTORISES:
        return False, "Objectif invalide"

    if not isinstance(data["materiels"], list):
        return False, "Les matériels doivent être une liste"

    return True, None

# ==============================
### MOTEUR DE GENERATION
# ==============================
 

def generer_programme(data):
    # Etape 1: validation des champs obligatoires.
    valide, erreur = valider_requete(data)
    if not valide:
        return {"error": erreur}

    niveau = data["niveau"]
    objectif = data["objectif"]
    materiels_user = data["materiels"]

    # Etape 2: chargement du catalogue d'exercices/materiels/liaisons.
    exercices = charger_exercices("exercices.txt")
    materiels = charger_materiels("materiels.txt")
    liaisons = charger_liaisons("exercice_materiel.txt")

    # Mapping des materiels utilisateur vers leurs IDs connus.
    ids_materiels_user = {
        m["id"] for m in materiels
        if m["nom"].lower() in [mu.lower() for mu in materiels_user]
    }

    # IDs exercices compatibles avec les materiels user.
    ids_exercices_autorises = {
        l["id_exercice"]
        for l in liaisons
        if l["id_materiel"] in ids_materiels_user
    }

    if niveau == "facile":
        nombre_exercices = 3
    elif niveau == "normal":
        nombre_exercices = 4
    else:
        nombre_exercices = 5

    # Etape 3: priorite aux exercices compatibles materiel + niveau.
    exercices_avec_materiel = [
        ex for ex in exercices
        if ex["niveau"] == niveau and ex["id"] in ids_exercices_autorises
    ]

    # Etape 4: fallback sur exercices sans materiel si liste trop courte.
    ids_exercices_avec_materiel = {l["id_exercice"] for l in liaisons}

    exercices_sans_materiel = [
        ex for ex in exercices
        if ex["niveau"] == niveau and ex["id"] not in ids_exercices_avec_materiel
    ]

    # On part d'abord des exercices avec materiel.
    exercices = exercices_avec_materiel.copy()

    # Fallback sans matériel si nécessaire
    if len(exercices) < nombre_exercices:
        manque = nombre_exercices - len(exercices)
        exercices.extend(exercices_sans_materiel[:manque])

    # Liste de detection cardio (lowercase pour comparaison robuste).
    MOTS_CLES_CARDIO = ["course / sprint", "mountain climbers", "stepper", "sauts corde", "air bike", "rameur", "elliptique", "frappe", "marche sur place", "jumping jacks", "burpees", "jump squats", "fentes sautées", "high knees"]

    # Regle metier: forcer au moins un exercice cardio si objectif perte de poids.
    if objectif == "perte_de_poids":
        exercices_cardio = [
            ex for ex in exercices
            if any(mot in ex["nom"].lower() for mot in MOTS_CLES_CARDIO)
        ]

        if exercices_cardio:
            cardio = exercices_cardio[0]

            if cardio not in exercices:
                exercices.pop()
                exercices.insert(0, cardio)

    contexte = construire_contexte(exercices)

    # Prompt strict pour limiter la derivation du modele et obtenir un JSON stable.
    prompt = f"""{contexte}
Tu es un coach fitness IA. Génère un programme JSON.
Objectif: {objectif}, Niveau: {niveau}
EXACTEMENT {nombre_exercices} exercices de la liste.
JSON STRICT:
{{
"niveau": "{niveau}",
"objectif": "{objectif}",
"programme": [{{"exercice": "Nom (Muscle)", "series": 0, "repetitions": 0, "temps_de_repos": 0}}],
"progression": {{"semaine": [1, 2, 3, 4]}}
}}
Interdit: aucun autre champ, aucune planification par jour, aucune explication.
"""

    # Etape 5: appel du LLM distant.
    response = call_hf_model(
        prompt,
        max_new_tokens=HF_MAX_TOKENS,
        temperature=0.2,  # Faible température: moins de créativité donc respecte plus le JSON stricte.
        top_p=0.9,  # Garde le noyau des tokens les plus probables jusqu'à 90% de probabilité cumulée --> moins de mots complexes.
        repetition_penalty=1.1  # Réduit les répétitions dans la réponse.
    )

    # Etape 6: parsing JSON de la sortie modele.
    parsed = extract_json(response)

    # Retry unique si JSON tronque/introuvable.
    if parsed.get("error") == "no json found":
        response = call_hf_model(
            prompt + "\nRappel final: renvoie uniquement l'objet JSON ci-dessus, sans details supplementaires.",
            max_new_tokens=HF_MAX_TOKENS + 220,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )
        parsed = extract_json(response)

    # Mise a jour des metriques de qualite.
    if "error" in parsed:
        METRICS["errors"] += 1
    else:
        METRICS["json_ok"] += 1
    return parsed

# ==============================
### TEST LOCAL (SIMULATION APPEL API)
# ==============================

if __name__ == "__main__":
    # Bloc de test manuel local pour valider rapidement la generation.
    requete_test = {
        "niveau": "intensif",
        "objectif": "perte_de_poids",
        "materiels": ["Banc d'entraînement","Haltères","Barres"]
    }

    # Affiche resultat metier + metriques runtime.
    resultat = generer_programme(requete_test)
    print(json.dumps(resultat, indent=2, ensure_ascii=False))
    
    # ==============================
    ### Soutenance: decommenter la ligne ci-dessous pour afficher les metrics.
    # ==============================
    # print(json.dumps({"metrics": metrics_snapshot()}, indent=2, ensure_ascii=False))