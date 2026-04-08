from ctransformers import AutoModelForCausalLM # Modèle LLM local (Mistral) via ctransformers
import json
import re # re pour Regex
import os

# ==============================
### CONFIG DU MODELE
# ==============================

MODEL_PATH = "models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

model = AutoModelForCausalLM.from_pretrained(   # Chargement du modèle
    MODEL_PATH,
    model_type="mistral",
    gpu_layers=0,
    threads=6,
    context_length=2048
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def lire_fichier(path):
    chemin = os.path.join(BASE_DIR, path)
    with open(chemin, "r", encoding="utf-8") as f:
        return f.read()

# ==============================
### CHARGEMENT DES DONNEES
# ==============================

def charger_exercices(path):
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
    lignes = lire_fichier(path).splitlines()
    return [
        {"id": idx + 1, "nom": l.strip()}
        for idx, l in enumerate(lignes)
        if l.strip()
    ]

def charger_liaisons(path):
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
    contexte = "Liste des exercices disponibles :\n"
    for ex in exercices:
        contexte += f"- {ex['nom']} ({ex['muscle']}) — niveau : {ex['niveau']}\n"
    return contexte

# ==============================
### EXTRACTION JSON
# ==============================

def extract_json(text):    # Extrait le premier JSON valide trouvé dans la sortie de l'IA. Empêche l'IA de rajouter du texte
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
    valide, erreur = valider_requete(data)
    if not valide:
        return {"error": erreur}

    niveau = data["niveau"]
    objectif = data["objectif"]
    materiels_user = data["materiels"]

    # Chargement des données
    exercices = charger_exercices("exercices.txt")
    materiels = charger_materiels("materiels.txt")
    liaisons = charger_liaisons("exercice_materiel.txt")

    # IDs des matériels utilisateur
    ids_materiels_user = {
        m["id"] for m in materiels
        if m["nom"].lower() in [mu.lower() for mu in materiels_user]
    }

    # IDs des exercices autorisés selon le matériel
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

    # Exercices AVEC matériel utilisateur
    exercices_avec_materiel = [
        ex for ex in exercices
        if ex["niveau"] == niveau and ex["id"] in ids_exercices_autorises
    ]

    # Exercices SANS matériel (pas de liaison)
    ids_exercices_avec_materiel = {l["id_exercice"] for l in liaisons}

    exercices_sans_materiel = [
        ex for ex in exercices
        if ex["niveau"] == niveau and ex["id"] not in ids_exercices_avec_materiel
    ]

    # Priorité au matériel
    exercices = exercices_avec_materiel.copy()

    # Fallback sans matériel si nécessaire
    if len(exercices) < nombre_exercices:
        manque = nombre_exercices - len(exercices)
        exercices.extend(exercices_sans_materiel[:manque])

    # Sécurité finale
    if len(exercices) < nombre_exercices:
        return {"error": "Pas assez d'exercices compatibles avec le niveau et le matériel"}


    contexte = construire_contexte(exercices)

    prompt = f"""
{contexte}

Tu es une IA experte en sport.

Objectif utilisateur : {objectif}
Niveau : {niveau}

Créer un programme d'entraînement cohérent.

Réponds UNIQUEMENT avec ce JSON strict :

{{
  "niveau": "{niveau}",
  "objectif": "{objectif}",
  "programme": [
      {{
        "exercice": "",
        "series": 0,
        "repetitions": 0,
        "temps_de_repos": 0
      }}
  ],
  "progression": {{
    "semaine": [1, 2, 3, 4]
  }}
}}

RÈGLES OBLIGATOIRES :
- Utilise uniquement les exercices listés
- programme contient EXACTEMENT {nombre_exercices} exercices
- Aucun texte hors JSON
- S'il n'y a pas assez d'exercices disponibles, tu DOIS quand même en générer {nombre_exercices}
  en réutilisant uniquement ceux fournis
- Tu n'as PAS le droit de modifier la structure du JSON, ni d'ajouter des champs, ni de faire du texte libre
"""

    response = model(
        prompt,
        max_new_tokens=800,
        temperature=0.2,   # paramètres de créativité = plus de respect des règles
        top_p=0.9,       # contrôle la diversité des mots que l’IA peut choisir plus bas = plus rigide sans créativité
        repetition_penalty=1.1    # pénalité pour éviter les répétitions dans la génération
    )

    return extract_json(response)

# ==============================
### TEST LOCAL (SIMULATION APPEL API)
# ==============================

if __name__ == "__main__":
    requete_test = {
        "niveau": "intensif",
        "objectif": "perte_de_poids",
        "materiels": ["kettlebells", "haltères"]
    }

    resultat = generer_programme(requete_test)
    print(json.dumps(resultat, indent=2, ensure_ascii=False))