import json
import os
import time
import unicodedata
import urllib.error
import urllib.request


# Racine du projet: sert a construire des chemins absolus vers .env et autres fichiers.
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
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_TIMEOUT_SEC = max(45.0, float(os.getenv("HF_TIMEOUT_SEC", "45")))
HF_MAX_RETRIES = max(3, int(os.getenv("HF_MAX_RETRIES", "3")))
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "1600"))
HF_MODEL_FALLBACKS = [
	HF_MODEL_ID,
	"meta-llama/Llama-3.1-8B-Instruct",
	"meta-llama/Meta-Llama-3-8B-Instruct",
]

INGREDIENTS_FILE = os.path.join(BASE_DIR, "final_ingredients_list.json")
RESTRICTIONS_EQ_FILE = os.path.join(BASE_DIR, "restrictions_equivalences.json")

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
	METRICS["calls"] += 1
	t0 = time.perf_counter()

	if not HF_API_TOKEN:
		METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
		return json.dumps({"error": "HF_API_TOKEN manquant"}, ensure_ascii=False)

	endpoint = "https://router.huggingface.co/v1/chat/completions"
	tried_models = []
	last_error = ""
	models_to_try = list(dict.fromkeys(HF_MODEL_FALLBACKS))

	for model_id in models_to_try:
		tried_models.append(model_id)
		for attempt in range(HF_MAX_RETRIES):
			payload = {
				"model": model_id,
				"messages": [
					{
						"role": "system",
						"content": "Tu es une IA qui repond strictement en JSON valide, sans texte additionnel.",
					},
					{"role": "user", "content": prompt},
				],
				"max_tokens": max_new_tokens,
				"temperature": temperature,
				"top_p": top_p,
				"response_format": {"type": "json_object"},
			}

			req = urllib.request.Request(
				endpoint,
				data=json.dumps(payload).encode("utf-8"),
				headers={
					"Authorization": f"Bearer {HF_API_TOKEN}",
					"Content-Type": "application/json",
				},
				method="POST",
			)

			try:
				with urllib.request.urlopen(req, timeout=HF_TIMEOUT_SEC) as response:
					data = json.loads(response.read().decode("utf-8"))

				if isinstance(data, dict):
					choices = data.get("choices", [])
					if choices and "message" in choices[0]:
						content = choices[0]["message"].get("content", "")
						if content:
							METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
							return content

				last_error = f"HF response invalid pour {model_id}"
			except urllib.error.HTTPError as e:
				raw = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
				last_error = f"HTTP {e.code} pour {model_id}: {raw or str(e)}"

				if e.code == 400 and "model_not_supported" in raw:
					break

				raw_lower = raw.lower()
				if e.code == 403 and (
					"cloudflare" in raw_lower
					or "access denied" in raw_lower
					or "api.together.xyz" in raw_lower
				):
					break

				if e.code in (401, 403):
					METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
					return json.dumps({"error": f"HF request failed: {last_error}"}, ensure_ascii=False)

				# 429/5xx: erreurs transitoires, on retente avec petit backoff.
				if e.code == 429 or 500 <= e.code <= 599:
					time.sleep(min(4.0, 0.8 * (attempt + 1)))
			except urllib.error.URLError as e:
				last_error = f"{model_id}: {str(e)}"
				time.sleep(min(4.0, 0.8 * (attempt + 1)))
			except Exception as e:
				last_error = f"{model_id}: {str(e)}"
				if "timed out" in last_error.lower() or "timeout" in last_error.lower():
					time.sleep(min(4.0, 0.8 * (attempt + 1)))

	METRICS["lat_ms_total"] += (time.perf_counter() - t0) * 1000
	return json.dumps(
		{
			"error": "HF request failed",
			"details": last_error,
			"models_testes": tried_models,
			"hint": "Definis HF_MODEL_ID dans .env avec un modele supporte et augmente HF_TIMEOUT_SEC (>=45) si besoin.",
		},
		ensure_ascii=False,
	)


# ==============================
### CONSTANTES
# ==============================

OBJECTIFS_ALIMENTAIRES_AUTORISES = {
	"perte_de_poids",
	"prise_de_masse"
}

BUDGETS_AUTORISES = {"economique", "standard", "premium"}
BUDGET_LEVEL_TO_LABEL = {1: "economique", 2: "standard", 3: "premium"}
BUDGET_LABEL_TO_LEVEL = {v: k for k, v in BUDGET_LEVEL_TO_LABEL.items()}


def normalize_text(value):
	"""Normalise un texte pour comparaisons robustes (accents/casse/espaces)."""
	text = str(value or "").strip().lower()
	text = unicodedata.normalize("NFKD", text)
	text = "".join(ch for ch in text if not unicodedata.combining(ch))
	return " ".join(text.split())


def load_restriction_equivalences(path):
	"""Charge les equivalences FR/EN depuis un JSON externe."""
	fallback = {
		"arachides": ["arachides", "arachide", "peanut", "peanuts"],
		"porc": ["porc", "pork", "ham", "bacon"],
		"poulet": ["poulet", "chicken"],
		"riz": ["riz", "rice"],
		"brocoli": ["brocoli", "broccoli"],
		"pain": ["pain", "bread", "white bread"],
	}

	if not os.path.exists(path):
		return fallback

	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if not isinstance(data, dict):
			return fallback

		normalized = {}
		for key, values in data.items():
			nk = normalize_text(key)
			if not nk:
				continue
			if isinstance(values, list):
				normalized[nk] = [normalize_text(v) for v in values if normalize_text(v)]

		return normalized or fallback
	except Exception:
		return fallback


def build_restriction_alias_index(equivalences):
	"""Index inverse: alias -> cles de restrictions (pour expansion auto)."""
	index = {}
	for key, values in equivalences.items():
		aliases = [key, *values]
		for alias in aliases:
			a = normalize_text(alias)
			if not a:
				continue
			index.setdefault(a, set()).add(key)
	return index


RESTRICTION_EQUIVALENCES = load_restriction_equivalences(RESTRICTIONS_EQ_FILE)
RESTRICTION_ALIAS_INDEX = build_restriction_alias_index(RESTRICTION_EQUIVALENCES)


def build_restriction_terms(restrictions, ingredient_names):
	"""Genere automatiquement une liste de termes interdits (FR/EN + derives)."""
	terms = []
	seen = set()
	ingredients_norm = [normalize_text(x) for x in ingredient_names]

	def add_term(term):
		t = normalize_text(term)
		if t and t not in seen:
			seen.add(t)
			terms.append(t)

	for raw in restrictions:
		base = normalize_text(raw)
		if not base:
			continue

		# Toujours conserver la restriction utilisateur telle quelle.
		add_term(base)

		# Cas 1: cle directe dans les equivalences.
		if base in RESTRICTION_EQUIVALENCES:
			add_term(base)
			for value in RESTRICTION_EQUIVALENCES[base]:
				add_term(value)

		# Cas 2: alias connu (ex: "bread" -> groupe "pain").
		for group_key in RESTRICTION_ALIAS_INDEX.get(base, set()):
			add_term(group_key)
			for value in RESTRICTION_EQUIVALENCES.get(group_key, []):
				add_term(value)

		# Cas 3: derivation auto depuis la base ingredients (sans dictionnaire manuel).
		for ing in ingredients_norm:
			if base in ing:
				add_term(ing)

		# Heuristique token: permet de capter des formes composees proches.
		for token in base.split():
			if len(token) < 4:
				continue
			for ing in ingredients_norm:
				if token in ing:
					add_term(ing)

	return terms


def find_forbidden_ingredients(plan_obj, restriction_terms):
	"""Retourne les ingredients detectes qui violent les restrictions."""
	if not isinstance(plan_obj, dict):
		return []

	violations = []
	plan_repas = plan_obj.get("plan_repas", [])
	for jour in plan_repas:
		for repas in jour.get("repas", []):
			for ingredient in repas.get("ingredients_principaux", []):
				ing = normalize_text(ingredient)
				for term in restriction_terms:
					if term and term in ing:
						violations.append(ingredient)
						break

	for item in plan_obj.get("liste_courses", []):
		ingredient = item.get("ingredient", "") if isinstance(item, dict) else ""
		ing = normalize_text(ingredient)
		for term in restriction_terms:
			if term and term in ing:
				violations.append(ingredient)
				break

	seen = set()
	return [x for x in violations if not (normalize_text(x) in seen or seen.add(normalize_text(x)))]


def filter_ingredients_by_restrictions(ingredients, restriction_terms):
	"""Retire de la base autorisee les ingredients qui matchent une restriction."""
	filtered = []
	for item in ingredients:
		nom = item.get("nom", "")
		norm_nom = normalize_text(nom)
		blocked = any(term and term in norm_nom for term in restriction_terms)
		if not blocked:
			filtered.append(item)
	return filtered


def normaliser_budget(budget_brut):
	"""Accepte 1/2/3 ou economique/standard/premium et retourne (niveau, libelle)."""
	if budget_brut is None:
		return 2, "standard"

	if isinstance(budget_brut, (int, float)):
		niveau = int(budget_brut)
		if niveau in BUDGET_LEVEL_TO_LABEL:
			return niveau, BUDGET_LEVEL_TO_LABEL[niveau]
		raise ValueError("Budget invalide")

	b = str(budget_brut).strip().lower()
	if b.isdigit():
		niveau = int(b)
		if niveau in BUDGET_LEVEL_TO_LABEL:
			return niveau, BUDGET_LEVEL_TO_LABEL[niveau]
		raise ValueError("Budget invalide")

	if b in BUDGET_LABEL_TO_LEVEL:
		return BUDGET_LABEL_TO_LEVEL[b], b

	raise ValueError("Budget invalide")


def charger_ingredients_par_budget(niveau_budget, max_items=120):
	"""Lit le fichier d'ingredients (format JSONL/NDJSON) et filtre par budget <= niveau."""
	if not os.path.exists(INGREDIENTS_FILE):
		return []

	ingredients = []
	deja_vus = set()

	with open(INGREDIENTS_FILE, "r", encoding="utf-8") as f:
		for ligne in f:
			ligne = ligne.strip()
			if not ligne:
				continue

			try:
				item = json.loads(ligne)
			except json.JSONDecodeError:
				continue

			budget_item = item.get("budget")
			nom_item = item.get("nom")
			if budget_item is None or not nom_item:
				continue

			try:
				budget_item = int(budget_item)
			except (TypeError, ValueError):
				continue

			# Un budget premium peut utiliser eco + standard + premium.
			if budget_item > niveau_budget:
				continue

			cle = nom_item.strip().lower()
			if cle in deja_vus:
				continue

			deja_vus.add(cle)
			ingredients.append({
				"nom": nom_item,
				"budget": budget_item,
				"calories": item.get("calories"),
				"proteines": item.get("proteines"),
				"glucides": item.get("glucides"),
				"lipides": item.get("lipides"),
			})

			if len(ingredients) >= max_items:
				break

	return ingredients


# ==============================
### EXTRACTION JSON
# ==============================

def extract_json(text):
	# Parse robuste: retrouve le premier objet JSON meme si du texte parasite est present.
	if not text:
		return {"error": "no json found", "raw": text}

	cleaned = text.strip()

	# Certains modeles encapsulent la sortie dans des fences markdown.
	if cleaned.startswith("```"):
		lines = cleaned.splitlines()
		if len(lines) >= 3 and lines[-1].strip().startswith("```"):
			cleaned = "\n".join(lines[1:-1]).strip()

	decoder = json.JSONDecoder()
	first_brace = cleaned.find("{")
	if first_brace == -1:
		return {"error": "no json found", "raw": text}

	for i in range(first_brace, len(cleaned)):
		if cleaned[i] != "{":
			continue
		try:
			obj, _ = decoder.raw_decode(cleaned[i:])
			if isinstance(obj, dict):
				return obj
		except json.JSONDecodeError:
			continue

	# Si un JSON semble avoir commence mais reste tronque, remonte une erreur explicite.
	return {"error": "json incomplete or invalid", "raw": cleaned}


def plan_repas_valide(obj, nb_jours, repas_par_jour):
	"""Valide la structure minimale attendue pour un plan repas complet."""
	if not isinstance(obj, dict):
		return False

	required_keys = {"objectif_alimentaire", "nb_jours", "repas_par_jour", "plan_repas"}
	if not required_keys.issubset(set(obj.keys())):
		return False

	if not isinstance(obj.get("plan_repas"), list):
		return False

	if len(obj["plan_repas"]) != nb_jours:
		return False

	for jour in obj["plan_repas"]:
		if not isinstance(jour, dict):
			return False
		repas = jour.get("repas")
		if not isinstance(repas, list):
			return False
		if len(repas) != repas_par_jour:
			return False

	return True


# ==============================
### VALIDATION DES ENTREES
# ==============================

def valider_requete(data):
	if not data:
		return False, "Aucune donnée reçue"

	if "objectif_alimentaire" not in data:
		return False, "L'objectif alimentaire est obligatoire"

	if "nb_jours" not in data:
		return False, "Le nombre de jours est obligatoire"

	if "repas_par_jour" not in data:
		return False, "Le nombre de repas par jour est obligatoire"

	objectif_alimentaire = data["objectif_alimentaire"].lower().strip()
	if objectif_alimentaire not in OBJECTIFS_ALIMENTAIRES_AUTORISES:
		return False, "Objectif alimentaire invalide"

	try:
		nb_jours = int(data["nb_jours"])
		repas_par_jour = int(data["repas_par_jour"])
	except (TypeError, ValueError):
		return False, "nb_jours et repas_par_jour doivent etre numeriques"

	if nb_jours < 1 or nb_jours > 31:
		return False, "nb_jours doit etre compris entre 1 et 31"

	if repas_par_jour < 1 or repas_par_jour > 6:
		return False, "repas_par_jour doit etre compris entre 1 et 6"

	try:
		normaliser_budget(data.get("budget"))
	except ValueError:
		return False, "Budget invalide (1=economique, 2=standard, 3=premium)"

	# Format strict: une seule liste de restrictions est acceptee.
	if "restrictions" not in data:
		return False, "restrictions est obligatoire"

	if not isinstance(data["restrictions"], list):
		return False, "restrictions doit etre une liste"

	return True, None


# ==============================
### MOTEUR DE GENERATION
# ==============================

def generer_recommandations_plats(data):
	valide, erreur = valider_requete(data)
	if not valide:
		return {"error": erreur}

	objectif_alimentaire = data["objectif_alimentaire"].lower().strip()
	nb_jours = int(data["nb_jours"])
	repas_par_jour = int(data["repas_par_jour"])
	niveau_budget, budget = normaliser_budget(data.get("budget"))
	restrictions = data.get("restrictions", [])

	# Dedoublonnage simple en preservant l'ordre.
	seen = set()
	restrictions = [x for x in restrictions if not (str(x).strip().lower() in seen or seen.add(str(x).strip().lower()))]

	ingredients_autorises = charger_ingredients_par_budget(niveau_budget)
	restriction_terms = build_restriction_terms(restrictions, [x.get("nom", "") for x in ingredients_autorises])
	ingredients_autorises = filter_ingredients_by_restrictions(ingredients_autorises, restriction_terms)
	if not ingredients_autorises:
		return {
			"error": "aucun_ingredient_autorise",
			"details": "Toutes les options d'ingredients sont bloquees par les restrictions.",
		}

	ingredients_autorises_noms = [x["nom"] for x in ingredients_autorises]

	prompt = f"""
Tu es une IA nutritionniste. Genere un plan de repas en JSON strict.

Contrainte utilisateur:
- objectif_alimentaire: {objectif_alimentaire}
- nb_jours: {nb_jours}
- repas_par_jour: {repas_par_jour}
- budget: {budget}
- restrictions: {json.dumps(restrictions, ensure_ascii=False)}
- restrictions_normalisees_fr_en: {json.dumps(restriction_terms, ensure_ascii=False)}
- ingredients_disponibles_budget: {json.dumps(ingredients_autorises_noms, ensure_ascii=False)}

JSON STRICT:
{{
  "objectif_alimentaire": "{objectif_alimentaire}",
  "nb_jours": {nb_jours},
  "repas_par_jour": {repas_par_jour},
  "plan_repas": [
	{{
	  "jour": 1,
	  "repas": [
		{{
		  "type": "repas_1",
		  "plat": "Nom du plat",
		  "ingredients_principaux": ["ingredient"],
		  "calories_estimees": 0,
		  "temps_preparation_min": 0
		}}
	  ]
	}}
  ],
  "liste_courses": [
	{{"ingredient": "", "quantite": "", "categorie": ""}}
  ]
}}

Regles strictes:
- Respecte exactement nb_jours et repas_par_jour.
- Evite strictement tous les aliments presentes dans restrictions et restrictions_normalisees_fr_en.
- Si un ingredient contient un terme interdit (ex: chicken pour poulet, bread pour pain), il est interdit.
- N'utilise que les ingredients presents dans ingredients_disponibles_budget.
- Aucun texte hors JSON.
""".strip()

	response = call_hf_model(
		prompt,
		max_new_tokens=HF_MAX_TOKENS,
		temperature=0.2,
		top_p=0.9,
		repetition_penalty=1.1,
	)

	parsed = extract_json(response)
	plan_ok = plan_repas_valide(parsed, nb_jours, repas_par_jour)
	violations = find_forbidden_ingredients(parsed, restriction_terms)

	if parsed.get("error") in {"no json found", "invalid json", "json incomplete or invalid"} or not plan_ok or violations:
		retry_tokens = max(HF_MAX_TOKENS + 400, int(HF_MAX_TOKENS * 1.6))
		violations_hint = ""
		if violations:
			violations_hint = f"\nCorrection obligatoire: retire ces ingredients interdits detectes: {json.dumps(violations, ensure_ascii=False)}"
		response = call_hf_model(
			prompt + "\nRappel final: renvoie uniquement l'objet JSON ci-dessus, sans details supplementaires." + violations_hint,
			max_new_tokens=retry_tokens,
			temperature=0.1,
			top_p=0.9,
			repetition_penalty=1.1,
		)
		parsed = extract_json(response)
		plan_ok = plan_repas_valide(parsed, nb_jours, repas_par_jour)
		violations = find_forbidden_ingredients(parsed, restriction_terms)

	if "error" not in parsed and not plan_ok:
		parsed = {"error": "json incomplete or invalid", "raw": response}

	if "error" not in parsed and violations:
		parsed = {
			"error": "restrictions_non_respectees",
			"details": "Des ingredients interdits ont ete detectes dans la sortie.",
			"ingredients_interdits_detectes": violations,
			"raw": response,
		}

	if "error" in parsed:
		METRICS["errors"] += 1
	else:
		METRICS["json_ok"] += 1
		parsed["budget"] = {
			"niveau": niveau_budget,
			"libelle": budget,
			"ingredients_reference_count": len(ingredients_autorises_noms),
		}

	return parsed


# ==============================
### TEST LOCAL (SIMULATION APPEL API)
# ==============================

if __name__ == "__main__":
	requete_test = {
		"objectif_alimentaire": "perte_de_poids",
		"nb_jours": 7,
		"repas_par_jour": 3,
		"budget": 2,
		# Vide = aucune interdiction appliquee.
		"restrictions": [],
	}

	resultat = generer_recommandations_plats(requete_test)
	print(json.dumps(resultat, indent=2, ensure_ascii=False))

	# print(json.dumps({"metrics": metrics_snapshot()}, indent=2, ensure_ascii=False))
