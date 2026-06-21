"""Small instruction parser for the Titanic training CLI."""
import json
import os
import re
import urllib.request
from pathlib import Path

class PlanResolutionError(ValueError): pass

# Naredne liste i rjecnici su za fallback opciju. Fallback se poziva ako i samo ako iz nekog razloga openrouter nije handlan dobro (zbog nedostupnosti, pogresnog api key- a ili neki drugi ralog).

DEFAULT_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

FEATURE_ALIASES = {
    "passenger id": "PassengerId", "passengerid": "PassengerId", "pclass": "Pclass", "passenger class": "Pclass",
    "name": "Name", "sex": "Sex", "gender": "Sex", "age": "Age", "sibsp": "SibSp", "parch": "Parch",
    "ticket": "Ticket", "fare": "Fare", "cabin": "Cabin", "embarked": "Embarked", "survived": "Survived",
}

MODEL_ALIASES = {
    "logisticregression": "logistic_regression", "decisiontree": "decision_tree", "randomforest": "random_forest",
    "gradientboosting": "gradient_boosting", "xgboost": "xgboost", "xgb": "xgboost", "knn": "knn", "knearest": "knn",
}
VALUE_ALIASES = {"average": "mean", "avg": "mean", "mode": "most_frequent", "mostfrequent": "most_frequent"}
SUPPORTED_MODELS = {"logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "xgboost"}
SUPPORTED_NUMERIC = {"mean", "median", "most_frequent"}
SUPPORTED_CATEGORICAL = {"most_frequent", "constant"}
OPENROUTER_DEFAULT_MODEL = "cohere/north-mini-code:free"
GENERIC_MODEL_WORDS = {
    "titanic", "survival", "binary", "classification", "classifier",
    "ml", "machine", "learning", "prediction", "predictive",
}


# Funkcija orkestrira validacije i fallback parsiranje. Pretvara sirove podatke u instrukcije iz kojih se moze kreirati plan pozivanja 
# handlanje slucaja kao sto su prazne instrukcije, target leackage (leakanje rezultata modelu), provjera openrouter dostupnosti

def parse_instruction(instruction):
    instruction = instruction.strip()
    if not instruction:
        raise PlanResolutionError("Instruction cannot be empty.")
    reject_target_leakage(instruction)
    raw_plan, parser, warnings = fallback_parse(instruction), "fallback", []
    env = read_env(); api_key = env.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        try:
            raw_plan = ask_openrouter(instruction, api_key, env.get("OPENROUTER_MODEL"))
            parser = "openrouter"
        except Exception as exc:
            warnings.append(f"OpenRouter parser failed ({exc}); used fallback parser.")
    else:
        warnings.append("OPENROUTER_API_KEY is not set; used fallback parser.")
    plan = build_plan(raw_plan, instruction, parser)
    plan["warnings"].extend(warnings)
    if parser == "fallback":
        plan["warnings"].append("Fallback parser handles simple Titanic instructions only.")
    return plan

# ucitavanje env varijabli
def read_env():
    values, path = {}, Path(".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    values.update(os.environ)
    return values

# Poziv prema openrouteru za kreiranje plana izvrsavanja u .json formatu. Kao payload se salju informacije o modelu koji koristi, temperature modela, 
# prompt - hardkodirana pravila koja ocekujemo da se postuju prilikom svakog upita LLM- u (bolje ih je u prod okruzenju drzati u bazi ali za poc necemo to raditi). Prompt ogranicava llm i 
# postavlja pravila sta smije vratiti. Naglasava da je prompt od sistema
# Instrukcije od korisnika daju konkretan zahtjev, npr. train random forest with Age and Sex i naglasavaju da ih je poslao korisnik. Ovo je vazno zbog nacina na koji llm funkcionise. 

def ask_openrouter(instruction, api_key, model):
    prompt = (
        "Return JSON only for a Titanic survival classifier. "
        "Allowed keys: selected_features, drop_features, model, numeric_imputation, "
        "categorical_imputation, scaling, test_size. Target is always Survived. "
        "Supported models are logistic_regression, decision_tree, random_forest, "
        "gradient_boosting, and xgboost. Do not replace an unknown requested model "
        "with a different supported model."
    )
    payload = {"model": (model or OPENROUTER_DEFAULT_MODEL).strip(), "temperature": 0, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": instruction}]}
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}") + 1
    if start < 0 or end <= start:
        raise PlanResolutionError("LLM did not return a JSON object.")
    raw_plan = json.loads(content[start:end])
    if not isinstance(raw_plan, dict):
        raise PlanResolutionError("LLM response must be a JSON object.")
    return raw_plan

# Kreiranje naivnog deterministickog fallback plana ukoliko openrouter vrati gresku. Koristi matching klj. rijeci
# vecinom ai koristen prilikom generisanja pogotovo regex izraza. 

def fallback_parse(instruction):
    text, plan = instruction.lower(), {}
    for key, model in MODEL_ALIASES.items():
        if key in normalize(instruction):
            plan["model"] = model
            break
    selected, dropped = selected_features_from(instruction), dropped_features_from(instruction)
    if selected: plan["selected_features"] = selected
    if dropped: plan["drop_features"] = dropped
    if re.search(r"\b(avg|average|mean)\b", text):
        plan["numeric_imputation"] = "mean"
    elif "median" in text:
        plan["numeric_imputation"] = "median"
    if re.search(r"\bmode\b|most[\s_-]*frequent", text):
        plan["categorical_imputation"] = "most_frequent"
    elif "constant" in text:
        plan["categorical_imputation"] = "constant"
    if re.search(r"\b(scale|scaling|standardize|standardise)\b", text): plan["scaling"] = True
    match = re.search(r"\btest(?:\s+size)?\s*(?:=|:|of)?\s*(0?\.\d+|\d+%)", text)
    if match:
        value = match.group(1)
        plan["test_size"] = float(value[:-1]) / 100 if value.endswith("%") else float(value)
    return plan


# Normalizacija i validacija plana izvrsavanja prije treninga. Ovo je vazno jer ml pipeline je izuzetno osjetljiv na greske.
def build_plan(raw_plan, instruction, parser):
    warnings, raw_plan = [], raw_plan or {}
    target = str(raw_plan.get("target") or "Survived").strip()
    if target.lower() != "survived":
        warnings.append(f"Unsupported target '{target}' ignored; using Survived.")
        target = "Survived"
    selected = clean_features(raw_plan.get("selected_features"), warnings)
    dropped = clean_features(raw_plan.get("drop_features"), warnings)
    if "Survived" in selected:
        raise PlanResolutionError("Unsafe instruction rejected: Survived cannot be a feature.")
    if "Survived" in dropped:
        dropped.remove("Survived")
        warnings.append("Ignored Survived in drop_features because it is the target.")
    if isinstance(raw_plan.get("warnings"), list):
        warnings.extend(str(warning) for warning in raw_plan["warnings"])
    model = requested_model_from(instruction) or raw_plan.get("model")
    return {
        "raw_instruction": instruction, "target": target, "selected_features": selected or DEFAULT_FEATURES,
        "drop_features": dropped, "model": clean_choice(model, "logistic_regression", SUPPORTED_MODELS, warnings, "model"),
        "numeric_imputation": clean_choice(raw_plan.get("numeric_imputation"), "median", SUPPORTED_NUMERIC, warnings, "numeric_imputation"),
        "categorical_imputation": clean_choice(raw_plan.get("categorical_imputation"), "most_frequent", SUPPORTED_CATEGORICAL, warnings, "categorical_imputation"),
        "scaling": clean_bool(raw_plan.get("scaling")), "test_size": clean_test_size(raw_plan.get("test_size"), warnings),
        "parser": parser, "warnings": warnings,
    }


def requested_model_from(instruction):
    normalized_instruction = normalize(instruction)
    for key, model in sorted(MODEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if key in normalized_instruction:
            return model

    patterns = [
        r"\btrain\s+(?:an?\s+|the\s+)?([a-zA-Z][a-zA-Z0-9_\s-]{0,40}?)\s+(?:model|classifier)\b",
        r"\buse\s+(?:an?\s+|the\s+)?([a-zA-Z][a-zA-Z0-9_\s-]{0,40}?)\s+(?:model|classifier)\b",
        r"\bmodel\s*(?:is|=|:)\s*([a-zA-Z][a-zA-Z0-9_\s-]{0,40})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" -_")
        if candidate and not is_generic_model_description(candidate):
            return candidate
    return None


def is_generic_model_description(value):
    words = re.findall(r"[a-z0-9]+", str(value).lower())
    return bool(words) and all(word in GENERIC_MODEL_WORDS for word in words)

def clean_choice(value, default, supported, warnings, label):
    if value in (None, ""):
        return default
    key = MODEL_ALIASES.get(normalize(value), VALUE_ALIASES.get(normalize(value), normalize(value)))
    if key in supported:
        return key
    warnings.append(f"Unsupported {label} '{value}' ignored; using {default}.")
    return default


# validacija i odbijanje nevalidnih fature- a, tj naziva kolona iz dataseta
def clean_features(value, warnings):
    if value in (None, ""):
        return []
    parts = re.split(r"[,;]|\band\b", value) if isinstance(value, str) else value
    if not isinstance(parts, list):
        warnings.append("Ignored invalid feature list.")
        return []
    features = []
    for part in parts:
        found = features_in(str(part))
        if not found:
            warnings.append(f"Ignored unknown feature '{part}'.")
        for feature in found:
            if feature not in features:
                features.append(feature)
    return features

# pretvaranje i validacija vrijednosti u boolean 
def clean_bool(value): return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "y"}

# Validacija test split omjera kako treningu u planu ne bi proslijedii nevalidne omjere 
def clean_test_size(value, warnings):
    if value in (None, ""):
        return 0.2
    try:
        text = str(value).strip()
        size = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except (TypeError, ValueError):
        warnings.append(f"Invalid test_size '{value}' ignored; using 0.2.")
        return 0.2
    if 0.05 <= size <= 0.5:
        return size
    warnings.append(f"Out-of-range test_size '{value}' ignored; using 0.2.")
    return 0.2

def selected_features_from(instruction):
    patterns = [
        r"\b(?:drop|remove)\s+all\s+columns?\s+except\s+([^.;]+)",
        r"\bfeatures?\s*(?:are|:|=)\s*([^.;]+)",
        r"\b(?:using|use)\s+([^.;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if not match:
            continue
        features = features_in(match.group(1))
        if "Survived" in features:
            raise PlanResolutionError("Unsafe instruction rejected: Survived cannot be a feature.")
        if len(features) > 1 or "except" in pattern or "features" in pattern or starts_with_feature(match.group(1)):
            return features
    return []

def dropped_features_from(instruction):
    match = re.search(r"\b(?:drop|exclude|without|remove)\s+([^.;]+)", instruction, flags=re.IGNORECASE)
    return [] if not match or "except" in match.group(1).lower() else features_in(match.group(1))

def features_in(text):
    matches = []
    for alias, feature in FEATURE_ALIASES.items():
        pattern = r"\b" + r"[\s_-]*".join(re.escape(word) for word in alias.split()) + r"\b"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), feature))
    features = []
    for _, feature in sorted(matches):
        if feature not in features:
            features.append(feature)
    return features

def starts_with_feature(text):
    text = re.sub(r"^(?:only|the)\s+", "", str(text).strip(), flags=re.IGNORECASE)
    for alias in FEATURE_ALIASES:
        pattern = r"^" + r"[\s_-]*".join(re.escape(word) for word in alias.split()) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False

# Odbijanje instrukcija koje pokusavaju podvuci rezultujucu kolonu kao input karakteristiku
def reject_target_leakage(instruction):
    feature_list = r"\bfeatures?\s*[:=][^.]*\bsurvived\b"
    target_as_input = r"\bsurvived\s+as\s+(?:an?\s+)?(?:input|feature|predictor)"
    if re.search(feature_list, instruction, flags=re.IGNORECASE) or re.search(target_as_input, instruction, flags=re.IGNORECASE):
        raise PlanResolutionError("Unsafe instruction rejected: Survived cannot be a feature.")


# Normalizcija teksta tako da pretraga aliasa ignorise space i znakove interpukcije. Mada, pravilnije bi bilo zvati standardizacija

def normalize(value): 
    return re.sub(r"[^a-z0-9]+", "", str(value).lower().replace("classifier", "").replace("regressor", ""))
