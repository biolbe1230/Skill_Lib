"""
Shared VerbNet-based subtask parsing utilities.

Used by both test_skill_lib.py and build_contrastive_skill_emb.py.
"""

import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import verbnet as nltk_verbnet

for _pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "wordnet", "verbnet"):
    nltk.download(_pkg, quiet=True)

_WNL = WordNetLemmatizer()

# Domain-specific disambiguation: for verbs that map to many VerbNet classes,
# choose the most task-relevant class for robotic manipulation.
_VN_CLASS_PREFERENCE: dict[str, str] = {
    "pick":  "get-13.5.1",
    "place": "put-9.1-2",
    "put":   "put-9.1-2",
    "open":  "other_cos-45.4",
    "close": "other_cos-45.4",
    "move":  "roll-51.3.1",
    "turn":  "turn-26.6.1-1",
    "pull":  "push-12-1",
    "push":  "push-12-1-1",
    "grasp": "hold-15.1-1",
    "grab":  "obtain-13.5.2",
    "lift":  "put_direction-9.4",
    "reach": "reach-51.8",
    "set":   "put-9.1-2",
}

_STOP_ARTICLES_VN = {"the", "a", "an", "this", "that", "its"}
_PREPOSITIONS_VN = {
    "to", "on", "in", "at", "from", "into", "onto", "of", "under",
    "below", "behind", "above", "inside", "outside", "with", "for",
    "off", "over", "between", "near",
}

_KNOWN_VERB_MAP = {
    "pick up": "pick",
    "move to": "move",
    "go to":   "go",
    "turn on": "turn",
    "turn off": "turn",
    "set down": "set",
    "place":   "place",
    "put":     "put",
    "open":    "open",
    "close":   "close",
    "pull":    "pull",
    "push":    "push",
    "grasp":   "grasp",
    "grab":    "grab",
    "lift":    "lift",
    "reach":   "reach",
    "shut":    "shut",
}
_KNOWN_VERB_SORTED = sorted(_KNOWN_VERB_MAP.keys(), key=len, reverse=True)


def _resolve_vn_class(lemma: str) -> str:
    """Look up the VerbNet class for a verb lemma using NLTK VerbNet."""
    classes = nltk_verbnet.classids(lemma=lemma)
    if not classes:
        return "unknown-0"
    if lemma in _VN_CLASS_PREFERENCE:
        pref = _VN_CLASS_PREFERENCE[lemma]
        if pref in classes:
            return pref
    return classes[0]


def _parse_subtask_verbnet(text: str) -> dict[str, str]:
    """Parse a subtask description using NLTK tokenisation + VerbNet.

    Returns dict with keys:
        verb_phrase, verb_lemma, verbnet_class, object_phrase, head_noun, level2_key
    """
    cleaned = text.strip().rstrip(".")
    lower = cleaned.lower()

    # 1. Try phrasal verbs first
    verb_phrase = ""
    verb_lemma = ""
    for pv in _KNOWN_VERB_SORTED:
        if lower.startswith(pv):
            verb_phrase = pv
            verb_lemma = _KNOWN_VERB_MAP[pv]
            break

    if not verb_phrase:
        tokens = word_tokenize(cleaned)
        tagged = pos_tag(tokens)
        for word, tag in tagged:
            if tag.startswith("VB"):
                verb_phrase = word.lower()
                verb_lemma = _WNL.lemmatize(word.lower(), "v")
                break
        if not verb_lemma:
            first = tokens[0].lower() if tokens else "unknown"
            verb_phrase = first
            verb_lemma = _WNL.lemmatize(first, "v")

    # 2. VerbNet class lookup
    vn_class = _resolve_vn_class(verb_lemma)

    # 3. Extract object phrase
    rest = lower
    if verb_phrase and lower.startswith(verb_phrase):
        rest = lower[len(verb_phrase):].strip()

    rest_tokens = rest.split()
    while rest_tokens and rest_tokens[0] in _STOP_ARTICLES_VN:
        rest_tokens = rest_tokens[1:]

    obj_tokens: list[str] = []
    for tok in rest_tokens:
        if tok in _PREPOSITIONS_VN and obj_tokens:
            break
        if tok not in _STOP_ARTICLES_VN:
            obj_tokens.append(tok)

    obj_phrase = " ".join(obj_tokens).strip()
    head_noun = obj_tokens[-1] if obj_tokens else "none"

    level2_key = f"{verb_phrase or 'unknown'}+{head_noun}"

    return {
        "verb_phrase": verb_phrase or "unknown",
        "verb_lemma": verb_lemma,
        "verbnet_class": vn_class,
        "object_phrase": obj_phrase,
        "head_noun": head_noun,
        "level2_key": level2_key,
    }
