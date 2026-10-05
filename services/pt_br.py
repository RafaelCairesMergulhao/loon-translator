"""Converte a saída do Argos (treinado sobretudo com português europeu) para o português do Brasil.

As regras são aplicadas em ordem, e cada etapa assume a anterior:
vocabulário → "estar a + infinitivo" → pronomes átonos → "tu" para "você" → possessivos.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_Rule = tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _words(pairs: dict[str, str]) -> list[_Rule]:
    rules: list[_Rule] = []
    for portugal in sorted(pairs, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w-]){re.escape(portugal)}(?![\w-])", re.IGNORECASE)
        rules.append((pattern, lambda match, brazil=pairs[portugal]: _match_case(match.group(0), brazil)))
    return rules


_VOCABULARY = _words({
    "a casa de banho": "o banheiro",
    "da casa de banho": "do banheiro",
    "na casa de banho": "no banheiro",
    "casa de banho": "banheiro",
    "o ecrã": "a tela",
    "do ecrã": "da tela",
    "no ecrã": "na tela",
    "ecrã": "tela",
    "autocarro": "ônibus",
    "autocarros": "ônibus",
    "comboio": "trem",
    "comboios": "trens",
    "telemóvel": "celular",
    "telemóveis": "celulares",
    "pequeno-almoço": "café da manhã",
    "rapariga": "garota",
    "raparigas": "garotas",
    "miúdo": "garoto",
    "miúdos": "garotos",
    "equipa": "equipe",
    "equipas": "equipes",
    "utilizador": "usuário",
    "utilizadores": "usuários",
    "fixe": "legal",
    "facto": "fato",
    "contacto": "contato",
    "contactos": "contatos",
    "óptimo": "ótimo",
    "actual": "atual",
    "registo": "registro",
    "sítio": "lugar",
    "casa-de-banho": "banheiro",
    "à espera": "esperando",
})

_GERUND = {"ar": "ando", "er": "endo", "ir": "indo", "or": "ondo", "ôr": "ondo"}
_ESTAR = (
    "estou|estás|está|estamos|estão|estava|estavas|estávamos|estavam|"
    "estive|esteve|estiveram|estar|estarei|estará|estaremos|estarão|esteja|estejam"
)


def _gerund(match: re.Match[str]) -> str:
    verb, stem, ending, clitic = match.group(1), match.group(2), match.group(3).lower(), match.group(4)
    reflexive = f"{clitic} " if clitic else ""
    return f"{verb} {reflexive}{stem}{_GERUND[ending]}"


_PROGRESSIVE: list[_Rule] = [
    (re.compile(rf"\b({_ESTAR}) a (\w+?)(ar|er|ir|or|ôr)(?:-(se|me|te|nos))?\b", re.IGNORECASE), _gerund),
    (re.compile(r"\besperando d(o|a|os|as)\b", re.IGNORECASE), lambda match: f"esperando {match.group(1)}"),
    (re.compile(r"\besperando de\b", re.IGNORECASE), "esperando"),
]

_ATONIC = "me|te|se|nos|lhe|lhes"


def _proclisis(match: re.Match[str]) -> str:
    verb, pronoun = match.group(1), match.group(2).lower()
    if verb[:1].isupper():
        return f"{pronoun.capitalize()} {verb[:1].lower()}{verb[1:]}"
    return f"{pronoun} {verb}"


_CLITICS: list[_Rule] = [
    # "ouvir-me" → "me ouvir"; "Envie-me" → "Me envie".
    (re.compile(rf"\b(\w+)-({_ATONIC})\b(?!-)", re.IGNORECASE), _proclisis),
    # "faço-o" → "faço": no Brasil o objeto fica implícito na fala.
    (re.compile(r"\b(\w+)-(o|a|os|as)\b(?!-)", re.IGNORECASE), lambda match: match.group(1)),
]

_SECOND_PERSON = {
    "consegues": "consegue", "podes": "pode", "queres": "quer", "tens": "tem",
    "estás": "está", "sabes": "sabe", "vens": "vem", "vais": "vai", "és": "é",
    "precisas": "precisa", "gostas": "gosta", "achas": "acha", "falas": "fala",
    "percebes": "entende", "entendes": "entende", "ouves": "ouve", "vês": "vê",
    "fazes": "faz", "dizes": "diz", "moras": "mora", "trabalhas": "trabalha",
    "viste": "viu", "fizeste": "fez", "disseste": "disse", "foste": "foi",
    "estiveste": "esteve", "tiveste": "teve", "puseste": "pôs", "quiseste": "quis",
    "recebeste": "recebeu", "percebeste": "entendeu", "ouviste": "ouviu",
    "precisares": "precisar", "quiseres": "quiser", "puderes": "puder", "tiveres": "tiver",
}

_NEGATIVE_IMPERATIVE = {
    "te preocupes": "se preocupe", "tenhas": "tenha", "faças": "faça", "esqueças": "esqueça",
    "digas": "diga", "vás": "vá", "sejas": "seja", "fiques": "fique", "desistas": "desista",
    "te esqueças": "se esqueça", "te atrases": "se atrase",
}


def _you(match: re.Match[str]) -> str:
    subject, verb = match.group(1), match.group(2)
    form = _SECOND_PERSON[verb.lower()]
    if subject and subject.lower() == "tu":
        return _match_case(subject, f"você {form}")
    if match.start() > 0 and match.string[match.start() - 1 : match.start()] in "-":
        return verb
    return _match_case(verb, f"você {form}")


def _second_person_rules() -> list[_Rule]:
    verbs = "|".join(sorted(map(re.escape, _SECOND_PERSON), key=len, reverse=True))
    negatives = "|".join(sorted(map(re.escape, _NEGATIVE_IMPERATIVE), key=len, reverse=True))
    return [
        (
            re.compile(rf"\b(não) ({negatives})\b", re.IGNORECASE),
            lambda match: f"{match.group(1)} {_NEGATIVE_IMPERATIVE[match.group(2).lower()]}",
        ),
        (re.compile(rf"(?:\b(tu) )?(?<![\w-])({verbs})(?![\w-])", re.IGNORECASE), _you),
    ]


_TREATMENT = _second_person_rules()

_POSSESSIVES: list[_Rule] = [
    (re.compile(r"\b(o|a|os|as) (meu|minha|meus|minhas|seu|sua|seus|suas|nosso|nossa|nossos|nossas)\b", re.IGNORECASE),
     lambda match: _match_case(match.group(1), match.group(2).lower())),
    (re.compile(r"\b(?:(o|a|os|as) )?(teu|tua|teus|tuas)\b", re.IGNORECASE),
     lambda match: _match_case(match.group(1) or match.group(2),
                               {"teu": "seu", "tua": "sua", "teus": "seus", "tuas": "suas"}[match.group(2).lower()])),
    (re.compile(r"\b(tenho|tem|temos|têm|ter) a certeza\b", re.IGNORECASE), lambda match: f"{match.group(1)} certeza"),
    (re.compile(r"\bvocê você\b", re.IGNORECASE), lambda match: match.group(0).split()[0]),
]

RULES: tuple[_Rule, ...] = (*_VOCABULARY, *_PROGRESSIVE, *_CLITICS, *_TREATMENT, *_POSSESSIVES)


def to_brazilian(text: str) -> str:
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text
