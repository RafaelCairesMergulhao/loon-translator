"""Equivalência cultural para gírias e expressões sem tradução literal.

Fora do modo IA, o Loon não depende de um modelo generativo. O tom da
conversa escolhe o registro (casual, formal ou viagem) e este módulo troca a
expressão pelo sentido real antes da tradução automática. Trechos
desconhecidos seguem para o Argos, DeepL ou Google já com a grafia coloquial
normalizada. No modo IA, o LLM recebe só as gírias que apareceram na frase.

A tabela cobre os doze idiomas do app. Uma gíria entra na frase quando ela
é dita; o resto do dicionário não vai para o modelo e não aumenta o atraso.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from domain.sorting import merge_sort


CONTEXTS = ("casual", "formal", "travel")

_CONTEXT_ALIASES = {
    "casual": "casual",
    "casual/gírias": "casual",
    "casual / gírias": "casual",
    "gírias": "casual",
    "slang": "casual",
    "informal": "casual",
    "formal": "formal",
    "formal/corporativo": "formal",
    "formal / corporativo": "formal",
    "corporativo": "formal",
    "business": "formal",
    "viagem": "travel",
    "travel": "travel",
}


def normalize_context(value: str | None) -> str:
    if not value:
        return "casual"
    return _CONTEXT_ALIASES.get(value.strip().lower(), "casual")


@dataclass(frozen=True, slots=True)
class Idiom:
    lang: str
    forms: tuple[str, ...]
    gloss: str
    casual: dict[str, str]
    formal: dict[str, str]
    travel: dict[str, str]
    # Palavra que também tem sentido literal: só vira dica para o modelo, sem troca automática.
    hint_only: bool = False


def _pt(forms: tuple[str, ...], gloss: str, casual: str, formal: str, travel: str) -> Idiom:
    return Idiom("pt", forms, gloss, {"en": casual}, {"en": formal}, {"en": travel})


def _en(forms: tuple[str, ...], gloss: str, casual: str, formal: str, travel: str) -> Idiom:
    return Idiom("en", forms, gloss, {"pt": casual}, {"pt": formal}, {"pt": travel})


IDIOMS: tuple[Idiom, ...] = (
    _pt(("tá de boa", "ta de boa", "tô de boa", "to de boa", "de boa", "de boas"), "está tudo bem", "it's all good", "everything is fine", "no problem"),
    _pt(("sem crise",), "não há problema", "no worries", "that is not a problem", "no problem"),
    _pt(("valeu, falou", "valeu falou"), "obrigado, tchau", "thanks, see ya", "thank you, goodbye", "thanks, bye"),
    _pt(("valeu",), "obrigado", "thanks", "thank you", "thank you"),
    # Adjetivo: trocar pelo sinônimo e traduzir a frase inteira preserva a gramática ("muito daora").
    Idiom("pt", ("daora", "dahora", "da hora"), "legal", {}, {}, {}),
    _pt(("tá ligado", "ta ligado", "sacou"), "entende?", "you know?", "do you understand?", "you know?"),
    _pt(("beleza", "blz"), "tudo bem", "alright", "very well", "okay"),
    _pt(("fechou",), "combinado", "deal", "agreed", "okay"),
    _pt(
        ("saiu, mas demorou muito para sair", "saiu mas demorou muito para sair"),
        "a fala saiu, mas demorou",
        "it came out, but it took a long time to come out",
        "it came out, but it took a long time to come out",
        "it came out, but it took a long time to come out",
    ),
    _pt(("bora",), "vamos", "let's go", "let us begin", "let's go"),
    _pt(("show de bola",), "muito bom", "awesome", "excellent", "great"),
    _pt(("mano", "véi", "vei", "parça"), "amigo", "dude", "my friend", "buddy"),
    _pt(("fala sério", "tá de brincadeira", "ta de brincadeira"), "é sério", "are you serious", "is that serious", "are you serious"),
    _pt(("quebrar um galho",), "ajudar", "help me out", "help with this", "help me out"),
    _pt(("dar um jeito",), "resolver", "figure it out", "find a solution", "sort it out"),
    _pt(("não estou nem aí", "nao estou nem ai", "tô nem aí", "to nem ai"), "não me importo", "I don't care", "that does not concern me", "I don't mind"),
    _pt(("encher o saco",), "incomodar", "get on my nerves", "be a nuisance", "bother me"),
    _pt(("cair a ficha",), "finalmente entender", "it finally clicked", "I finally understood", "it finally clicked"),
    _pt(("viajar na maionese",), "falar sem sentido", "talk nonsense", "speak without making sense", "space out"),
    _pt(("pagar mico",), "passar vergonha", "embarrass yourself", "make a fool of yourself", "embarrass yourself"),
    _pt(("chutar o balde",), "desistir", "give up", "abandon the effort", "give up"),
    _pt(("na moral",), "sinceramente", "for real", "honestly", "seriously"),
    _pt(("foi mal",), "desculpe", "my bad", "I apologize", "sorry"),
    _pt(("e aí", "e ai", "eae"), "olá", "what's up", "hello", "hi"),
    _pt(("cê tá doido", "ce ta doido", "cê tá louco", "ce ta louco"), "você está brincando", "you're kidding", "that is hard to believe", "you're kidding"),
    _pt(("tá osso", "ta osso"), "está difícil", "that's tough", "that is difficult", "that's tough"),
    _pt(("tá suave", "ta suave"), "está tudo bem", "it's all good", "everything is fine", "no problem"),
    _pt(("eu topo", "tô dentro", "to dentro"), "eu aceito", "I'm in", "I accept", "I'm in"),
    _pt(("me chama",), "entre em contato", "hit me up", "contact me", "call me"),
    _pt(("quanto tempo",), "há quanto tempo", "long time no see", "it has been a long time", "long time no see"),
    _pt(("os olhos da cara",), "muito caro", "an arm and a leg", "a very high price", "an arm and a leg"),
    _pt(("engolir sapo",), "tolerar algo desagradável", "put up with it", "tolerate an offense", "put up with it"),
    _pt(("dar mole",), "perder a oportunidade", "miss the chance", "fail to pay attention", "miss the chance"),
    _pt(("em cima do muro",), "indeciso", "on the fence", "undecided", "on the fence"),
    _pt(("ficar de boa",), "ficar tranquilo", "take it easy", "remain calm", "take it easy"),
    _pt(("me quebra",), "isso é demais para mim", "give me a break", "that is unreasonable", "give me a break"),
    _pt(("uma vez na vida",), "muito raramente", "once in a blue moon", "very rarely", "almost never"),
    _en(("break a leg",), "good luck", "boa sorte", "desejo sucesso", "boa sorte"),
    _en(("piece of cake",), "very easy", "é moleza", "é muito fácil", "é fácil"),
    _en(("under the weather",), "not feeling well", "não estou muito bem", "não estou me sentindo bem", "estou indisposto"),
    _en(("cost an arm and a leg", "costs an arm and a leg"), "is very expensive", "custa os olhos da cara", "é muito caro", "é muito caro"),
    _en(("once in a blue moon",), "very rarely", "uma vez na vida e outra na morte", "muito raramente", "quase nunca"),
    _en(("spill the beans",), "reveal the secret", "abrir o jogo", "revelar o segredo", "contar o segredo"),
    _en(("hit the road",), "leave", "cair fora", "partir", "ir embora"),
    _en(("when pigs fly",), "that is very unlikely", "quando a vaca tossir", "isso é muito improvável", "isso não vai acontecer"),
    _en(("the ball is in your court",), "the decision is yours", "a bola está com você", "a decisão é sua", "agora depende de você"),
    _en(("bite the bullet",), "face it and go on", "aguentar firme", "aceitar e seguir em frente", "encarar mesmo assim"),
    _en(("let the cat out of the bag",), "reveal the secret", "deixar o segredo escapar", "revelar a informação", "acabar contando"),
    _en(("hang in there",), "keep going", "aguenta firme", "mantenha a calma", "tenha paciência"),
    _en(("no big deal",), "it is not important", "sem crise", "não é importante", "sem problema"),
    _en(("my bad",), "I am sorry", "foi mal", "peço desculpas", "desculpe"),
    _en(("what's up", "whats up"), "hello", "e aí", "olá", "oi"),
    _en(("i'm down", "im down"), "I am willing", "eu topo", "estou disposto", "pode contar comigo"),
    _en(("hit me up",), "contact me", "me chama", "entre em contato", "me liga"),
    _en(("for real",), "seriously", "na moral", "de verdade", "sério"),
    _en(("long time no see",), "it has been a long time", "quanto tempo", "quanto tempo sem nos vermos", "quanto tempo"),
    _en(("easy does it",), "go slowly", "vai com calma", "faça com cuidado", "vai com calma"),
    _en(("keep me posted",), "keep me informed", "me mantém por dentro", "mantenha-me informado", "me avise"),
    _en(("touch base",), "talk briefly", "falar rapidinho", "alinhar rapidamente", "conversar depois"),
    _en(("circle back",), "return to this subject", "voltar nesse assunto", "retomar este assunto", "falar disso depois"),
    _en(("take this offline", "let's take this offline"), "discuss this privately", "ver isso em particular", "tratar isso em particular", "falar disso a sós"),
    _en(("low-hanging fruit",), "the easiest part", "o mais fácil de resolver", "as ações mais simples", "o mais fácil"),
    _en(("move the needle",), "make a real difference", "fazer diferença de verdade", "gerar um impacto relevante", "fazer diferença"),
    _en(("deep dive",), "a detailed look", "olhar a fundo", "uma análise detalhada", "ver com calma"),
    _en(("don't have the bandwidth", "do not have the bandwidth"), "I am not available", "não vou dar conta", "não tenho disponibilidade", "não consigo agora"),
    _en(("call it a day",), "stop for today", "encerrar por hoje", "encerrar o trabalho de hoje", "parar por hoje"),
    _en(("get out of hand", "gets out of hand"), "become uncontrolled", "sair do controle", "ficar fora de controle", "sair do controle"),
    _en(("it's on me", "its on me", "this one's on me"), "I will pay", "eu pago", "eu assumo a despesa", "eu pago"),
    _en(("safe travels",), "have a good trip", "boa viagem", "tenha uma boa viagem", "boa viagem"),
    _en(("cheers",), "thank you", "valeu", "obrigado", "obrigado"),
)


def _pack(value: str | tuple[str, str, str] | None) -> tuple[str, str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value, value, value)
    return value


def _idiom(
    lang: str,
    forms: str | tuple[str, ...],
    gloss: str,
    pt: str | tuple[str, str, str] | None = None,
    en: str | tuple[str, str, str] | None = None,
    hint: bool = False,
) -> Idiom:
    if isinstance(forms, str):
        forms = (forms,)
    casual: dict[str, str] = {}
    formal: dict[str, str] = {}
    travel: dict[str, str] = {}
    for code, packed in (("pt", _pack(pt)), ("en", _pack(en))):
        if packed:
            casual[code], formal[code], travel[code] = packed
    return Idiom(lang, forms, gloss, casual, formal, travel, hint)


from services.slang_table import SLANG

IDIOMS = IDIOMS + tuple(_idiom(*item) for item in SLANG)


def _replacement(pattern: re.Pattern[str], repl: str, text: str) -> str:
    def _apply(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.isupper():
            return repl.upper()
        if word[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    return pattern.sub(_apply, text)


_COLLOQUIAL: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "pt": (
        (re.compile(r"\bc[eê]\b", re.IGNORECASE), "você"),
        (re.compile(r"\bvc\b", re.IGNORECASE), "você"),
        (re.compile(r"\bt[oô]\b", re.IGNORECASE), "estou"),
        (re.compile(r"\bt[aá]\b", re.IGNORECASE), "está"),
        (re.compile(r"\bvamo\b", re.IGNORECASE), "vamos"),
        (re.compile(r"\bpra\b", re.IGNORECASE), "para"),
        (re.compile(r"\bpro\b", re.IGNORECASE), "para o"),
        (re.compile(r"\bpq\b", re.IGNORECASE), "porque"),
        (re.compile(r"\btb\b", re.IGNORECASE), "também"),
        (re.compile(r"\bkd\b", re.IGNORECASE), "cadê"),
        (re.compile(r"\bq\b", re.IGNORECASE), "que"),
        (re.compile(r"\bmt\b", re.IGNORECASE), "muito"),
        (re.compile(r"\bhj\b", re.IGNORECASE), "hoje"),
        (re.compile(r"\bdps\b", re.IGNORECASE), "depois"),
        (re.compile(r"\bagr\b", re.IGNORECASE), "agora"),
        (re.compile(r"\bcmg\b", re.IGNORECASE), "comigo"),
        (re.compile(r"\bctg\b", re.IGNORECASE), "contigo"),
        (re.compile(r"\bngm\b", re.IGNORECASE), "ninguém"),
        (re.compile(r"\bvdd\b", re.IGNORECASE), "verdade"),
    ),
    "es": (
        (re.compile(r"\btb\b", re.IGNORECASE), "también"),
        (re.compile(r"\bxq\b", re.IGNORECASE), "porque"),
        (re.compile(r"\bpq\b", re.IGNORECASE), "porque"),
        (re.compile(r"\bdnd\b", re.IGNORECASE), "dónde"),
        (re.compile(r"\bporfa\b", re.IGNORECASE), "por favor"),
        (re.compile(r"\bq\b", re.IGNORECASE), "que"),
    ),
    "fr": (
        (re.compile(r"\bjsp\b", re.IGNORECASE), "je ne sais pas"),
        (re.compile(r"\bpk\b", re.IGNORECASE), "pourquoi"),
        (re.compile(r"\bstp\b", re.IGNORECASE), "s'il te plaît"),
        (re.compile(r"\bajd\b", re.IGNORECASE), "aujourd'hui"),
        (re.compile(r"\bbjr\b", re.IGNORECASE), "bonjour"),
        (re.compile(r"\bbcq\b", re.IGNORECASE), "parce que"),
    ),
    "de": (
        (re.compile(r"\bhab\b", re.IGNORECASE), "habe"),
        (re.compile(r"\bnem\b", re.IGNORECASE), "einem"),
        (re.compile(r"\bwasn\b", re.IGNORECASE), "was denn"),
        (re.compile(r"\bgrad\b", re.IGNORECASE), "gerade"),
    ),
    "it": (
        (re.compile(r"\bke\b", re.IGNORECASE), "che"),
        (re.compile(r"\bcmq\b", re.IGNORECASE), "comunque"),
        (re.compile(r"\bnn\b", re.IGNORECASE), "non"),
        (re.compile(r"\bxke\b", re.IGNORECASE), "perché"),
        (re.compile(r"\bqlc\b", re.IGNORECASE), "qualcosa"),
    ),
    "ru": (
        (re.compile(r"\bщас\b", re.IGNORECASE), "сейчас"),
        (re.compile(r"\bща\b", re.IGNORECASE), "сейчас"),
        (re.compile(r"\bспс\b", re.IGNORECASE), "спасибо"),
        (re.compile(r"\bпж\b", re.IGNORECASE), "пожалуйста"),
        (re.compile(r"\bпжлст\b", re.IGNORECASE), "пожалуйста"),
    ),
    "en": (
        (re.compile(r"\bgonna\b", re.IGNORECASE), "going to"),
        (re.compile(r"\bwanna\b", re.IGNORECASE), "want to"),
        (re.compile(r"\bgotta\b", re.IGNORECASE), "have to"),
        (re.compile(r"\bkinda\b", re.IGNORECASE), "kind of"),
        (re.compile(r"\bsorta\b", re.IGNORECASE), "sort of"),
        (re.compile(r"\blemme\b", re.IGNORECASE), "let me"),
        (re.compile(r"\bgimme\b", re.IGNORECASE), "give me"),
        (re.compile(r"\bdunno\b", re.IGNORECASE), "do not know"),
        (re.compile(r"\bcuz\b", re.IGNORECASE), "because"),
        (re.compile(r"\b'cause\b", re.IGNORECASE), "because"),
    ),
}

_FORMAL_ONLY: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "en": (
        (re.compile(r"\byeah\b", re.IGNORECASE), "yes"),
        (re.compile(r"\byep\b", re.IGNORECASE), "yes"),
        (re.compile(r"\bnope\b", re.IGNORECASE), "no"),
    ),
    "pt": (
        (re.compile(r"\bné\b", re.IGNORECASE), "não é"),
    ),
}


def language_base(code: str | None) -> str:
    raw = (code or "auto").strip().lower().replace("_", "-")
    if raw.startswith("zh"):
        return "zh"
    return raw.split("-")[0] or "auto"


def normalize_colloquial(text: str, source_code: str, context: str) -> str:
    source = language_base(source_code)
    rules = _COLLOQUIAL.get(source, ())
    if normalize_context(context) == "formal":
        rules = rules + _FORMAL_ONLY.get(source, ())
    for pattern, repl in rules:
        text = _replacement(pattern, repl, text)
    return text


def _match_text(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def _resolve(idiom: Idiom, target: str, context: str) -> tuple[str, str] | None:
    table = getattr(idiom, normalize_context(context))
    if target in table:
        return ("direct", table[target])
    if idiom.gloss:
        return ("gloss", idiom.gloss)
    return None


def _apply_case(raw: str, replacement: str, at_start: bool) -> str:
    if not replacement:
        return replacement
    if at_start or (raw[:1].isupper()):
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _form_pattern(lang: str, form: str) -> re.Pattern[str]:
    body = form[3:] if form.startswith("re:") else re.escape(form)
    # Chinês não separa palavras com espaço. Japonês e coreano colam a partícula no fim.
    if lang == "zh" or (lang == "ko" and form.startswith("re:")):
        return re.compile(body)
    if lang in {"ja", "ko"}:
        return re.compile(rf"(?<!\w){body}")
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


_COMPILED: tuple[tuple[Idiom, tuple[re.Pattern[str], ...]], ...] = tuple(
    (idiom, tuple(_form_pattern(idiom.lang, form) for form in idiom.forms)) for idiom in IDIOMS
)
_SLANG_LANGS = frozenset(idiom.lang for idiom in IDIOMS)


def _find_spans(
    text: str,
    source: str,
    target: str,
    context: str,
    include_hints: bool = True,
) -> list[tuple[int, int, str, str, str]]:
    searchable = _match_text(text)
    found: list[tuple[int, int, str, str, str]] = []
    for idiom, patterns in _COMPILED:
        if idiom.lang != source:
            continue
        if idiom.hint_only and not include_hints:
            continue
        resolved = _resolve(idiom, target, context)
        if resolved is None:
            continue
        kind, value = resolved
        for pattern in patterns:
            for match in pattern.finditer(searchable):
                found.append((match.start(), match.end(), kind, value, text[match.start() : match.end()]))
    # Expressão mais longa primeiro; no empate, vale a ordem da tabela IDIOMS.
    found = merge_sort(found, key=lambda item: (-(item[1] - item[0]), item[0]))
    chosen: list[tuple[int, int, str, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, kind, value, raw in found:
        if any(start < prev_end and end > prev_start for prev_start, prev_end in occupied):
            continue
        occupied.append((start, end))
        chosen.append((start, end, kind, value, raw))
    return merge_sort(chosen, key=lambda item: item[0])


def _has_letters(text: str) -> bool:
    return any(character.isalpha() for character in text)


def detect_idiom_language(text: str) -> str | None:
    scores = {
        lang: sum(end - start for start, end, *_rest in _find_spans(text, lang, "en", "casual"))
        for lang in _SLANG_LANGS
    }
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return None
    return best


def glossary(text: str, source_code: str, target_code: str, context: str) -> list[tuple[str, str]]:
    """Gírias da frase com o equivalente no registro pedido, na ordem em que aparecem."""
    source = language_base(source_code)
    if source == "auto":
        source = detect_idiom_language(text) or "auto"
    if source not in _SLANG_LANGS:
        return []
    spans = _find_spans(text, source, language_base(target_code), normalize_context(context))
    return [(raw, value) for _start, _end, _kind, value, raw in spans]


@dataclass(frozen=True, slots=True)
class CulturalRender:
    text: str
    used_idiom: bool
    fully_resolved: bool


def render_cultural(
    text: str,
    source_code: str,
    target_code: str,
    context: str,
    translate_chunk: Callable[[str], str],
) -> CulturalRender:
    """Troca expressões conhecidas e traduz só o que sobrou."""
    context = normalize_context(context)
    source = language_base(source_code)
    target = language_base(target_code)
    if source == "auto":
        source = detect_idiom_language(text) or "auto"
    spans = _find_spans(text, source, target, context, include_hints=False) if source in _SLANG_LANGS else []

    parts: list[str] = []
    direct: list[str] = []
    cursor = 0
    for start, end, kind, value, raw in spans:
        parts.append(text[cursor:start])
        if kind == "gloss":
            parts.append(value)
        else:
            direct.append(_apply_case(raw, value, at_start=start == 0))
            parts.append(f"\uE000{len(direct) - 1}\uE001")
        cursor = end
    parts.append(text[cursor:])
    merged = normalize_colloquial("".join(parts), source, context)

    if not direct:
        if not _has_letters(merged):
            return CulturalRender(merged, bool(spans), True)
        return CulturalRender(translate_chunk(merged), bool(spans), False)

    pieces = re.split(r"(\uE000\d+\uE001)", merged)
    rendered: list[str] = []
    needs_machine = False
    for piece in pieces:
        marker = re.fullmatch(r"\uE000(\d+)\uE001", piece)
        if marker:
            rendered.append(direct[int(marker.group(1))])
            continue
        if _has_letters(piece):
            prefix, core, suffix = _EDGES.fullmatch(piece).groups()
            opens_sentence = not "".join(rendered).strip() or "".join(rendered).rstrip()[-1] in _SENTENCE_END
            rendered.append(f"{prefix}{_fit_fragment(core, translate_chunk(core), opens_sentence)}{suffix}")
            needs_machine = True
        else:
            rendered.append(piece)
    return CulturalRender(_tidy("".join(rendered)), True, not needs_machine)


_SENTENCE_END = ".!?…。！？"
_DOUBLED = re.compile(r"([?!])\s*[.?!]+")
# Vírgulas nas pontas ficam fora do trecho: o tradutor as descarta.
_EDGES = re.compile(r"([\s,;:]*)(.*?)([\s,;:]*)", re.DOTALL)


def _fit_fragment(original: str, translated: str, opens_sentence: bool) -> str:
    """O tradutor trata um pedaço solto como frase: tira o ponto final e a maiúscula que ele inventou."""
    if original[-1:] not in _SENTENCE_END:
        translated = translated.rstrip("".join(_SENTENCE_END))
    if (
        not opens_sentence
        and original[:1].islower()
        and translated[:1].isupper()
        and translated[1:2].islower()
    ):
        translated = translated[:1].lower() + translated[1:]
    return translated


def _tidy(text: str) -> str:
    """Remove a pontuação repetida quando a expressão trocada já traz a sua ("you know??")."""
    return _DOUBLED.sub(r"\1", text)
