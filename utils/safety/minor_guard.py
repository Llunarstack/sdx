"""Unconditional minor-sexualization guard for every text path into an SDX model.

Adult NSFW is allowed. Sexual content involving minors is not, and nothing can switch
this block off: the module reads no CLI args, env vars, config or presets, and callers
never wrap it in ``try``/``except``.

* Generation: block a hard term, or youth and sexual terms together in the positive, or
  a negative that removes age ("adult", "mature") next to a sexual positive, or a negative
  that removes clothing ("clothes", "sfw", "censored") next to a youth positive.
* Training data: block a hard term or any youth term at any rating, which keeps
  depictions of children apart from adult sexual content (Thorn / All Tech Is Human,
  *Safety by Design for Generative AI*, 2024), plus the age-removing negative caption.
* Every sexual prompt also gets :data:`YOUTH_NEGATIVE`, a negative-prompt form of the
  safety guidance in Safe Latent Diffusion (Schramowski et al., CVPR 2023).

A plain word list is trivially evaded, so the text is checked in several spellings: NFKC
without format characters or diacritics, lookalike letters folded to Latin (Cyrillic,
Greek, Armenian, small capitals, regional indicators), prompt syntax stripped, leetspeak,
spaced-out letters, tokens rejoined ("chi ld"), and one-edit typos ("chlid"). Youth also
covers ages in digits or words, grades, negated adulthood ("not an adult") and a
non-English lexicon. The text layer is stdlib only, so ``config`` and data loaders can
import it cheaply.

Keyword filters are provably bypassable (SneakyPrompt, Ring-A-Bell, MMA-Diffusion), so
:func:`require_image_allowed` adds an image layer: a zero-shot CLIP ViT-L/14 age check
on sexual images (prompt-driven or NSFW classifier above :data:`NSFW_FLAG`). Its heavy
imports load only when it runs.
"""

from __future__ import annotations

import functools
import os
import re
import unicodedata
import warnings

__all__ = [
    "HARD",
    "HALF_FLAG",
    "NSFW_FLAG",
    "SEXUAL",
    "WHOLE_FLAG",
    "YOUTH",
    "YOUTH_NEGATIVE",
    "NegativeText",
    "ProhibitedContentError",
    "ZeroShotAge",
    "is_sexual",
    "normalize",
    "reasons",
    "require_allowed",
    "require_encodable",
    "require_image_allowed",
]

YOUTH_NEGATIVE = "child, childlike body, underage, young-looking face"


class ProhibitedContentError(ValueError):
    """The text asks for sexual content involving minors."""


class NegativeText(str):
    """A negative prompt already screened by :func:`require_allowed`.

    Built-in negatives legitimately pair youth and sexual words ("child, loli, extra
    genitals"), so :func:`require_encodable` skips this marker. Any string edit (f-string,
    ``.strip()``, ``str()``) returns a plain ``str``, so rewritten text is checked again
    as a positive.
    """

    __slots__ = ()


_KEEP_MARKS = frozenset(
    "\N{COMBINING TILDE}"  # keeps Spanish/Portuguese ñ apart from n for the non-English lexicon
    "\N{COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK}\N{COMBINING KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK}"
)
_FILLERS = str.maketrans(
    dict.fromkeys(
        "\N{HANGUL CHOSEONG FILLER}\N{HANGUL JUNGSEONG FILLER}\N{HANGUL FILLER}\N{HALFWIDTH HANGUL FILLER}"
        "\N{BRAILLE PATTERN BLANK}",
        " ",
    )
)
# Cyrillic, Greek, Armenian and IPA letters drawn like Latin ones (one row per script).
_LOOKALIKES = str.maketrans(
    "авгеіјкмнопрстухѕԁӏԛԝһү" "αβεηικνορτυχωγϲϳ" "օսհոցզ" "ɩɑ",
    "abreijkmhonpctyxsdlqwhy" "abenikvoptuxwycj" "ouhngq" "ia",
)  # fmt: skip
_LATIN_NAME = re.compile(
    r"(?:(?:NEGATIVE |SQUARED |CIRCLED )+)?LATIN (?:SMALL |CAPITAL )?LETTER (?:SMALL CAPITAL |SCRIPT |DOTLESS )?"
    r"([A-Z])(?: WITH .+)?|REGIONAL INDICATOR SYMBOL LETTER ([A-Z])"
)


class _LatinFold(dict):
    """``str.translate`` table: the lookalikes plus any character Unicode names as a Latin letter
    ("LATIN LETTER SMALL CAPITAL L", "L WITH STROKE", "REGIONAL INDICATOR SYMBOL LETTER L")."""

    def __missing__(self, cp: int) -> str:
        m = _LATIN_NAME.fullmatch(unicodedata.name(chr(cp), "")) if cp > 0x7F else None
        self[cp] = out = (m.group(1) or m.group(2)).lower() if m else chr(cp)
        return out


_FOLD = _LatinFold(_LOOKALIKES)
_FOLD[ord("\N{COMBINING TILDE}")] = None
_LEET = {"0": "o", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "+": "t", "€": "e"}
_LEET_I = str.maketrans({**_LEET, "1": "i", "!": "i", "|": "i", "¡": "i"})
_LEET_L = str.maketrans({**_LEET, "1": "l", "!": "i", "|": "l", "¡": "i"})

_LESS_THAN = re.compile(r"<\s*(?=\d)")
_LORA = re.compile(r"<\s*(?:lora|lyco|lycoris|hypernet)\s*:\s*([^:>]*)[^>]*>")
_EMBEDDING = re.compile(r"\bembedding\s*:")
_WEIGHT = re.compile(r":\s*-?\d*\.?\d+\s*(?=[)\]}>])")
_BRACKETS = re.compile(r"[()\[\]{}<>]")
_BREAK = re.compile(r"\bbreak\b|\s*\n\s*")
_INNER_SEP = re.compile(r"_|(?<=\w)[-./'\u2019`](?=\w)")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_ASCII_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_LETTER_RUN = re.compile(r"\b(?:[a-z][\W_]{1,3}){2,}[a-z]\b")
_NON_LETTER = re.compile(r"[^a-z]")
_REPEAT = re.compile(r"(.)\1{2,}")
_WORD = re.compile(r"[a-z]{4,}")
_JOIN_TOKEN = re.compile(r"[a-z0-9]+")
_SEGMENTS = re.compile(r"[,;|]")
_GLUED_DIGITS = re.compile(r"\b(\d+)([a-z]{3,})|([a-z]{3,})(\d+)\b")  # "loli2", "3loli"
_PAREN_NUM = re.compile(r"\(\s*(\d{1,2})\s*\)")
_CJK_GAP = re.compile(
    r"(?<=[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3])[\s\W_]+(?=[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3])"
)

_ALLOW_PHRASES = re.compile(
    r"\b(?:aizawa shou?ta|shou?ta aizawa|"
    r"(?:gothic|goth|sweet|classic|punk|wa|qi|hime|kuro|shiro|country|sailor|casual|steampunk) lolitas?|"
    r"lolita (?:fashion|dress\w*|style|outfit|clothes|clothing|coord\w*|aesthetic|lace|hairband|headdress|bonnet|"
    r"shoes|jsk|skirt))\b"
)

# Typo-tolerant spelling (one deleted, inserted or swapped letter) of these words is read as the word,
# which catches "chlid", "chld", "childs", "lollicon", "litle girl", "yung girl", "nakd".
_TYPO_ROOTS = frozenset(
    {
        "child", "children", "toddler", "toddlers", "teenager", "teenagers", "preteen", "preteens", "kiddie",
        "kiddies", "infant", "infants", "newborn", "newborns", "schoolgirl", "schoolboy", "underage", "underaged",
        "pubescent", "prepubescent", "adolescent", "juvenile", "kindergarten", "kindergartner", "little", "young",
        "lolicon", "shotacon", "toddlercon", "jailbait", "pedophile", "naked", "nudity", "sexual", "topless", "hentai",
        "explicit",
    }
)  # fmt: skip
_NOT_TYPOS = frozenset(
    {"childe", "schild", "chil", "chid", "hild", "toddle", "toddles", "infanta", "infantas", "snaked", "asexual"}
)


def _edits(word: str) -> set[str]:
    """``word`` with one letter deleted or two adjacent letters swapped."""
    dels = {word[:i] + word[i + 1 :] for i in range(len(word))}
    swaps = {word[:i] + word[i + 1] + word[i] + word[i + 2 :] for i in range(len(word) - 1)}
    return dels | swaps


_TYPOS = {e: r for r in sorted(_TYPO_ROOTS) for e in _edits(r) if e not in _TYPO_ROOTS and e not in _NOT_TYPOS}


@functools.lru_cache(maxsize=1 << 16)
def _respell(tok: str) -> str:
    if tok in _TYPO_ROOTS or tok in _NOT_TYPOS:
        return tok
    if len(tok) >= 6 and (split := _split_compound(tok)) != tok:
        return split
    if tok in _TYPOS:
        return _TYPOS[tok]
    if len(tok) > 5:
        for i in range(len(tok)):
            if (cand := tok[:i] + tok[i + 1 :]) in _TYPO_ROOTS:
                return cand
    return tok


def _respelled(text: str) -> str:
    fixes = {t: f for t in set(_WORD.findall(text)) if (f := _respell(t)) != t}
    return _WORD.sub(lambda m: fixes.get(m.group(), m.group()), text) if fixes else text


_HARD_ROOTS = frozenset(
    {
        "loli", "lolis", "lolicon", "lolikon", "rori", "rorikon", "roricon", "shota", "shotas",
        "shotacon", "shotakon", "shouta", "toddlercon", "todlercon", "jailbait", "pedo",
        "pedophile", "pedophilia", "paedo", "paedophile", "paedophilia", "csam", "pthc",
    }
)  # fmt: skip
# Words that tokens split by spaces or punctuation are rejoined into ("chi ld", "nu de").
_JOIN_ROOTS = {
    **dict.fromkeys(_HARD_ROOTS, "hard"),
    **dict.fromkeys(
        (
            "child", "children", "childs", "toddler", "toddlers", "teen", "teens", "teenager", "teenagers", "preteen",
            "preteens", "kids", "kiddie", "kiddies", "underage", "schoolgirl", "schoolgirls", "schoolboy",
            "schoolboys", "infant", "infants", "newborn", "minors", "tween", "tweens", "juvenile", "pubescent",
            "prepubescent", "adolescent",
        ),
        "youth",
    ),
    **dict.fromkeys(
        (
            "nude", "nudes", "nudity", "naked", "nsfw", "sexy", "sexual", "porn", "hentai", "erotic", "explicit",
            "topless", "lewd", "nipples", "genitals", "vagina", "pussy",
        ),
        "sexual",
    ),
}  # fmt: skip
_JOIN_PREFIXES = frozenset(r[:k] for r in _JOIN_ROOTS for k in range(1, len(r)))
# A glued token made only of these words is split ("nudechild", "sexylittlegirl", "lolinsfw").
_COMPOUND_WORDS = frozenset(_JOIN_ROOTS) | {
    "sex", "girl", "girls", "boy", "boys", "little", "young", "small", "tiny", "kid", "baby", "lil",
}  # fmt: skip


def _split_compound(tok: str) -> str:
    """``tok`` as two or more :data:`_COMPOUND_WORDS` separated by spaces, else ``tok`` unchanged."""
    parts: list[list[str] | None] = [[]] + [None] * len(tok)
    for end in range(3, len(tok) + 1):
        for start in range(end - 3, -1, -1):
            if parts[start] is not None and tok[start:end] in _COMPOUND_WORDS:
                parts[end] = [*parts[start], tok[start:end]]
                break
    words = parts[-1]
    return " ".join(words) if words and len(words) > 1 else tok


HARD = re.compile(
    r"\b(?:lolis?|loli[ck]on|loli(?:baba|porn\w*|hentai)|rori(?:kon|con)?|shotas?|shota[ck]on|shouta|todd?lercon|"
    r"jail ?bait|cunny|pa?edo(?:s|bear|phil\w*|fil\w*)?|csam|pthc|(?:child|kiddie|kiddy) ?porn\w*)\b"
)
_HARD_CJK = re.compile(
    r"(?<![\u3040-\u30ff])(?:ロリ|ろり|ショタ|しょた)(?:コン|こん)?(?![\u3040-\u30ff])|萝莉|正太控|"
    r"(?<![\uac00-\ud7a3])(?:로리|쇼타)(?![\uac00-\ud7a3])|児童ポルノ|儿童色情"
)

_NUM_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
)
_ORDINALS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
    "fifteenth|sixteenth|seventeenth"
)
YOUTH = re.compile(
    r"\b(?:children|child(?! bearing hips)|kids?|kidz|kiddos?|kiddies?|toddlers?|infants?|newborns?|randoseru|"
    r"bab(?:y|ies)(?! (?:oil|blue|pink|bump|dolls?|powder))|tots?(?<!tater tot)(?<!tater tots)|"
    r"pre ?teens?|tweens?|tweenies?|teenies?|teenybopper|under ?aged?|teen(?:s|agers?|age|aged)?|"
    r"adolescen(?:t|ts|ce)|(?:pre ?)?pubescen(?:t|ce)|puberty|juveniles?|youngsters?|younglings?|"
    r"lolitas?|lollis?|lol+(?:ie|ee|ey|y)s?(?<!ice lolly)(?<!ice lollies)|shotta|junior idols?|"
    r"school ?(?:girl|boy|kid|child)s?|(?:little|lil|smol|wee|young|small|tiny|teeny) (?:girl|boy)s?|(?:little|lil|wee) ones?|"
    r"(?:female|male|girl|boy) child(?:ren)?|child ?(?:like|body)|"
    r"young(?:er)? looking|looks? young(?:er)?|aged? down|age regress\w*|de ?aged|"
    r"(?:elementary|grade|primary|middle|pre|nursery) ?school(?:ers?| students?| girls?| boys?)?|junior high|"
    r"elementary (?:students?|pupils?|girls?|boys?|kids?|age)|school ?age[ds]?|high ?schoolers?|"
    r"high school (?:students?|girls?|boys?|freshm[ae]n|sophomores?)|kindergart(?:en|ens|ner|ners)|"
    r"(?:boy|girl|cub) scouts?|"
    r"shougakusei|chuugakusei|joshi ?kousei|cubs?|minors|"
    r"(?:a|an|the|is|as|underage|sexuali[sz]ed|young|nude|naked|sexy|female|male) minor"
    r"(?! (?:key|chord|scale|details?|role|character|changes?|issues?|injur\w*|flaws?|league))|"
    r"minor (?:girl|boy|child|kid|female|male|aged?)s?|"
    r"(?:not|non|isn ?t|is not|never|not yet|barely|almost|nearly) (?:an? |the )?(?:adults?|grown ?ups?|of age|"
    r"legal|18|eighteen)|(?:below|younger than|less than|under) ?(?:the )?(?:age of )?(?:18|eighteen|consent|"
    r"majority|legal age)|u ?18|sweet (?:16|sixteen)|"
    rf"(?:(?:[1-9]|1[0-2])(?:st|nd|rd|th)|{_ORDINALS}) grade(?:rs?)?(?! teachers?)|"
    r"(?:grade|year) (?:[1-9]|1[0-2]) (?:students?|pupils?|girls?|boys?|kids?)|grade (?:[1-9]|1[0-2])(?= ?(?:$|[,;|.!?]))|"
    rf"(?:[1-9]|1[0-7])(?:st|nd|rd|th) birthday|(?:{_ORDINALS}) birthday)\b"
)
_YOUTH_INTL = re.compile(
    r"\b(?:niñ[ao]s?|niñit[ao]s?|menor(?:es)? de (?:edad|idade)|colegialas?|chiquill[ao]s?|"
    r"criancas?|meninas?|meninos?|novinhas?|"
    r"enfants?|fillettes?|petites? filles?|petits? garcons?|mineures?|ecolieres?|"
    r"minderjahrig\w*|kleine[nrs]? (?:madchen|junge)|schulmadchen|minderjarig\w*|klein meisje|"
    r"(?:ein|das|den|dem|des|kein|nackte[nrs]?|kleine[nrs]?|junge[nrs]?) kind(?:er|ern)?|"
    r"bambin[aeio]|ragazzin[aeio]|minorenn[ei]|scolarett[ae]|"
    r"dziewczynk\w*|dzieck\w*|dzieci|nieletni\w*|"
    r"девочк\w*|девчонк\w*|малолетк\w*|школьни\w*|ребен\w*|дети|детск\w*|несовершеннолетн\w*|подрост\w*|малыш\w*|"
    r"мальчик\w*|kodomo|youjo|youji)\b|"
    r"子供|子ども|児童|幼女|幼児|少女|小学生|中学生|女子高生|未成年|女児|男児|ランドセル|ジュニアアイドル|"
    r"儿童|幼儿|小女孩|小男孩|女童|男童|孩子|正太|初中生|아동|미성년자|어린이|초등학생|중학생|여중생|여고생|소녀|꼬마|"
    r"[\N{BABY}\N{CHILD}\N{GIRL}\N{BOY}\N{CHILDREN CROSSING}\N{SCHOOL SATCHEL}]"
)
_AGE = re.compile(
    rf"\b(\d{{1,2}}|{_NUM_WORDS}) ?(?:y ?o|(?:yrs?|years?) ?old|years? of age)\b"
    rf"|\b(\d{{1,2}}|{_NUM_WORDS}) ?(?:years?|yrs?|y)(?= ?(?:$|[,;|.!?]))"
    rf"|\bage[ds]? ?(?:[:=] ?)?(?:of )?(\d{{1,2}}|{_NUM_WORDS})\b"
    rf"|\b(?:is|(?:she|he|who|girl|boy|kid|child) s|was|turned|turns|turning|looks|looking|appears?) "
    rf"(?:about |around |like |only |just |barely )?"
    rf"(\d{{1,2}}|{_NUM_WORDS})(?= ?(?:$|[,;|.!?]))"
    r"|\b(\d{1,2}) ?(?:an[oñ]s|ans|jahre?n?|jahrig\w*|anni|lat|let|лет|года?)\b|(\d{1,2}) ?(?:歳|才|岁|살)"
)
_YOUTH_SEGMENT = re.compile("young|minor")
_AGE_SEGMENT = re.compile(rf"[1-9]|1[0-7]|{_NUM_WORDS}")  # "nude girl, 12"

# Deliberately absent (false positives on ordinary family/beach/school prompts): facial,
# bikini, swimsuit, bath, school uniform, shirtless, and bare cowgirl/missionary/doggy/milking/tentacle.
SEXUAL = re.compile(
    r"\b(?:nsfw|nud(?:e|es|ity|ist|ism|ie)|naked|unclothed|undress\w*|topless|bottomless|explicit|erotic\w*|"
    r"sensual|sex(?:y|ier|iest|ual\w*|ed|ting|bots?|toys?)?|sex ?bots?|sex ?toys?|seduc\w*|provocative|sultry|"
    r"suggestive|arous\w*|horny|lewd|ecchi|hentai|porn\w*|xxx|fuck\w*|fan ?service|risque|bedroom eyes|"
    r"come hither|gravure|playboy|onlyfans|nood[sz]?|nekk?id|nekked|nakey|seggs|secks|naturis[mt]s?|au naturel|"
    r"privates|private parts|buttocks|not safe for work|adults? (?:only|content)|x rated|rated x|humping|"
    r"(?:no|without|zero|removed?|minus) (?:clothes|clothing|garments)|(?:clothes|clothing) (?:off|removed)|"
    r"wearing nothing|nothing but (?:a |an )?(?:smile|towel|apron|socks|underwear|panties|stockings|pantyhose)|in the buff|birthday suit|skinny ?dipping|unclad|disrob\w*|strip ?tease|"
    r"strippers?(?<!wire stripper)(?<!wire strippers)(?<!paint stripper)|strip(?:ped|ping|s)? (?:naked|nude|down|off)|"
    r"bare (?:bottom|butt|ass)|naked apron|"
    r"(?:only|just) (?:pantyhose|stockings|an apron|apron|underwear|panties)|"
    r"(?:pantyhose|stockings|apron|underwear|panties) only|"
    r"nipples?|areola[es]?|(?<!chicken )(?<!turkey )breasts|boobs?|(?<!blue )(?<!great )(?<!coal )tits|titties|oppai|cleavage|"
    r"penis(?:es)?|cocks?|vaginas?|vulvas?|labia|clit|clitoris|genital(?:s|ia)?|anus|testicles|scrotum|"
    r"foreskin|pubic hair|pubes|pussy(?! ?cats?)|crotch|(?:ass|butt|groin|panty|pantsu|breast|nipple) focus|"
    r"intercourse|orgasm\w*|orgy|orgies|threesome|gang ?bang\w*|penetrat(?:ion|ed|ing)|"
    r"(?:blow|hand|foot|tit|rim|boob|thigh)jobs?|(?:face|tit|thigh|throat)fuck\w*|deep ?throat\w*|paizuri|"
    r"creampie|ahegao|cum|cums|cumming|cumshot|semen|sperm(?! whales?)|ejaculat\w*|bukkake|fellatio|cunnilingus|anal|"
    r"masturbat\w*|fingering|erections?|boners?|condoms?|dildos?|vibrators?|incest\w*|deflower\w*|virginity|"
    r"impregnat\w*|mating press|lap ?dance\w*|twerk\w*|making out|makeout|nip ?slip|wardrobe malfunction|"
    r"wet t shirt|grop(?:e|ed|es|ing)|fondl\w*|molest\w*|rap(?:e|ed|es|ing)|bondage|bdsm|fetish\w*|kinky?|"
    r"lingerie|boudoir|pinups?|pin up (?:girl|model|style|pose)s?|thong|panties|pantsu|panty ?shot|"
    r"upskirt|downblouse|camel ?toe|underwear|bra|spread legs|legs spread|spreading (?:her |his |their )?legs|"
    r"(?:micro|sling) ?bikini|see through|transparent clothing|uncensored|futanari|futa|missionary position|"
    r"cowgirl position|reverse cowgirl|doggy ?style|r ?18|rating ?:? ?(?:e|q|explicit|questionable))\b"
)
_SEXUAL_INTL = re.compile(
    r"\b(?:desnud[ao]s?|sexo|tetas|coño|follando|follar|nuas?|transando|nues?|nudite|sexe|baise\w*|erotique|"
    r"(?:enfant|fillette|fille|garcon|bebe|corps)s? nue?s?|tout nu|"
    r"nackt\w*|bruste|titten|ficken|nud[aei]|sesso|tette|scopare|scopando|seks\w*|"
    r"гол(?:ая|ые|ый|ой|енк\w*)|обнаж\w*|секс\w*|порн\w*|эрот\w*|сиськ\w*|хентай)\b|"
    r"エロ|セックス|裸(?!足)|ヌード|性的|エッチ(?!ング)|色情|おっぱい|乳首|中出し|性交|做爱|乳房|阴部|淫|누드|섹스|알몸|나체|야한|"
    r"\N{NO ONE UNDER EIGHTEEN SYMBOL}"
)
_AGE_UP = re.compile(
    r"\b(?:adults?|grown(?: ?ups?)?|grown (?:wo)?m[ae]n|of age|mature\w*|milf|old enough|over (?:18|eighteen)|"
    r"aged? up)\b|\b18 ?\+"
)
# A negative segment that asks to remove clothing or censoring.
_UNCLOTHE = re.compile(
    r"(?:(?:all|any|wearing|with|fully|full|some) )*(?:clothes|clothing|clothed|dressed|garments?|attire|apparel|"
    r"sfw|safe for work|(?:mosaic )?censor(?:ed|ing|ship|s)?(?: bars?)?|mosaic(?: censoring)?|modest(?:y)?|covered)"
)


def _unicode(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = "".join(c for c in s if c.isspace() or unicodedata.category(c) not in ("Cf", "Cc"))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if c in _KEEP_MARKS or unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s).casefold().translate(_FILLERS)


def _syntax(s: str) -> str:
    s = _LESS_THAN.sub(" under ", s)
    s = _PAREN_NUM.sub(r", \1,", s)  # "girl (12)" reads as an age, not as emphasis
    s = _LORA.sub(r" \1 ", s)
    s = _EMBEDDING.sub(" ", s)
    s = _WEIGHT.sub("", s)
    s = _BRACKETS.sub(" ", s)
    s = _BREAK.sub(", ", s)
    s = _INNER_SEP.sub(" ", s)
    return _SPACES.sub(" ", s).strip(" ,")


def _forms(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalized text, and native-script forms (Cyrillic and ñ kept, CJK runs rejoined) for the
    non-English lexicon."""
    s = str(text or "")
    if s.isascii():
        n = _syntax(_ASCII_CONTROL.sub("", s).lower())
        return n, (n,)
    u = _unicode(s)
    native = _syntax(u)
    return _syntax(u.translate(_FOLD)), tuple(dict.fromkeys((native, _CJK_GAP.sub("", native))))


def normalize(text: str) -> str:
    """Casefolded text with format chars, diacritics, lookalike letters and prompt syntax removed."""
    return _forms(text)[0]


def _unglue(m: re.Match[str]) -> str:
    return " ".join(g for g in m.groups() if g)


def _views(norm: str) -> tuple[str, ...]:
    """The normalized text plus leetspeak, spaced-letter, squeezed-repeat, split-compound and
    typo-corrected spellings."""
    bases = dict.fromkeys((norm, norm.translate(_LEET_I), norm.translate(_LEET_L), _GLUED_DIGITS.sub(_unglue, norm)))
    views = dict(bases)
    for v in bases:
        doubled = _REPEAT.sub(r"\1\1", v)
        views[_LETTER_RUN.sub(lambda m: _NON_LETTER.sub("", m.group()), v)] = None
        views[_REPEAT.sub(r"\1", v)] = None
        views[doubled] = None
        views[_respelled(doubled)] = None
    return tuple(views)


def _first(rx: re.Pattern[str], views: tuple[str, ...]) -> str:
    for v in views:
        m = rx.search(v)
        if m:
            return m.group(0)
    return ""


def _joined(views: tuple[str, ...]) -> dict[str, str]:
    """Roots split over 2-4 tokens ("lo li", "c h il d"), compared exactly so "solo lighting" stays clean."""
    found: dict[str, str] = {}
    for v in views:
        toks = _JOIN_TOKEN.findall(v)
        for i, tok in enumerate(toks):
            joined = tok
            for nxt in toks[i + 1 : i + 4]:
                if joined not in _JOIN_PREFIXES:
                    break
                joined += nxt
                if joined in _JOIN_ROOTS:
                    found.setdefault(_JOIN_ROOTS[joined], joined)
    return found


def _hard(natives: tuple[str, ...], views: tuple[str, ...], joined: dict[str, str]) -> str:
    return _first(HARD, views) or _first(_HARD_CJK, natives) or joined.get("hard", "")


def _age(texts: tuple[str, ...]) -> str:
    for v in texts:
        for m in _AGE.finditer(v):
            n = next(g for g in m.groups() if g)
            if not n.isdigit() or 0 < int(n) < 18:
                return m.group(0)
    return ""


def _youth(
    norm: str, natives: tuple[str, ...], views: tuple[str, ...], joined: dict[str, str], *, bare_age: bool
) -> str:
    segments = [s.strip() for s in _SEGMENTS.split(norm)]
    return (
        _first(YOUTH, views)
        or joined.get("youth", "")
        or _first(_YOUTH_INTL, (*natives, norm))
        or _age(tuple(dict.fromkeys((norm, *natives))))
        or next((s for s in segments if _YOUTH_SEGMENT.fullmatch(s) or (bare_age and _AGE_SEGMENT.fullmatch(s))), "")
    )


def _sexual(norm: str, natives: tuple[str, ...], views: tuple[str, ...], joined: dict[str, str]) -> str:
    return _first(SEXUAL, views) or _first(_SEXUAL_INTL, (*natives, norm)) or joined.get("sexual", "")


def is_sexual(text: str) -> bool:
    """True when ``text`` asks for sexual or nude content (any age)."""
    norm, natives = _forms(text)
    if not norm:
        return False
    views = _views(norm)
    return bool(_sexual(norm, natives, views, _joined(views)))


def _unclothing(norm: str) -> str:
    for v in _views(norm):
        for seg in _SEGMENTS.split(v):
            if _UNCLOTHE.fullmatch(seg.strip()):
                return seg.strip()
    return ""


def reasons(positive: str, negative: str = "", *, training: bool = False) -> tuple[str, ...]:
    """Why ``positive`` (with ``negative``) is prohibited; ``()`` means allowed.

    ``training=True`` applies the stricter data rule: any youth term blocks, at any rating.
    """
    norm, natives = _forms(positive)
    norm, natives = _ALLOW_PHRASES.sub(" ", norm), tuple(_ALLOW_PHRASES.sub(" ", n) for n in natives)
    if not norm.strip(" ,"):
        return ()
    views = _views(norm)
    joined = _joined(views)
    found: list[str] = []
    hard = _hard(natives, views, joined)
    if hard:
        found.append(f"hard:{hard}")
    youth = _youth(norm, natives, views, joined, bare_age=not training)
    sexual = _sexual(norm, natives, views, joined) if negative or not training else ""
    if youth and training:
        found.append(f"youth:{youth}")
    elif youth and sexual:
        found.append(f"youth+sexual:{youth}+{sexual}")
    neg = normalize(negative) if negative and (youth or sexual) else ""
    if neg and sexual:
        up = _first(_AGE_UP, _views(neg))
        if up:
            found.append(f"age_down_negative:{up}")
    if neg and youth and not training:
        off = _unclothing(neg)
        if off:
            found.append(f"youth+unclothing_negative:{youth}+{off}")
    return tuple(found)


def _refuse(found: tuple[str, ...]) -> ProhibitedContentError:
    return ProhibitedContentError(
        "Refused: sexual content involving minors is never generated or trained on "
        f"({', '.join(found)}). This block cannot be disabled."
    )


def require_allowed(positive: str, negative: str = "") -> NegativeText:
    """Raise :class:`ProhibitedContentError` or return the screened negative.

    Sexual prompts always get :data:`YOUTH_NEGATIVE` appended.
    """
    found = reasons(positive, negative)
    if found:
        raise _refuse(found)
    neg = str(negative or "")
    if YOUTH_NEGATIVE not in neg and is_sexual(positive):
        neg = f"{neg}, {YOUTH_NEGATIVE}" if neg.strip() else YOUTH_NEGATIVE
    return NegativeText(neg)


def _texts(groups):
    for g in groups:
        if isinstance(g, str):
            yield g
        elif isinstance(g, (list, tuple)):
            yield from _texts(g)


def require_encodable(*groups) -> None:
    """Backstop at the text encoder: every caption not marked :class:`NegativeText` must pass."""
    for text in _texts(groups):
        if not isinstance(text, NegativeText):
            found = reasons(text)
            if found:
                raise _refuse(found)


# Image-layer thresholds. Each FLAG is (min P(person), min P(child | person)) for a view.
# Calibrated with CLIP ViT-L/14 fp16 on 1,631 adult-tagged booru images from
# data/integration_smoke, 200 canny maps and synthetic noise/solid/gradient images. No image
# of a minor was or may be used, so recall is unmeasured and the thresholds lean toward
# refusing. Refusal rates on that adult content: rule34 2%, danbooru 6%, e621 (furry) 14%,
# canny maps 8%, synthetic 0%. "Teen" is a sink class, not counted as minor: CLIP ties it to
# adult "teen" porn and to the anime style (32-44% of the mass on the adult sets), and
# counting it refused 57-84% of adult content. Teens are left to the text guard.
NSFW_FLAG = 0.5
WHOLE_FLAG = (0.5, 0.5)
HALF_FLAG = (0.9, 0.8)

_AGE_TEMPLATES = (
    "a photo of {}.",
    "a close-up photo of {}.",
    "a full-body photo of {}.",
    "a candid snapshot of {}.",
    "a painting of {}.",
    "a drawing of {}.",
    "a digital illustration of {}.",
    "a 3d render of {}.",
    "an anime illustration of {}.",
    "a manga drawing of {}.",
    "a cartoon of {}.",
)
_AGE_CLASSES = (
    (
        "child",
        (
            "a child", "a little girl", "a little boy", "a young child", "a toddler", "a schoolchild",
            "a preteen girl", "a preteen boy",
        ),
    ),
    ("teen", ("a teenager", "a teenage girl", "a teenage boy", "a high school student", "a young adolescent")),
    (
        "adult",
        (
            "an adult woman", "an adult man", "a young adult woman", "a young adult man", "a woman in her twenties",
            "a man in his twenties", "a woman in her thirties", "a man in his forties", "a middle-aged woman",
            "a middle-aged man", "an elderly woman", "an elderly man", "an adult anthropomorphic furry character",
        ),
    ),
    (
        "none",
        (
            "an empty landscape", "an empty room", "a building", "a still life of objects", "food on a plate",
            "a vehicle", "an abstract pattern", "a page of text",
        ),
    ),
)  # fmt: skip


class ZeroShotAge:
    """Zero-shot CLIP age classifier with prompt ensembling (Radford et al., ICML 2021, sec. 3.1.4).

    Same recipe as the Q16 CLIP-prompt classifier (Schramowski et al., FAccT 2022): each class
    embedding is the normalized mean of its phrases under every photo/illustration/anime
    template. ``encode_text(list[str])`` and ``encode_images(list[HxWx3 uint8])`` return
    ``(N, D)`` CLIP embeddings.
    """

    def __init__(self, encode_text, encode_images, *, logit_scale: float = 100.0) -> None:
        import torch

        rows = []
        for _name, phrases in _AGE_CLASSES:
            emb = torch.nn.functional.normalize(
                encode_text([t.format(p) for p in phrases for t in _AGE_TEMPLATES]), dim=-1
            )
            rows.append(torch.nn.functional.normalize(emb.float().mean(0), dim=0))
        self.class_emb = torch.stack(rows)
        self.encode_images = encode_images
        self.logit_scale = float(logit_scale)

    def class_probs(self, views):
        """``(V, 4)`` softmax over (child, teen, adult, none) for square uint8 views."""
        import torch

        emb = torch.nn.functional.normalize(self.encode_images(views).float(), dim=-1)
        return (self.logit_scale * emb @ self.class_emb.to(emb.device).T).softmax(-1).cpu()

    def child_scores(self, images) -> list[tuple[float, float]]:
        """Per image, P(child | person) of the whole view and of the likelier half (0 without a person).

        Views are the whole image padded to a square plus its two halves along the long side,
        so a child beside an adult is not averaged away. Halves often crop to a torso, so they
        need the stricter :data:`HALF_FLAG`.
        """
        import torch

        views = [v for im in images for v in _age_views(im)]
        child, _teen, _adult, none = self.class_probs(views).view(len(images), 3, 4).unbind(-1)
        person = 1.0 - none
        share = child / person.clamp_min(1e-6)
        whole = torch.where(person[:, 0] >= WHOLE_FLAG[0], share[:, 0], 0.0)
        half = torch.where(person[:, 1:] >= HALF_FLAG[0], share[:, 1:], 0.0).amax(1)
        return list(zip(whole.tolist(), half.tolist(), strict=True))


def _square(a, fill: int = 127):
    import numpy as np

    h, w = a.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), fill, dtype=np.uint8)
    out[(s - h) // 2 : (s - h) // 2 + h, (s - w) // 2 : (s - w) // 2 + w] = a
    return out


def _age_views(a) -> list:
    h, w = a.shape[:2]
    halves = (a[:, : w // 2], a[:, w // 2 :]) if w >= h else (a[: h // 2], a[h // 2 :])
    return [_square(v) for v in (a, *halves)]


def _rgb_uint8(x):
    import numpy as np
    from PIL import Image

    if isinstance(x, (str, os.PathLike)):
        with Image.open(x) as im:
            return np.asarray(im.convert("RGB"))
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    a = np.asarray(x)
    if a.dtype != np.uint8:
        a = np.clip(np.nan_to_num(a.astype(np.float32)), 0.0, 255.0).round().astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return np.ascontiguousarray(a[..., :3])


def _image_list(images) -> list:
    """One image or a sequence of paths / PIL images / HxWx3 arrays; empty or missing paths are dropped."""
    import numpy as np
    from PIL import Image

    if isinstance(images, (str, os.PathLike, Image.Image)) or (isinstance(images, np.ndarray) and images.ndim < 4):
        images = [images]
    return [x for x in images if not isinstance(x, (str, os.PathLike)) or (str(x).strip() and os.path.isfile(x))]


def _cached_weights(src: str) -> bool:
    """``src`` is a local folder with weights, or a hub id whose weights are already in the HF cache.

    Never triggers a download.
    """
    if os.path.isdir(src):
        from utils.modeling.model_paths import has_local_weights

        return has_local_weights(src)
    from huggingface_hub import try_to_load_from_cache

    return any(isinstance(try_to_load_from_cache(src, f), str) for f in ("model.safetensors", "pytorch_model.bin"))


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.cache
def _age_model() -> ZeroShotAge | None:
    """CLIP ViT-L/14 from ``pretrained/CLIP-ViT-L-14`` or the HF cache; ``None`` when absent."""
    from utils.modeling.model_paths import model_dir, resolve_model_path_require_weights

    weights = resolve_model_path_require_weights("CLIP-ViT-L-14", "openai/clip-vit-large-patch14")
    if not _cached_weights(weights):
        return None
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from utils.generation.guidance import _clip_feature_tensor

    # The folder may hold only tokenizer/config files while the weights sit in the HF cache.
    local = model_dir() / "CLIP-ViT-L-14"
    proc_src = (
        str(local)
        if (local / "tokenizer.json").is_file() and (local / "preprocessor_config.json").is_file()
        else weights
    )
    proc = CLIPProcessor.from_pretrained(proc_src, local_files_only=True)
    device = _device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained(weights, local_files_only=True, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def encode_text(texts):
        tok = proc.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        return _clip_feature_tensor(model.get_text_features(**tok)).float()

    @torch.inference_mode()
    def encode_images(views):
        px = proc.image_processor(images=list(views), return_tensors="pt")["pixel_values"].to(device, dtype)
        return _clip_feature_tensor(model.get_image_features(pixel_values=px)).float()

    return ZeroShotAge(encode_text, encode_images, logit_scale=float(model.logit_scale.detach().exp()))


def _nsfw_scorer():
    """``utils.modeling.hf_loaders.score_nsfw_probability`` when its weights are local, else ``None``."""
    from utils.modeling.hf_loaders import score_nsfw_probability
    from utils.modeling.model_paths import default_nsfw_detector_path

    if not _cached_weights(default_nsfw_detector_path()):
        return None
    device = _device()
    return lambda rgb: score_nsfw_probability(rgb, device=device)


def require_image_allowed(images, *, sexual: bool, conditioning: bool = False) -> None:
    """Raise :class:`ProhibitedContentError` when a sexual image likely depicts a minor.

    ``sexual`` says whether the prompt is sexual (:func:`is_sexual`). An output image of a
    non-sexual prompt is checked only when the NSFW classifier, if its weights are local,
    gives a probability above :data:`NSFW_FLAG`. Conditioning images (``conditioning=True``)
    are checked only for sexual prompts. Without the CLIP weights, sexual conditioning is
    refused, while outputs proceed with a warning, since the text guard and
    :data:`YOUTH_NEGATIVE` still apply.

    An image is refused when its whole view passes :data:`WHOLE_FLAG` or either half passes
    :data:`HALF_FLAG` (see the calibration note above them).
    """
    if conditioning and not sexual:
        return
    nsfw = None if sexual else _nsfw_scorer()
    if not sexual and nsfw is None:
        return
    rgbs = [_rgb_uint8(x) for x in _image_list(images)]
    if nsfw is not None:
        rgbs = [im for im in rgbs if (nsfw(im) or 0.0) > NSFW_FLAG]
    if not rgbs:
        return
    age = _age_model()
    if age is None:
        missing = (
            "the CLIP ViT-L/14 age check has no weights (pretrained/CLIP-ViT-L-14 or the "
            "openai/clip-vit-large-patch14 Hugging Face cache)"
        )
        if conditioning:
            raise ProhibitedContentError(
                f"Refused: a sexual prompt with conditioning images needs the image age check, and {missing}. "
                "Drop the conditioning images or install those weights."
            )
        warnings.warn(f"Image age check skipped: {missing}; text guard still active.", RuntimeWarning, stacklevel=2)
        return
    for whole, half in age.child_scores(rgbs):
        if whole >= WHOLE_FLAG[1] or half >= HALF_FLAG[1]:
            raise ProhibitedContentError(
                "Refused: a sexual image likely depicts a minor "
                f"(P(child | person) whole={whole:.2f}, half={half:.2f}). This block cannot be disabled."
            )
