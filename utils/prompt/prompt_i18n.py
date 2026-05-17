"""
Lightweight i18n for prompt helper suggestions.

Goal:
- Let ``--lang`` accept many languages/dialects.
- Provide localized *generic* suggestion messages when we don't have
  fully translated per-category text.

Custom i18n:
- Users can still override with ``--suggestions-json`` in the tools.
"""

from __future__ import annotations

from typing import List


def normalize_lang_code(lang: str) -> str:
    if not lang:
        return "en"
    s = str(lang).strip().lower()
    if not s:
        return "en"

    if s in ("en", "eng", "english"):
        return "en"

    if s in ("zh", "zh-cn", "zh_hans", "zh-hans", "中文", "zh_cn"):
        return "zh"
    if s in ("zh-tw", "zh_hant", "zh-hant", "zh_tw"):
        return "zh"

    if s.startswith("es"):
        return "es"
    if s.startswith("fr"):
        return "fr"
    if s.startswith("ru"):
        return "ru"
    if s.startswith("ja"):
        return "ja"
    if s.startswith("it"):
        return "it"
    if s.startswith("ar"):
        return "ar"
    if s in ("la", "lat", "latin"):
        return "la"

    if s.startswith("hi"):
        return "hi"
    if s.startswith("bn"):
        return "bn"
    if s.startswith("ta"):
        return "ta"
    if s.startswith("te"):
        return "te"
    if s.startswith("mr"):
        return "mr"
    if s.startswith("kn"):
        return "kn"
    if s.startswith("ml"):
        return "ml"
    if s.startswith("gu"):
        return "gu"
    if s.startswith("pa"):
        return "pa"
    if s.startswith("ur"):
        return "ur"

    return "en"


def _examples_to_string(examples: List[str], limit: int = 6) -> str:
    cleaned = [e for e in examples if e]
    if not cleaned:
        return ""
    return ", ".join(cleaned[:limit])


def generic_suggestion(category: str, examples: List[str], lang: str) -> str:
    lang_code = normalize_lang_code(lang)
    ex = _examples_to_string(examples)
    ex_part = f" (e.g., {ex})" if ex else ""

    templates = {
        "en": "Add related keyword terms for '{category}' early in the caption and keep it consistent across the dataset."
        + ex_part,
        "es": "Para reforzar '{category}', agrega terminos relacionados al inicio del caption y mantente consistente en todo el dataset."
        + ex_part,
        "fr": "Pour renforcer '{category}', ajoutez des termes lies des le debut des captions et gardez la coherence dans tout le dataset."
        + ex_part,
        "ru": "chtoby usilit '{category}', dobavlyayte svyazannye slova v nachale caption i derzhite yedinstvo po vsemu datasetu."
        + ex_part,
        "ja": "tame ni '{category}' o tsuyokusuru: kyappushon no hayai tokoro ni kanren kyuudo o irete, zenkoku de ichiji ni shite kudasai."
        + ex_part,
        "zh": "xiangqianghua '{category}', qing zai caption de zaoqi bufen fangru guanlian keyici, bing zai quanbu shuju zhong baochi yizhi."
        + ex_part,
        "hi": "'{category}' ko behtar banane ke liye, caption ki shuruaat me sambandhit keywords add karein aur poore dataset me consistent rakhein."
        + ex_part,
        "bn": "'{category}' aro bhalo korte, caption-er shuru te sambandhito keywords add korun ebong poora dataset-e consistent rakhen."
        + ex_part,
        "ta": "'{category}'-ai balappadutha, captions-in aarambathile sambandhamaana keywords serthu, muzhum dataset-il ore maadhiri vaitthukollungal."
        + ex_part,
        "te": "'{category}' ni balam cheyyadaniki, caption start-lo sambandhit keywords add cheyyandi, mariyu motta dataset-lo consistency unchandi."
        + ex_part,
        "mr": "'{category}' majboot karanyasathi, caption cha suruvatichya pankhtit sambandhit keywords joda ani poora dataset madhye consistency theva."
        + ex_part,
        "kn": "'{category}' anu balapaadalu, caption shuruvall-e sambandhita keywords add maadi mattu poorti dataset-alli consistency irisi."
        + ex_part,
        "ml": "'{category}' balam aakkikan, caption-inde aarambhathil sambandhita keywords cherkkuka, poora dataset-ile consistency paalikuka."
        + ex_part,
        "gu": "'{category}' ne majboot banava mate, caption ni shuruaat ma sambandhit keywords jodo ane poora dataset ma consistency rakhjo."
        + ex_part,
        "pa": "'{category}' nu behtar banan layi, caption di shuruaat vich sambandhit keywords jodo te poore dataset vich consistency rakho."
        + ex_part,
        "ur": "'{category}' ko behtar banane ke liye, caption ke shuru me mutalliq keywords shamil karein aur poore dataset me consistency rakhein."
        + ex_part,
        "it": "Per rafforzare '{category}', aggiungi termini correlati all'inizio dei caption e mantieni coerenza in tutto il dataset."
        + ex_part,
        "ar": "litaqwiyat '{category}', adif kalimat muhtamimah fi awwal al-caption wa ahtafidh bi-al-itqan fi jami' al-dataset."
        + ex_part,
        "la": "Ad '{category}' roborandum, adde vocabula coniuncta primo in captione et eadem manere per totam tuam materiam."
        + ex_part,
    }

    return templates.get(lang_code, templates["en"]).format(category=category)
