"""
Lightweight i18n for prompt helper suggestions.

Goal:
- Let `--lang` accept many languages/dialects.
- Provide localized *generic* suggestion messages when we don't have
  fully translated per-category text.

Custom i18n:
- Users can still override with `--suggestions-json` in the tools.
"""

from __future__ import annotations

from typing import List


def normalize_lang_code(lang: str) -> str:
    if not lang:
        return "en"
    s = str(lang).strip().lower()
    if not s:
        return "en"

    # English
    if s in ("en", "eng", "english"):
        return "en"

    # Chinese (keep a single template; simplified/traditional users can override with --suggestions-json)
    if s in ("zh", "zh-cn", "zh_hans", "zh-hans", "中文", "zh_cn"):
        return "zh"
    if s in ("zh-tw", "zh_hant", "zh-hant", "zh_tw"):
        return "zh"

    # Spanish (Spain + Mexico + other)
    if s.startswith("es"):
        return "es"

    # French
    if s.startswith("fr"):
        return "fr"

    # Russian
    if s.startswith("ru"):
        return "ru"

    # Japanese
    if s.startswith("ja"):
        return "ja"

    # Italian
    if s.startswith("it"):
        return "it"

    # Arabic
    if s.startswith("ar"):
        return "ar"

    # Latin
    if s in ("la", "lat", "latin"):
        return "la"

    # Indian / South Asian common language codes
    # (Users can override with --suggestions-json for precise translations.)
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

    # Unknown -> English
    return "en"


def _examples_to_string(examples: List[str], limit: int = 6) -> str:
    cleaned = [e for e in examples if e]
    if not cleaned:
        return ""
    return ", ".join(cleaned[:limit])


def generic_suggestion(category: str, examples: List[str], lang: str) -> str:
    l = normalize_lang_code(lang)
    ex = _examples_to_string(examples)
    ex_part = f" (e.g., {ex})" if ex else ""

    # ASCII-only templates to avoid UnicodeEncodeError on Windows consoles.
    # Your environment appears to use cp1252; non-ascii output can crash or render garbled.
    ascii_templates = {
        "en": "Add related keyword terms for '{category}' early in the caption and keep it consistent across the dataset." + ex_part,
        "es": "Para reforzar '{category}', agrega terminos relacionados al inicio del caption y mantente consistente en todo el dataset." + ex_part,
        "fr": "Pour renforcer '{category}', ajoutez des termes lies des le debut des captions et gardez la coherence dans tout le dataset." + ex_part,
        "ru": "chtoby usilit '{category}', dobavlyayte svyazannye slova v nachale caption i derzhite yedinstvo po vsemu datasetu." + ex_part,
        "ja": "tame ni '{category}' o tsuyokusuru: kyappushon no hayai tokoro ni kanren kyuudo o irete, zenkoku de ichiji ni shite kudasai." + ex_part,
        "zh": "xiangqianghua '{category}', qing zai caption de zaoqi bufen fangru guanlian keyici, bing zai quanbu shuju zhong baochi yizhi." + ex_part,
        "hi": "'{category}' ko behtar banane ke liye, caption ki shuruaat me sambandhit keywords add karein aur poore dataset me consistent rakhein." + ex_part,
        "bn": "'{category}' aro bhalo korte, caption-er shuru te sambandhito keywords add korun ebong poora dataset-e consistent rakhen." + ex_part,
        "ta": "'{category}'-ai balappadutha, captions-in aarambathile sambandhamaana keywords serthu, muzhum dataset-il ore maadhiri vaitthukollungal." + ex_part,
        "te": "'{category}' ni balam cheyyadaniki, caption start-lo sambandhit keywords add cheyyandi, mariyu motta dataset-lo consistency unchandi." + ex_part,
        "mr": "'{category}' majboot karanyasathi, caption cha suruvatichya pankhtit sambandhit keywords joda ani poora dataset madhye consistency theva." + ex_part,
        "kn": "'{category}' anu balapaadalu, caption shuruvall-e sambandhita keywords add maadi mattu poorti dataset-alli consistency irisi." + ex_part,
        "ml": "'{category}' balam aakkikan, caption-inde aarambhathil sambandhita keywords cherkkuka, poora dataset-ile consistency paalikuka." + ex_part,
        "gu": "'{category}' ne majboot banava mate, caption ni shuruaat ma sambandhit keywords jodo ane poora dataset ma consistency rakhjo." + ex_part,
        "pa": "'{category}' nu behtar banan layi, caption di shuruaat vich sambandhit keywords jodo te poore dataset vich consistency rakho." + ex_part,
        "ur": "'{category}' ko behtar banane ke liye, caption ke shuru me mutalliq keywords shamil karein aur poore dataset me consistency rakhein." + ex_part,
        "it": "Per rafforzare '{category}', aggiungi termini correlati all'inizio dei caption e mantieni coerenza in tutto il dataset." + ex_part,
        "ar": "litaqwiyat '{category}', adif kalimat muhtamimah fi awwal al-caption wa ahtafidh bi-al-itqan fi jami' al-dataset." + ex_part,
        "la": "Ad '{category}' roborandum, adde vocabula coniuncta primo in captione et eadem manere per totam tuam materiam." + ex_part,
    }

    return ascii_templates.get(l, ascii_templates["en"]).format(category=category)

    templates = {
        "en": "Add related keyword terms for '{category}' early in the caption and keep it consistent across the dataset." + ex_part,
        "es": "Para reforzar '{category}', añade términos relacionados al inicio del caption y mantén la consistencia en todo el dataset." + ex_part,
        "fr": "Pour renforcer '{category}', ajoutez des termes liés dès le début des captions et gardez la cohérence dans tout le dataset." + ex_part,
        "ru": "Чтобы усилить '{category}', добавляйте связанные термины в начале caption и сохраняйте единообразие по всему датасету." + ex_part,
        "ja": "'{category}' を強化するには、キャプションの早い段階に関連キーワードを入れ、データセット全体で一貫させてください。" + ex_part,
        "zh": "想强化 '{category}'，请尽量把相关关键词放在 caption 的前半段，并在整个数据集中保持一致。" + ex_part,
        "hi": "‘{category}’ को बेहतर बनाने के लिए कैप्शन की शुरुआत में संबंधित कीवर्ड जोड़ें और पूरे डेटासेट में एकरूपता रखें।" + ex_part,
        "bn": "‘{category}’ আরও ভালো করতে ক্যাপশনের শুরুতেই সম্পর্কিত কীওয়ার্ড যোগ করুন এবং পুরো ডেটাসেটজুড়ে সামঞ্জস্য রাখুন।" + ex_part,
        "ta": "'{category}' யை வலுப்படுத்த, கேப்ஷனின் ஆரம்பத்திலேயே தொடர்புடைய முக்கிய சொற்களை சேர்த்து, முழு டேட்டாசெட்டிலும் ஒரே மாதிரி வைத்திருங்கள்。" + ex_part,
        "te": "'{category}' ను బలపర్చడానికి, క్యాప్షన్ ప్రారంభంలో సంబంధిత కీవర్డ్స్ జోడించి, మొత్తం డేటాసెట్‌లో స్థిరంగా ఉంచండి。" + ex_part,
        "mr": "'{category}' मजबूत करण्यासाठी कॅप्शनच्या सुरुवातीलाच संबंधित कीवर्ड जोडा आणि संपूर्ण डेटासेटमध्ये सातत्य ठेवा।" + ex_part,
        "kn": "'{category}' ಅನ್ನು ಬಲಪಡಿಸಲು, ಕ್ಯಾಪ್ಷನ್‌ನ ಆರಂಭದಲ್ಲೇ ಸಂಬಂಧಿತ ಕೀವರ್ಡ್‌ಗಳನ್ನು ಸೇರಿಸಿ ಮತ್ತು ಸಂಪೂರ್ಣ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಏಕತೆ ಇಟ್ಟುಕೊಳ್ಳಿ。" + ex_part,
        "ml": "'{category}' ശക്തമാക്കാൻ, ക്യാപ്ഷന്റെ ആദിയിൽ ബന്ധപ്പെട്ട കീവേഡുകൾ ചേർത്ത് ഡാറ്റാസെറ്റിലുടനീളം സ്ഥിരത പാലിക്കുക。" + ex_part,
        "gu": "'{category}' ને મજબૂત કરવા માટે કેપ્શનની શરૂઆતમાં સંબંધિત કીવર્ડ્સ ઉમેરો અને આખા ડેટાસેટમાં સતતતા રાખો。" + ex_part,
        "pa": "‘{category}’ ਨੂੰ ਬਿਹਤਰ ਬਣਾਉਣ ਲਈ ਕੈਪਸ਼ਨ ਦੀ ਸ਼ੁਰੂਆਤ ਵਿੱਚ ਸੰਬੰਧਿਤ ਕੀਵਰਡ ਜੋੜੋ ਅਤੇ ਪੂਰੇ ਡਾਟਾਸੈਟ ਵਿੱਚ ਲਗਾਤਾਰਤਾ ਰੱਖੋ।" + ex_part,
        "ur": "‘{category}’ کو بہتر کرنے کے لیے کیپشن کے شروع میں متعلقہ keywords شامل کریں اور پورے ڈیٹاسیٹ میں مستقل مزاجی رکھیں۔" + ex_part,
        "it": "Per rafforzare '{category}', aggiungi termini correlati all'inizio dei caption e mantieni coerenza in tutto il dataset." + ex_part,
        "ar": "لتقوية '{category}'، أضف كلمات مفتاحية مرتبطة في بداية الـcaption واحتفظ بالاتساق عبر مجموعة البيانات كلها." + ex_part,
        "la": "Ad '{category}' roborandum, adde vocabula coniuncta primo in captione et eadem manere per totam tuam materiam." + ex_part,
    }

    return templates.get(l, templates["en"]).format(category=category)

