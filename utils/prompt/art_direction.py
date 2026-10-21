"""Art direction planner: prompt -> explicit shot, lens, lighting and palette decisions.

Plan-then-generate (LayoutGPT, Feng et al. NeurIPS 2023; RPG, Yang et al. ICML 2024): decide
the picture before sampling instead of echoing keywords. Cues the user wrote are parsed and
marked explicit; open fields are filled from the defaults photographers, cinematographers and
painters use for that subject (portrait: 85 mm, loop light from high front-side, 3:1 ratio;
landscape: wide lens, deep focus, horizon on a third, low raking sun; noir: hard split light,
8:1). When the prompt leaves a choice open between equally good options (which third, which
time of day), the seed picks, so different seeds explore different shots instead of all
collapsing to the same golden-hour centred frame.

``plan_art_direction`` returns an :class:`ArtDirection`; ``caption()`` is the conditioning text
(only decided fields, phrased for the medium: no lens terms for paintings), ``to_dict()`` the
plan critics score against (``utils.quality.art_critic`` keys: grade, key, composition,
harmony, contrast, medium, pixel_art). ``gate_tag`` drops pack text that contradicts the plan
(skin and hands with no people, camera terms on a painting).
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from config.defaults.art_mediums import detect_all_mediums

__all__ = [
    "ArtDirection",
    "apply_art_direction_to_args",
    "gate_tag",
    "mentions_people",
    "plan_art_direction",
]


def _rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.I)


_PEOPLE = _rx(
    r"person|persons|people|man|men|woman|women|girls?|boys?|lady|ladies|guy|gentleman|child(?:ren)?|kids?",
    r"baby|toddler|teen\w*|family|couple|crowd|figures?|characters?|portrait|selfie|faces?|headshot|model",
    r"bride|groom|[123](?:girl|boy)s?|solo|humanoid|android|cyborg|astronaut|soldiers?|warriors?|knights?",
    r"samurai|ninja|wizard|witch|sorcer\w+|mage|monk|priest|nun|king|queen|prince|princess|detective",
    r"pilot|nurse|doctor|chef|cowboy|pirate|student|teacher|dancers?|singer|musician|athlete|player|rider",
    r"hunter|farmer|workers?|fisherm[ae]n|police\w*|business\w*|elder|grand(?:mother|father)|mother|father",
    r"daughter|son|friends|pedestrians?|shoppers|tourists|commuters|vendor|merchant|travell?er|hiker",
    r"climber|surfer|skater|elf|elves|dwarf|orc|vampire|angel|goddess|hero(?:ine)?|villain|adventurer",
    r"explorer|scientist|engineer|reader|nude|naked|body|hands?|anatomy",
)
_GROUP = _rx(
    r"people|crowd|family|couple|group|friends|team|band|pedestrians|men|women|children|kids|[23](?:girl|boy)s",
    r"duo|trio|soldiers|dancers|shoppers|commuters|tourists",
)
_PET = _rx(r"cats?|kittens?|dogs?|pupp(?:y|ies)|rabbits?|bunn(?:y|ies)|hamsters?|parrots?|guinea pigs?")
_CREATURE = _rx(
    r"animals?|cats?|kittens?|dogs?|pupp(?:y|ies)|fox(?:es)?|wol(?:f|ves)|bears?|lions?|tigers?|deer|horses?",
    r"cows?|sheep|rabbits?|bunn(?:y|ies)|squirrels?|owls?|eagles?|hawks?|birds?|parrots?|ducks?|swans?",
    r"penguins?|fish|whales?|dolphins?|sharks?|octopus|turtles?|frogs?|snakes?|lizards?|insects?|bees?",
    r"butterfl(?:y|ies)|moths?|beetles?|ants?|dragons?|creatures?|monsters?|beasts?|unicorns?",
    r"griffins?|phoenix|dinosaurs?|elephants?|giraffes?|zebras?|monkeys?|apes?|gorillas?|pandas?|koalas?",
    r"hedgehogs?|mice|mouse|rats?|hamsters?",
)
_FIGURE = _rx(r"robots?|mecha|mascots?|statues?|sculptures?|figurines?|dolls?|toys?|golems?|automatons?")
_ACTION = _rx(
    r"standing|walking|running|dancing|jumping|fighting|leaping|striding|full[- ]body|full[- ]length",
    r"skating|surfing|climbing|riding",
)
# Ordered: first matching kind wins after people / creature checks.
_KIND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("macro", _rx(r"macro", r"extreme close-?up", r"dew ?drops?", r"droplets?", r"snowflakes?", r"spider ?web")),
    ("product", _rx(r"product", r"packshot", r"catalog(?:ue)?", r"e-?commerce", r"advert\w*", r"commercial shot")),
    ("food", _rx(r"food", r"dish", r"meal", r"ramen", r"sushi", r"pizza", r"burger", r"pasta", r"salad", r"soup")),
    (
        "food",
        _rx(r"cake", r"dessert", r"bread", r"coffee", r"latte", r"cocktail", r"breakfast", r"bowl of", r"plate of"),
    ),
    ("still_life", _rx(r"still[- ]life", r"vase of", r"bouquet", r"arrangement of")),
    ("vehicle", _rx(r"(?:sports )?cars?", r"trucks?", r"motorcycles?", r"motorbikes?", r"bicycles?", r"locomotives?")),
    ("vehicle", _rx(r"airplanes?", r"aircraft", r"jets?", r"helicopters?", r"ships?", r"boats?", r"yachts?")),
    ("vehicle", _rx(r"spaceships?", r"starships?", r"submarines?", r"tanks?", r"buses", r"vans?")),
    ("interior", _rx(r"interior", r"rooms?", r"kitchen", r"bedroom", r"bathroom", r"hallway", r"corridor")),
    ("interior", _rx(r"library", r"office", r"restaurant", r"lobby", r"attic", r"reading nook")),
    ("architecture", _rx(r"architect\w*", r"buildings?", r"facade", r"skyscrapers?", r"towers?", r"cathedral")),
    ("architecture", _rx(r"church", r"temple", r"museum", r"house", r"villa", r"bridge", r"palace", r"brutalist")),
    ("street", _rx(r"street", r"streets", r"alley", r"crosswalk", r"intersection", r"market", r"sidewalk")),
    ("city", _rx(r"city", r"cityscape", r"skyline", r"downtown", r"town", r"village", r"harbou?r", r"metropolis")),
    (
        "landscape",
        _rx(
            r"landscape",
            r"mountains?",
            r"valley",
            r"lakes?",
            r"rivers?",
            r"waterfall",
            r"forest",
            r"woods",
            r"garden",
            r"park",
            r"meadow",
            r"fields?",
            r"desert",
            r"dunes?",
            r"canyon",
            r"coast(?:line)?",
            r"beach",
            r"ocean",
            r"sea(?:scape)?",
            r"shore",
            r"cliffs?",
            r"island",
            r"glacier",
            r"hills?",
            r"countryside",
            r"sky",
            r"aurora",
            r"jungle",
            r"volcano",
            r"mars",
            r"planets?",
            r"lunar",
            r"lighthouse",
            r"castle",
            r"ruins?",
            r"trees?",
        ),
    ),
)

# Genre defaults: shot, angle, placement, focal mm, f-number, key azimuth/elevation (deg), hard key, key:fill.
_KIND_DEFAULTS: dict[str, dict[str, Any]] = {
    "portrait": dict(shot="close_up", angle="eye_level", placement="thirds", focal=85, f=2.0, az=45, el=35, ratio=3.0),
    "person": dict(shot="medium", angle="eye_level", placement="thirds", focal=50, f=2.8, az=45, el=30, ratio=3.0),
    "figure": dict(shot="medium", angle="eye_level", placement="thirds", focal=85, f=4.0, az=45, el=35, ratio=3.0),
    "group": dict(shot="medium_wide", angle="eye_level", placement="center", focal=35, f=5.6, az=30, el=30, ratio=2.0),
    "animal": dict(shot="medium", angle="eye_level", placement="thirds", focal=300, f=4.0, az=45, el=20, ratio=2.5),
    "macro": dict(shot="macro", angle="eye_level", placement="thirds", focal=100, f=8.0, az=60, el=40, ratio=2.0),
    "product": dict(shot="close_up", angle="high", placement="center", focal=100, f=11.0, az=45, el=50, ratio=2.0),
    "food": dict(shot="close_up", angle="high", placement="thirds", focal=85, f=4.0, az=150, el=30, ratio=3.0),
    "still_life": dict(shot="close_up", angle="high", placement="thirds", focal=85, f=8.0, az=60, el=40, ratio=4.0),
    "vehicle": dict(shot="wide", angle="low", placement="thirds", focal=50, f=8.0, az=60, el=25, ratio=3.0),
    "interior": dict(shot="wide", angle="eye_level", placement="symmetry", focal=20, f=8.0, az=80, el=20, ratio=3.0),
    "architecture": dict(shot="wide", angle="eye_level", placement="thirds", focal=24, f=11.0, az=70, el=25, ratio=4.0),
    "street": dict(shot="medium_wide", angle="eye_level", placement="thirds", focal=35, f=8.0, az=60, el=30, ratio=3.0),
    "city": dict(shot="extreme_wide", angle="high", placement="thirds", focal=35, f=8.0, az=80, el=15, ratio=3.0),
    "landscape": dict(
        shot="extreme_wide", angle="eye_level", placement="thirds", focal=24, f=11.0, az=90, el=10, ratio=4.0
    ),
    "scene": dict(shot="wide", angle="eye_level", placement="thirds", focal=35, f=5.6, az=45, el=30, ratio=3.0),
}
_OUTDOOR = frozenset({"landscape", "city", "street", "architecture", "vehicle", "animal"})

# Light by time of day: sun elevation (deg), colour temperature (K), hard sun, key:fill, palette harmony, grade.
_TIMES: dict[str, dict[str, Any]] = {
    "dawn": dict(el=4, kelvin=4300, hard=False, ratio=2.5, harmony="analogous", grade="cool"),
    "morning": dict(el=20, kelvin=5000, hard=True, ratio=3.0, harmony="analogous", grade=None),
    "midday": dict(el=65, kelvin=5600, hard=True, ratio=5.0, harmony="analogous", grade=None),
    "afternoon": dict(el=30, kelvin=5200, hard=True, ratio=3.5, harmony="analogous", grade="warm"),
    "golden_hour": dict(el=6, kelvin=3500, hard=True, ratio=4.0, harmony="complementary", grade="warm"),
    "blue_hour": dict(el=-4, kelvin=9000, hard=False, ratio=3.0, harmony="complementary", grade="cool"),
    "night": dict(el=30, kelvin=4100, hard=True, ratio=8.0, harmony="complementary", grade="cool"),
    "overcast": dict(el=60, kelvin=6500, hard=False, ratio=1.5, harmony="analogous", grade="muted"),
}
_TIME_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("golden_hour", _rx(r"golden hour", r"sunset", r"sundown", r"magic hour", r"setting sun", r"late afternoon")),
    ("blue_hour", _rx(r"blue hour", r"twilight", r"dusk")),
    ("dawn", _rx(r"dawn", r"sunrise", r"daybreak", r"first light", r"early morning")),
    ("night", _rx(r"night\w*", r"midnight", r"moonlit", r"moonlight", r"starry", r"nocturnal")),
    ("overcast", _rx(r"overcast", r"cloudy", r"rain\w*", r"fog\w*", r"mist\w*", r"haz[ey]", r"drizzle", r"snowfall")),
    ("midday", _rx(r"noon", r"midday", r"harsh sun\w*", r"high sun")),
    ("afternoon", _rx(r"afternoon")),
    ("morning", _rx(r"morning")),
)
# Open time-of-day choices per kind (seeded), so outdoor scenes are not all golden hour.
_TIME_CHOICES: dict[str, tuple[str, ...]] = {
    "landscape": ("golden_hour", "dawn", "overcast", "afternoon"),
    "city": ("blue_hour", "golden_hour", "overcast"),
    "street": ("overcast", "afternoon", "blue_hour"),
    "architecture": ("afternoon", "golden_hour", "overcast"),
    "vehicle": ("golden_hour", "overcast", "blue_hour"),
    "animal": ("morning", "overcast", "golden_hour"),
    "macro": ("morning", "overcast"),
}

_PATTERN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rembrandt", _rx(r"rembrandt")),
    ("butterfly", _rx(r"butterfly light\w*", r"paramount", r"beauty dish")),
    ("split", _rx(r"split light\w*", r"half[- ]lit face", r"side[- ]lit")),
    ("loop", _rx(r"loop light\w*")),
    ("rim", _rx(r"rim light\w*", r"back ?lit", r"backlight\w*", r"edge light\w*", r"contre[- ]jour", r"silhouette")),
    ("broad", _rx(r"broad light\w*", r"flat light\w*", r"even light\w*")),
    ("chiaroscuro", _rx(r"chiaroscuro", r"tenebris\w*")),
    ("three_point", _rx(r"three[- ]point light\w*", r"studio light\w*", r"softbox")),
)
_PATTERN_GEOMETRY = {  # azimuth (0 = from camera), elevation
    "rembrandt": (45, 40),
    "loop": (35, 35),
    "butterfly": (0, 45),
    "split": (90, 15),
    "rim": (150, 20),
    "broad": (20, 25),
    "chiaroscuro": (70, 35),
    "three_point": (45, 35),
}
_SOURCE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("window", _rx(r"windows?", r"skylight")),
    ("candle", _rx(r"candle\w*")),
    ("fire", _rx(r"fire\w*", r"campfire", r"hearth", r"torch\w*", r"lantern\w*")),
    ("neon", _rx(r"neon", r"cyberpunk")),
    ("lamp", _rx(r"lamps?", r"lamplight", r"desk lamp", r"practical\w*", r"bulbs?")),
    ("streetlight", _rx(r"street ?lights?", r"street ?lamps?")),
    ("screen", _rx(r"screens?", r"monitors?", r"phone glow")),
    ("moon", _rx(r"moon\w*")),
)
_SOURCE_KELVIN = {"candle": 1900, "fire": 2000, "lamp": 2800, "streetlight": 2200, "neon": 6500, "window": 5600}

_GRADE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("teal_orange", re.compile(r"\bteal[\s-]*(?:and|&|/)?[\s-]*orange\b|\borange[\s-]*(?:and|&|/)[\s-]*teal\b", re.I)),
    ("mono", _rx(r"black[\s-]*and[\s-]*white", r"b&w", r"monochrome", r"gr[ae]yscale", r"film noir", r"sepia")),
    ("neon", _rx(r"neon", r"cyberpunk", r"synthwave", r"vaporwave")),
    ("vivid", _rx(r"vibrant", r"vivid", r"saturated", r"psychedelic", r"pop art", r"technicolou?r", r"kodachrome")),
    ("muted", _rx(r"muted", r"desaturated", r"pastel", r"faded", r"bleach bypass", r"washed[\s-]out")),
    ("mixed", _rx(r"cinestill", r"mixed lighting")),
    ("warm", _rx(r"warm (?:tones?|grade|light|palette)", r"amber", r"tungsten", r"portra")),
    ("cool", _rx(r"cool (?:tones?|grade|light|palette)", r"icy", r"cold light")),
)
_HARMONY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("split", _rx(r"split[\s-]*complementary")),
    ("complementary", _rx(r"complementary", r"teal[\s-]*(?:and|&)?[\s-]*orange", r"blue and orange")),
    ("triadic", _rx(r"triadic")),
    ("analogous", _rx(r"analogous")),
    ("mono", _rx(r"monochromatic", r"black[\s-]*and[\s-]*white", r"monochrome", r"film noir", r"sepia")),
)
_KEY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("low", _rx(r"low[\s-]*key", r"chiaroscuro", r"noir", r"dark and moody", r"tenebris\w*")),
    ("high", _rx(r"high[\s-]*key", r"bright and airy", r"white background", r"seamless white")),
)
_SHOT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("macro", _rx(r"macro")),
    ("extreme_close_up", _rx(r"extreme close-?up", r"ecu")),
    ("full", _rx(r"full[- ]body", r"full[- ]length", r"full figure", r"head to toe")),
    ("medium", _rx(r"medium shot", r"waist[- ]up", r"half[- ]body", r"upper body", r"cowboy shot", r"mid shot")),
    ("close_up", _rx(r"close-?up", r"closeup", r"headshot", r"head and shoulders", r"bust shot")),
    ("extreme_wide", _rx(r"extreme wide", r"establishing shot", r"panorama\w*", r"vista")),
    ("wide", _rx(r"wide shot", r"long shot", r"full shot of")),
)
_ANGLE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dutch", _rx(r"dutch angle", r"dutch tilt", r"canted", r"tilted (?:frame|camera|horizon)")),
    ("birds_eye", _rx(r"bird'?s[- ]eye", r"aerial", r"drone shot", r"top[- ]down", r"overhead", r"flat ?lay")),
    ("worms_eye", _rx(r"worm'?s[- ]eye", r"from (?:far )?below", r"ground level")),
    ("low", _rx(r"low[- ]angle", r"looking up")),
    ("high", _rx(r"high[- ]angle", r"from above", r"looking down")),
    ("eye_level", _rx(r"eye[- ]level", r"straight[- ]on")),
)
_PLACEMENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("symmetry", _rx(r"symmetr\w*", r"mirror(?:ed)? symmetry", r"wes anderson", r"one[- ]point perspective")),
    ("center", _rx(r"cent(?:er|re)d", r"dead cent(?:er|re)", r"hero shot")),
    ("golden", _rx(r"golden (?:ratio|spiral|section)")),
    ("thirds", _rx(r"rule of thirds", r"off[- ]cent(?:er|re)d?")),
    ("negative_space", _rx(r"negative space", r"minimalis[tm]", r"empty space")),
)
_PERSPECTIVE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fisheye", _rx(r"fish-?eye")),
    ("tilt_shift", _rx(r"tilt[- ]shift", r"miniature effect")),
    ("anamorphic", _rx(r"anamorphic", r"cinemascope")),
    ("isometric", _rx(r"isometric", r"axonometric")),
)
_FOCAL_RE = re.compile(r"\b(\d{2,3})\s?mm\b(?!\s*(?:film|format|still|camera|photo))", re.I)
_APERTURE_RE = re.compile(r"\bf\s?/\s?(\d{1,2}(?:\.\d)?)\b", re.I)
_RATIO_RE = re.compile(r"\b([1-9]|1[0-6])\s?:\s?1\b")
_LENS_WORDS: tuple[tuple[re.Pattern[str], int], ...] = (
    (_rx(r"telephoto", r"long lens", r"compressed perspective"), 200),
    (_rx(r"wide[- ]angle", r"ultra[- ]wide"), 20),
    (_rx(r"macro lens"), 100),
)
_DOF_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "shallow",
        _rx(r"shallow depth of field", r"shallow dof", r"bokeh", r"blurr?(?:ed|y) background", r"background blur"),
    ),
    ("deep", _rx(r"deep focus", r"deep depth of field", r"everything in focus", r"sharp throughout")),
)
_SATURATION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vivid", _rx(r"vibrant", r"vivid", r"saturated", r"colou?rful", r"bold colou?rs?")),
    ("muted", _rx(r"muted", r"desaturated", r"pastel", r"faded", r"subdued")),
)
_COLOR_WORDS = _rx(
    r"colou?r\w*", r"red", r"orange", r"yellow", r"green", r"blue", r"purple", r"violet", r"pink", r"teal", r"gold\w*"
)
_BROAD_VIEW = _rx(
    r"landscape", r"valley", r"mountains?", r"vista", r"panoram\w*", r"plains", r"desert", r"ocean", r"sea", r"lake"
)
_SKY = _rx(r"dawn", r"sunrise", r"sunset", r"sky", r"clouds?", r"aurora", r"stars?", r"starry", r"storm\w*", r"dusk")
_GROUND = _rx(r"reflect\w*", r"lake", r"field", r"meadow", r"flowers?", r"beach", r"shore", r"puddles?", r"snow")
_ATMOSPHERE = _rx(r"mist\w*", r"fog\w*", r"haz[ey]")
_FACING = (("right", _rx(r"(?:facing|looking|walking|heading|running|gazing) (?:to the )?left")),)
_PHOTO_CUE = _rx(
    r"photo\w*",
    r"dslr",
    r"mirrorless",
    r"camera",
    r"lens",
    r"bokeh",
    r"\d{2,3}\s?mm",
    r"f/\d+",
    r"shot on",
    r"film still",
    r"polaroid",
)

# Pack text that contradicts the plan (see gate_tag).
_HUMAN_ONLY_TERMS = _rx(
    r"skin",
    r"pores?",
    r"freckles?",
    r"faces?",
    r"facial",
    r"makeup",
    r"lips",
    r"hands?",
    r"fingers?",
    r"fingertips",
    r"nails",
    r"wrists?",
    r"catchlights?",
    r"irises",
    r"micro-expressions?",
    r"clothing",
    r"wardrobe",
    r"garments?",
    r"hair",
    r"selfie",
    r"portrait",
)
_BODY_TERMS = _rx(
    r"eyes?",
    r"anatomy",
    r"anatomical",
    r"limbs?",
    r"joints?",
    r"skeleton",
    r"skeletal",
    r"muscles?",
    r"muscular",
    r"body",
    r"bodies",
    r"pose[sd]?",
    r"posing",
    r"feet",
    r"foot",
    r"teeth",
    r"expressions?",
    r"proportions",
    r"shoulders?",
    r"torso",
)
_PHOTO_TERMS = _rx(
    r"photo\w*",
    r"dslr",
    r"mirrorless",
    r"cameras?",
    r"lens(?:es)?",
    r"bokeh",
    r"sensor",
    r"iso",
    r"shutter",
    r"aperture",
    r"f/\d[\d.]*",
    r"\d{2,3}\s?mm",
    r"raw",
    r"exposure",
    r"film grain",
    r"film stock",
    r"analog",
    r"captures?",
    r"snapshot",
    r"shot on",
    r"depth of field",
    r"dof",
    r"focal",
    r"optical",
    r"chromatic aberration",
    r"vignett\w*",
    r"white balance",
    r"documentary",
    r"candid",
)
_RENDER_CAMERA_OK = _rx(r"lens(?:es)?", r"bokeh", r"depth of field", r"dof", r"focal", r"\d{2,3}\s?mm", r"f/\d[\d.]*")


@dataclass
class ArtDirection:
    """One decided shot. Angles in degrees; azimuth 0 = light from the camera, 90 = from the side."""

    medium: str = "unspecified"
    medium_family: str = "generic"
    photographic: bool = False
    subject: str = "scene"
    has_people: bool = False
    has_creature: bool = False
    shot: str = "wide"
    angle: str = "eye_level"
    placement: str = "thirds"
    subject_side: str = "left"
    subject_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    horizon: float | None = None
    focal_mm: int = 35
    aperture: float = 5.6
    dof: str = "medium"
    perspective: str = "natural"
    time_of_day: str = "unspecified"
    light_pattern: str = "natural"
    key_side: str = "left"
    key_azimuth_deg: float = 45.0
    key_elevation_deg: float = 30.0
    hard_light: bool = False
    key_fill_ratio: float = 3.0
    tonal_key: str = "mid"
    color_temp_k: int = 5600
    sources: tuple[str, ...] = ()
    atmosphere: bool = False
    harmony: str = "analogous"
    grade: str = "natural"
    saturation: str = "natural"
    explicit: frozenset[str] = field(default_factory=frozenset)

    def caption(self) -> str:
        """Conditioning text for the decided (non-explicit) fields, phrased for the medium."""
        return ", ".join(p for p in _caption_parts(self) if p)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["explicit"] = sorted(self.explicit)
        d["subject_box"] = [round(v, 3) for v in self.subject_box]
        d["sources"] = list(self.sources)
        d["caption"] = self.caption()
        d["medium_name"] = self.medium
        d.update(self.critic_targets())  # "medium" becomes the family, as art_critic expects
        return d

    def critic_targets(self) -> dict[str, Any]:
        """Plan keys in ``utils.quality.art_critic`` vocabulary (absent when not decided)."""
        out: dict[str, Any] = {"medium": self.medium_family, "pixel_art": self.medium == "pixel_art"}
        if self.grade in ("teal_orange", "warm", "cool", "neon", "mono", "vivid", "muted", "mixed"):
            out["grade"] = self.grade
        if self.tonal_key in ("low", "high"):
            out["key"] = self.tonal_key
        if self.angle == "dutch":
            out["composition"] = "dutch"
        elif self.shot in ("macro", "extreme_close_up"):
            out["composition"] = "fill"
        elif self.placement in ("thirds", "golden", "center", "symmetry", "negative_space"):
            out["composition"] = self.placement
        if self.harmony in ("mono", "analogous", "complementary", "split"):
            out["harmony"] = self.harmony
        if self.key_fill_ratio <= 2.0 or self.atmosphere:
            out["contrast"] = "low"
        return out


def _first(rules: tuple[tuple[str, re.Pattern[str]], ...], text: str) -> str | None:
    for name, rx in rules:
        if rx.search(text):
            return name
    return None


def mentions_people(text: str) -> bool:
    return bool(_PEOPLE.search(text or ""))


def _subject_kind(text: str, place: str | None, people: bool, creature: bool) -> str:
    if people:
        if _GROUP.search(text):
            return place if place in ("street", "city") else "group"
        if re.search(r"\b(?:portrait|headshot|close-?up|face|selfie|bust)\b", text, re.I) and not re.search(
            r"\bfull[- ]?(?:body|length)\b", text, re.I
        ):
            return "portrait"
        return "person"
    if _FIGURE.search(text) and place not in ("landscape", "city", "street"):
        return "figure"
    if creature and place not in ("macro", "landscape", "city", "architecture"):
        return "animal"
    return place or ("animal" if creature else "scene")


def _medium(text: str) -> tuple[str, str]:
    found = detect_all_mediums(text)
    if not found:
        return "unspecified", "generic"
    families = {m.family for m in found}
    # "photo of an oil painting" is still a painting; a photo cue only wins when nothing else is named.
    for m in found:
        if m.family != "photo":
            return m.name, m.family
    return found[0].name, "photo" if families == {"photo"} else found[0].family


def plan_art_direction(prompt: str, *, seed: int = 0, width: int = 0, height: int = 0) -> ArtDirection:
    """Parse explicit cues, then fill the open fields from genre defaults (seeded where several are good)."""
    text = (prompt or "").strip()
    rng = random.Random(f"{seed}:{text.lower()}")
    explicit: set[str] = set()

    def pick(name: str, rules: tuple[tuple[str, re.Pattern[str]], ...], default: Any) -> Any:
        hit = _first(rules, text)
        if hit is not None:
            explicit.add(name)
            return hit
        return default

    medium, family = _medium(text)
    if medium != "unspecified":
        explicit.add("medium")
    people = mentions_people(text)
    creature = bool(_CREATURE.search(text))
    place = _first(_KIND_RULES, text)
    kind = _subject_kind(text, place, people, creature)
    creature = creature or bool(_FIGURE.search(text))
    base = _KIND_DEFAULTS[kind]
    ad = ArtDirection(
        medium=medium,
        medium_family=family,
        photographic=family == "photo" or (family == "generic" and bool(_PHOTO_CUE.search(text))),
        subject=kind,
        has_people=people,
        has_creature=creature,
    )
    environment = kind == "animal" and _first(_KIND_RULES[-5:], text) is not None
    pet = kind == "animal" and bool(_PET.search(text))
    default_shot = base["shot"]
    if environment or (kind == "landscape" and not _BROAD_VIEW.search(text)):
        default_shot = "wide"
    elif kind == "person" and _ACTION.search(text):
        default_shot = "full"

    ad.shot = pick("shot", _SHOT_RULES, default_shot)
    ad.angle = pick("angle", _ANGLE_RULES, base["angle"])
    ad.placement = pick("placement", _PLACEMENT_RULES, base["placement"])
    if "placement" not in explicit and kind == "landscape" and re.search(r"\breflect\w*", text, re.I):
        ad.placement = "symmetry"
    facing = _first(_FACING, text)
    ad.subject_side = facing or rng.choice(("left", "right"))
    ad.key_side = "right" if ad.subject_side == "left" else "left"

    focal = _FOCAL_RE.search(text)
    lens_word = next((mm for rx, mm in _LENS_WORDS if rx.search(text)), None)
    if focal:
        ad.focal_mm, _ = int(focal.group(1)), explicit.add("focal_mm")
    elif lens_word:
        ad.focal_mm, _ = lens_word, explicit.add("focal_mm")
    else:
        ad.focal_mm = 200 if environment else 85 if pet else int(base["focal"])
    aperture = _APERTURE_RE.search(text)
    if aperture:
        ad.aperture, _ = float(aperture.group(1)), explicit.add("aperture")
    else:
        ad.aperture = 8.0 if environment else 2.8 if pet else float(base["f"])
    dof = _first(_DOF_RULES, text)
    if dof:
        ad.dof, _ = dof, explicit.add("dof")
    elif kind == "macro" or ad.shot == "macro":
        ad.dof = "shallow"  # magnification makes the focal plane millimetres deep even at f/8
    else:
        ad.dof = "shallow" if ad.aperture <= 2.8 else "deep" if ad.aperture >= 8 else "medium"
    ad.perspective = pick("perspective", _PERSPECTIVE_RULES, "")
    if not ad.perspective:
        if family in ("2d",) and medium in ("pixel_art", "vector"):
            ad.perspective = "flat"
        elif ad.focal_mm <= 28:
            ad.perspective = "wide"
        elif ad.focal_mm >= 85:
            ad.perspective = "compressed"
        else:
            ad.perspective = "natural"

    _plan_light(ad, text, kind, place, base, rng, explicit)
    _plan_palette(ad, text, kind, rng, explicit)
    _plan_frame(ad, text, kind, width, height)
    ad.explicit = frozenset(explicit)
    return ad


def _plan_light(
    ad: ArtDirection,
    text: str,
    kind: str,
    place: str | None,
    base: dict,
    rng: random.Random,
    explicit: set[str],
) -> None:
    # People and groups placed in a landscape or street are lit by the sky, not by a studio pattern.
    outdoor = kind in _OUTDOOR | {"scene"} or place in ("landscape", "city", "street")
    time = _first(_TIME_RULES, text)
    if time:
        explicit.add("time_of_day")
    elif kind in _TIME_CHOICES or (outdoor and place in _TIME_CHOICES):
        time = rng.choice(_TIME_CHOICES.get(kind) or _TIME_CHOICES[place])
    ad.time_of_day = time or ("studio" if kind in ("product", "portrait", "figure") else "interior")
    if kind in ("interior", "food", "still_life") and ad.time_of_day not in _TIMES:
        ad.time_of_day = "afternoon"
    ad.atmosphere = bool(_ATMOSPHERE.search(text))
    ad.sources = tuple(name for name, rx in _SOURCE_RULES if rx.search(text))
    if ad.sources:
        explicit.add("sources")
    elif kind in ("food", "still_life", "interior"):
        ad.sources = ("window",)

    tod = _TIMES.get(ad.time_of_day)
    pattern = _first(_PATTERN_RULES, text)
    if pattern:
        explicit.add("light_pattern")
    tonal = _first(_KEY_RULES, text)
    if tonal:
        explicit.add("tonal_key")
    noir = bool(re.search(r"\bnoir\b", text, re.I))

    ad.light_pattern = pattern or (
        "split"
        if noir
        else "loop"
        if kind in ("portrait", "person") and "window" not in ad.sources and not outdoor
        else "three_point"
        if kind in ("product", "figure")
        else "natural"
    )
    if ad.light_pattern in _PATTERN_GEOMETRY:
        ad.key_azimuth_deg, ad.key_elevation_deg = _PATTERN_GEOMETRY[ad.light_pattern]
    else:
        ad.key_azimuth_deg, ad.key_elevation_deg = float(base["az"]), float(base["el"])
    if tod and (outdoor or kind == "group") and ad.light_pattern == "natural":
        ad.key_elevation_deg = float(max(tod["el"], 0))
    if "window" in ad.sources and kind in ("portrait", "person", "interior"):
        ad.key_azimuth_deg, ad.key_elevation_deg = 70.0, 15.0

    sunlit = tod is not None and ad.time_of_day not in ("night", "blue_hour") and (outdoor or kind == "group")
    ad.hard_light = bool(tod["hard"]) if sunlit else kind in ("architecture", "vehicle", "street")
    if noir or ad.light_pattern in ("split", "chiaroscuro"):
        ad.hard_light = True
    if ad.atmosphere or ad.time_of_day == "overcast" or ad.light_pattern in ("butterfly", "broad"):
        ad.hard_light = False
    if re.search(r"\b(?:hard|harsh|direct) (?:light|sun\w*)\b|\bspotlight\b", text, re.I):
        ad.hard_light, _ = True, explicit.add("hard_light")
    elif re.search(r"\b(?:soft|diffused?) light\w*\b", text, re.I):
        ad.hard_light, _ = False, explicit.add("hard_light")

    ratio = _RATIO_RE.search(text)
    if ratio:
        ad.key_fill_ratio, _ = float(ratio.group(1)), explicit.add("key_fill_ratio")
    elif tonal == "low" or noir:
        ad.key_fill_ratio = 8.0
    elif tonal == "high":
        ad.key_fill_ratio = 1.5
    elif kind in ("portrait", "person", "group", "product", "food", "macro", "still_life") and not (outdoor and tod):
        ad.key_fill_ratio = float(base["ratio"])
    else:
        ad.key_fill_ratio = float(tod["ratio"]) if tod else float(base["ratio"])
    if ad.atmosphere and "key_fill_ratio" not in explicit:
        ad.key_fill_ratio = min(ad.key_fill_ratio, 2.0)
    ad.tonal_key = tonal or (
        "low" if ad.key_fill_ratio >= 8 or ad.time_of_day == "night" else "high" if kind == "product" else "mid"
    )

    kelvin = int(tod["kelvin"]) if tod else 5600
    warm_sources = [_SOURCE_KELVIN[s] for s in ad.sources if s in _SOURCE_KELVIN and s != "window"]
    if warm_sources and ad.time_of_day in ("night", "blue_hour", "interior", "unspecified", "afternoon"):
        kelvin = min(warm_sources)
    ad.color_temp_k = kelvin


def _plan_palette(ad: ArtDirection, text: str, kind: str, rng: random.Random, explicit: set[str]) -> None:
    grade = _first(_GRADE_RULES, text)
    if grade:
        explicit.add("grade")
    harmony = _first(_HARMONY_RULES, text)
    if harmony:
        explicit.add("harmony")
    tod = _TIMES.get(ad.time_of_day)
    linework = ad.medium in ("ink", "drawing") and not _COLOR_WORDS.search(text)
    if not grade:
        if linework or (re.search(r"\bnoir\b", text, re.I) and "neon" not in text.lower()):
            grade = "mono"
        elif "neon" in ad.sources:
            grade = "neon"
        elif tod and tod["grade"]:
            grade = tod["grade"]
        elif ad.color_temp_k <= 3200:
            grade = "warm"
        else:
            grade = "natural"
    ad.grade = grade
    if not harmony:
        if grade == "mono":
            harmony = "mono"
        elif grade in ("teal_orange", "mixed"):
            harmony = "complementary"
        elif grade == "neon":
            harmony = "split"
        elif kind == "product":
            harmony = "analogous"
        elif tod:
            harmony = tod["harmony"]
        else:
            harmony = rng.choice(("analogous", "complementary"))
    ad.harmony = harmony
    sat = _first(_SATURATION_RULES, text)
    if sat:
        explicit.add("saturation")
    else:
        sat = (
            "muted"
            if grade in ("muted", "mono") or ad.atmosphere or ad.medium in ("watercolor", "drawing", "ink")
            else "vivid"
            if grade in ("neon", "vivid") or ad.medium == "anime"
            else "natural"
        )
    ad.saturation = sat


def _plan_frame(ad: ArtDirection, text: str, kind: str, width: int, height: int) -> None:
    """Subject box (normalised x0, y0, x1, y1) and horizon height (fraction from the top)."""
    if kind in ("landscape", "city", "street", "architecture", "vehicle") or ad.shot in ("wide", "extreme_wide"):
        if ad.angle in ("birds_eye", "worms_eye") or kind == "interior":
            ad.horizon = None
        elif ad.placement == "symmetry" and kind == "landscape":
            ad.horizon = 0.5
        elif _SKY.search(text) and not _GROUND.search(text):
            ad.horizon = 0.67
        elif _GROUND.search(text) and not _SKY.search(text):
            ad.horizon = 0.33
        else:
            ad.horizon = 0.67 if ad.time_of_day in ("golden_hour", "dawn", "blue_hour") else 0.62
    size = {
        "macro": 0.55,
        "extreme_close_up": 0.7,
        "close_up": 0.5,
        "medium": 0.45,
        "full": 0.38,
        "medium_wide": 0.32,
        "wide": 0.25,
        "extreme_wide": 0.18,
    }.get(ad.shot, 0.3)
    aspect = (width / height) if width > 0 and height > 0 else 1.0
    w = min(0.9, size / max(aspect, 0.5) ** 0.5)
    h = min(0.9, size * max(aspect, 0.5) ** 0.5 * (1.6 if ad.shot in ("full", "medium") else 1.0))
    if ad.placement in ("center", "symmetry"):
        cx = 0.5
    elif ad.placement == "golden":
        cx = 0.382 if ad.subject_side == "left" else 0.618
    else:
        cx = 1 / 3 if ad.subject_side == "left" else 2 / 3
    cy = 0.5 if ad.horizon is None else min(0.6, ad.horizon)
    if ad.shot in ("close_up", "medium") and ad.has_people:
        cy = 0.45
    x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    ad.subject_box = (x0, y0, min(1.0, x0 + w), min(1.0, y0 + h))


_SHOT_TEXT = {
    "macro": "macro close-up",
    "extreme_close_up": "extreme close-up",
    "close_up": "close-up",
    "medium": "medium shot",
    "full": "full-length shot",
    "medium_wide": "medium-wide shot",
    "wide": "wide shot",
    "extreme_wide": "extreme wide view",
}
_ANGLE_TEXT = {
    "low": "low camera angle",
    "high": "slightly high angle",
    "birds_eye": "top-down view",
    "worms_eye": "worm's-eye view",
    "dutch": "dutch tilt",
    "eye_level": "eye-level",
}
_TIME_TEXT = {
    "dawn": "low dawn light",
    "morning": "morning sun",
    "midday": "high midday sun",
    "afternoon": "afternoon sun",
    "golden_hour": "low golden-hour sun",
    "blue_hour": "blue-hour ambient light",
    "night": "night light from a few motivated sources",
    "overcast": "overcast daylight",
}
_PATTERN_TEXT = {
    "rembrandt": "Rembrandt key light",
    "loop": "loop key light",
    "butterfly": "butterfly light",
    "split": "split light",
    "rim": "rim light",
    "broad": "broad light",
    "chiaroscuro": "chiaroscuro light",
    "three_point": "three-point studio light",
}
_HARMONY_TEXT = {
    "analogous": "analogous palette",
    "complementary": "complementary warm-cool palette",
    "split": "split-complementary palette",
    "triadic": "triadic palette with one dominant hue",
    "mono": "monochrome palette",
}
_GRADE_TEXT = {
    "warm": "warm grade",
    "cool": "cool grade",
    "muted": "muted grade",
    "teal_orange": "teal-and-orange grade",
    "neon": "neon magenta and cyan accents",
    "mono": "black-and-white tonality",
    "vivid": "vivid color",
    "mixed": "mixed tungsten and daylight",
}


def _direction(ad: ArtDirection, *, with_height: bool) -> str:
    az, side = ad.key_azimuth_deg, ad.key_side
    height = ""
    if with_height:
        height = "high " if ad.key_elevation_deg >= 35 else "low " if ad.key_elevation_deg <= 12 else ""
    if az < 15:
        return "from above the lens" if height == "high " else "from the front"
    if az < 60:
        return f"from the {height}front {side}"
    if az < 120:
        return f"from the {height}{side}"
    return f"from behind on the {side}"


def _light_phrase(ad: ArtDirection) -> str:
    """Key light: what it is, its quality and where it comes from (only the undecided parts)."""
    ex = ad.explicit
    quality = "" if "hard_light" in ex else ("hard " if ad.hard_light else "soft ")
    if "light_pattern" in ex:
        return f"{quality}key light".strip() if quality else ""
    if ad.light_pattern != "natural":
        return f"{quality}{_PATTERN_TEXT[ad.light_pattern]} {_direction(ad, with_height=True)}"
    if ad.time_of_day in _TIME_TEXT and "time_of_day" not in ex and not ad.sources:
        if ad.time_of_day in ("overcast", "night", "blue_hour"):
            return _TIME_TEXT[ad.time_of_day]
        lead = "" if ad.time_of_day in ("dawn", "golden_hour", "midday") else quality
        return f"{lead}{_TIME_TEXT[ad.time_of_day]} {_direction(ad, with_height=False)}"
    if ad.sources:
        return f"{quality}{ad.sources[0]} light {_direction(ad, with_height=ad.sources[0] != 'window')}"
    # A natural pattern with a time of day and no named source is sky light.
    if ad.time_of_day in ("dawn", "morning", "midday", "afternoon", "golden_hour"):
        return f"{quality}sunlight {_direction(ad, with_height=True)}"
    if ad.time_of_day == "overcast":
        return "soft diffuse daylight from above"
    return f"{quality}key light {_direction(ad, with_height=True)}"


def _caption_parts(ad: ArtDirection) -> list[str]:
    ex = ad.explicit
    painted = ad.medium_family in ("2d", "traditional")
    graphic = ad.medium in ("pixel_art", "vector")
    linework = ad.medium in ("ink", "drawing")
    parts: list[str] = []

    if "shot" not in ex:
        parts.append(_SHOT_TEXT.get(ad.shot, ""))
    if "angle" not in ex and ad.angle != "eye_level":
        parts.append(_ANGLE_TEXT[ad.angle])
    if "placement" not in ex:
        if ad.placement == "thirds":
            scenic = ad.shot == "extreme_wide" and not (ad.has_people or ad.has_creature)
            parts.append(f"{'focal point' if scenic else 'subject'} on the {ad.subject_side} third")
        elif ad.placement == "symmetry":
            parts.append("symmetrical composition")
        elif ad.placement == "center":
            parts.append(
                "centered with generous negative space" if ad.subject == "product" else "balanced centered framing"
            )
    if ad.horizon is not None:
        parts.append(
            "horizon at mid-frame"
            if abs(ad.horizon - 0.5) < 0.05
            else f"horizon on the {'lower' if ad.horizon > 0.5 else 'upper'} third"
        )
    if ad.subject in ("landscape", "city") and ad.shot in ("wide", "extreme_wide") and not graphic:
        parts.append("layered depth with atmospheric perspective")
    if ad.subject in ("architecture", "interior") and not painted:
        parts.append("vertical lines kept parallel")
    if ad.has_people and ad.placement == "thirds" and ad.shot in ("close_up", "medium", "full"):
        parts.append("open space in front of the subject")

    if ad.photographic or (ad.medium_family == "3d" and not graphic):
        lens = []
        if "focal_mm" not in ex:
            lens.append(f"{ad.focal_mm}mm lens")
        if "aperture" not in ex:
            lens.append(f"f/{ad.aperture:g}")
        if lens:
            parts.append(" at ".join(lens))
    if "dof" not in ex and not graphic:
        if ad.dof == "shallow":
            parts.append("soft background" if painted else "background softly out of focus")
        elif ad.dof == "deep" and not painted:
            parts.append("sharp from foreground to background")

    if graphic:
        if "light_pattern" not in ex:
            parts.append(f"single light source from the top {ad.key_side}")
        parts.append("two-tone shading" if ad.medium == "pixel_art" else "flat shapes with minimal shading")
    else:
        parts.append(_light_phrase(ad))
        if "key_fill_ratio" not in ex:
            r = ad.key_fill_ratio
            if painted:
                parts.append(
                    "strong value contrast"
                    if r >= 6
                    else "gentle value contrast"
                    if r <= 2
                    else "clear light and shadow shapes"
                )
            elif r >= 8:
                parts.append("deep shadows at an 8:1 key-to-fill ratio")
            elif r <= 1.5:
                parts.append("low-contrast even fill")
            else:
                parts.append(f"{r:g}:1 key-to-fill ratio")
        if "tonal_key" not in ex and ad.tonal_key in ("low", "high"):
            parts.append(f"{ad.tonal_key}-key values")

    if ad.medium == "pixel_art":
        parts.append("limited palette with hue-shifted shadow ramps")
    elif not (linework and ad.harmony == "mono"):
        if "harmony" not in ex and "grade" not in ex:
            parts.append(_HARMONY_TEXT.get(ad.harmony, ""))
        if "grade" not in ex and ad.grade in _GRADE_TEXT and not (ad.grade == "mono" and ad.harmony == "mono"):
            parts.append(_GRADE_TEXT[ad.grade])
        if "saturation" not in ex and ad.saturation != "natural" and ad.grade not in ("muted", "vivid", "mono"):
            parts.append(f"{ad.saturation} saturation")
    return parts


# Positive pack tags that assert the opposite of what the plan decided.
_CONTRADICTS_PLAN: tuple[tuple[Any, re.Pattern[str]], ...] = (
    (lambda ad: ad.dof == "deep", _rx(r"bokeh", r"shallow depth of field", r"blurr?(?:ed|y) background")),
    (lambda ad: ad.dof == "shallow", _rx(r"deep focus", r"everything in focus", r"sharp throughout")),
    (lambda ad: ad.tonal_key == "low", _rx(r"high[- ]key", r"bright and airy")),
    (lambda ad: ad.tonal_key == "high", _rx(r"low[- ]key", r"dark and moody")),
    (lambda ad: ad.saturation == "muted" or ad.grade == "mono", _rx(r"vivid", r"vibrant", r"saturated colou?rs?")),
    (lambda ad: ad.saturation == "vivid", _rx(r"muted", r"desaturated")),
    (
        lambda ad: ad.placement in ("center", "symmetry"),
        _rx(r"rule of thirds", r"off[- ]cent(?:er|re)", r"balanced asymmetry", r"thirds intersection"),
    ),
    (
        lambda ad: ad.placement == "thirds",
        _rx(r"cent(?:er|re)d (?:composition|subject|framing)", r"symmetrical composition"),
    ),
)


# Negative tags that would push away from what the plan decided.
_NEGATES_PLAN: tuple[tuple[Any, re.Pattern[str]], ...] = (
    (lambda ad: ad.placement in ("center", "symmetry"), _rx(r"\w*cent(?:er|re)\w*", r"symmetr\w*")),
    (lambda ad: ad.grade == "mono", _rx(r"monochrome", r"black and white", r"gr[ae]yscale", r"desaturated")),
    (lambda ad: ad.tonal_key == "low", _rx(r"too dark", r"underexposed", r"crushed blacks", r"dark image")),
    (lambda ad: ad.tonal_key == "high", _rx(r"overexposed", r"washed[- ]out", r"too bright")),
    (lambda ad: ad.saturation == "vivid", _rx(r"oversaturated", r"saturated", r"candy saturation")),
    (lambda ad: ad.saturation == "muted", _rx(r"desaturated", r"dull colou?rs?", r"muted", r"washed[- ]out")),
    (lambda ad: ad.grade == "teal_orange", _rx(r"teal", r"orange")),
)


def gate_tag(tag: str, ad: ArtDirection, *, user_text: str = "", negative: bool = False) -> bool:
    """False when a pack tag contradicts the plan.

    Positive tags: body text with nobody in frame, camera text on a painting, the opposite of a planned
    look ("bokeh" for a deep-focus plan, "rule of thirds" for a centred one). Negative tags: body text
    with nobody in frame, and negations of a decided look (a "dead-center framing" negative against a
    centred product shot, "monochrome" against a noir plan); camera terms stay, a painting's negative
    may rightly name them.
    """
    if not ad.has_people and _HUMAN_ONLY_TERMS.search(tag):
        return False
    if not ad.has_people and not ad.has_creature and _BODY_TERMS.search(tag):
        return False
    if negative:
        return not any(applies(ad) and rx.search(tag) for applies, rx in _NEGATES_PLAN)
    if any(applies(ad) and rx.search(tag) for applies, rx in _CONTRADICTS_PLAN):
        return False
    if ad.medium_family in ("2d", "traditional", "3d") and not _PHOTO_CUE.search(user_text):
        if ad.medium_family == "3d":
            return not _PHOTO_TERMS.search(_RENDER_CAMERA_OK.sub("", tag))
        return not _PHOTO_TERMS.search(tag)
    return True


def apply_art_direction_to_args(args: Any) -> ArtDirection | None:
    """Plan from the user's own prompt; sets ``args._art_direction`` and ``args.art_direction_plan``."""
    mode = str(getattr(args, "art_direction", "auto") or "auto").strip().lower()
    text = str(getattr(args, "_user_prompt", "") or getattr(args, "prompt", "") or "")
    if mode in ("off", "none", "0", "false") or not text.strip():
        args._art_direction = None
        args.art_direction_plan = None
        return None
    ad = plan_art_direction(
        text,
        seed=int(getattr(args, "seed", 0) or 0),
        width=int(getattr(args, "width", 0) or 0),
        height=int(getattr(args, "height", 0) or 0),
    )
    args._art_direction = ad
    args.art_direction_plan = ad.to_dict()
    return ad
