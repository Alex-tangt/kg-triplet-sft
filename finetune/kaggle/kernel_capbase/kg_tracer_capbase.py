#!/usr/bin/env python3
"""Inference on a CAPACITY-CURVE LoRA adapter (bf16 Unsloth line, ADR-0006).

Loads the base Qwen3 (Kaggle model slug), merges a capacity adapter mounted
from the `idalextan/cap-upload` dataset (qwen3-0.6b / qwen3-1.7b / qwen3-4b),
and runs single-shot extraction on the 200-passage referent sample
(sample_200_passages.jsonl) with the STUDENT_PROMPT (thinking OFF).

`NO_ADAPTER` (baked True by `make_capinfer.py --no-adapter`) skips the LoRA
merge entirely and runs the same protocol/schema-prefix/parser on the raw base
model — the base-baseline leg (ADR-0009): everything is identical to the
adapter kernels except that no adapter is loaded, so the base-vs-masked delta
isolates what SFT added.

PROTOCOL selects the prompt wrapper (baked per model by make_capinfer.py):
  - "chat":  Qwen3 chat template (reproduction-line protocol, self-consistent
             with the chat-trained reference baseline);
  - "plain": training-faithful format for the capacity line
             (STUDENT_PROMPT\\n<passage>\\n\\n, no chat markers — matches
             train_unsloth.py build_text; see issue 08 Finding 2026-09-06).
Both keep the schema-prefix injection + tolerant parser + length-sorted batched
decode + one worker per GPU, so the two differ only in the wrapper.

Writes predictions.jsonl + raw_<id>.txt + summary.csv under /kaggle/working.

Output rows match the deploy extraction format consumed by eval/referent_*:
  {"id", "entities": [{title,type,description}], "relationships":
   [{source,target,description,strength}]}

Usage on Kaggle: python /kaggle/src/script.py  (metadata per model dir)
"""

import json
import os
import re
import sys

# Device selection is per-worker inside _infer_worker; the parent must NOT pin
# CUDA_VISIBLE_DEVICES (see main()).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ---- embedded eval inputs (STUDENT_PROMPT only; injected by merge) ----
INPUTS = {
 "student_prompt": "-GOAL-\nGiven a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.\n\n-STEPS-\n1. Identify all entities. For each identified entity, record the following:\n- title: name of the entity in canonical uppercase form, consistent across the whole output\n- type: one of PERSON, ORGANIZATION, GEO, EVENT, CONCEPT\n- description: a concise factual description of the entity's attributes and role in the text (one or two sentences directly supported by the text; include numeric details such as dates, counts and measurements when the text states them)\n\n2. From the entities recorded in step 1, identify all pairs of (source, target) that are *clearly related* to each other.\nFor each related pair, record the following:\n- source: title of the source entity, as recorded in step 1\n- target: title of the target entity, as recorded in step 1\n- description: why you think the source entity and the target entity are related, grounded in the text\n- strength: a numeric score from 0 to 10 indicating how strong the relation between the source and target entities is\n\n-OUTPUT-\nReturn a single JSON object with exactly two keys:\n{\"entities\": [{\"title\", \"type\", \"description\"}], \"relationships\": [{\"source\", \"target\", \"description\", \"strength\"}]}\n\"source\" and \"target\" must exactly equal the \"title\" of an entity listed in \"entities\". Only include entities and relationships explicitly supported by the text; never invent facts, entities or relations. If nothing can be extracted, return {\"entities\": [], \"relationships\": []}.",
 "passages": [
  {
   "id": "wikipedia-02909",
   "text": "Arne Kaijser (born 1950) is a professor emeritus of history of technology at the KTH Royal Institute of Technology in Stockholm, and a former president of the Society for the History of Technology. Kaijser has published two books in Swedish: Stadens ljus. Etableringen av de första svenska gasverken and I fädrens spår. Den svenska infrastrukturens historiska utveckling och framtida utmaningar, and has co-edited several anthologies. Kaijser is a member of the Royal Swedish Academy of Engineering Sciences since 2007 and also a member of the editorial board of two scientific journals: Journal of Urban Technology and Centaurus. Lately, he has been occupied with the history of Large Technical Systems."
  },
  {
   "id": "arxiv-00944",
   "text": "the three - gluon vertex is a basic object of interest in nonabelian gauge theory . it contains important structural information , in particular on infrared divergences , and also figures prominently in the schwinger - dyson equations . at the one - loop level , it has been calculated and analyzed by a number of authors . here we use the worldline formalism to unify the calculations of the scalar , spinor and gluon loop contributions to the one - loop vertex , leading to an extremely compact representation . the susy - related sum rule found by binger and brodsky follows from an off - shell extension of the bern - kosower replacement rules . we explain the relation of the structure of our representation to the low - energy effective action ."
  },
  {
   "id": "wikipedia-01557",
   "text": "Its territory can be subdivided into four terrestrial ecoregions of the Palearctic realm namely within the Illyrian deciduous forests, Balkan mixed forests, Pindus Mountains mixed forests and Dinaric Mountains mixed forests. Approximately 3,500 different species of plants can be found in Albania which refers principally to a Mediterranean and Eurasian character. The country maintains a vibrant tradition of herbal and medicinal practices. At the minimum 300 plants growing locally are used in the preparation of herbs and medicines. The trees within the forests are primarily fir, oak, beech and pine. Conservation Albania has been an active participant in numerous international agreements and conventions aimed at strengthing its commitment to the preservation and sustainable management of biological diversity. Since 1994, the country is a party to the Convention on Biological Diversity (CBD) and its associated Cartagena and Nagoya Protocols. To uphold these commitments, it has developed and implemented a comprehensive National Biodiversity Strategy and Action Plan (NBSAP). Furthermore, Albania has established a partnership with the International Union for Conservation of Nature (IUCN), advancing its conservation efforts on both national and international scales. Guided by the IUCN, the country has made substantial progress in the foundation of protected areas within its boundaries, encompassing 12 national parks among others Butrint, Karaburun-Sazan, Llogara, Prespa and Vjosa. As a signatory to the Ramsar Convention, Albania has granted special recognition upon four wetlands, designating them as Wetlands of International Importance, including Buna-Shkodër, Butrint, Karavasta and Prespa. The country's dedication to protection extends further into the sphere of UNESCO's World Network of Biosphere Reserves, operating within the framework of the Man and the Biosphere Programme, evidenced by its engagement in the Ohrid-Prespa Transboundary Biosphere Reserve."
  },
  {
   "id": "wikipedia-02266",
   "text": "Audi AG () is a German automotive manufacturer of luxury vehicles headquartered in Ingolstadt, Bavaria, Germany. A subsidiary of the Volkswagen Group, Audi produces vehicles in nine production facilities worldwide. The origins of the company are complex, dating back to the early 20th century and the initial enterprises (Horch and the Audiwerke) founded by engineer August Horch (1868–1951). Two other manufacturers (DKW and Wanderer) also contributed to the foundation of Auto Union in 1932. The modern Audi era began in the 1960s, when Auto Union was acquired by Volkswagen from Daimler-Benz. After relaunching the Audi brand with the 1965 introduction of the Audi F103 series, Volkswagen merged Auto Union with NSU Motorenwerke in 1969, thus creating the present-day form of the company. The company name is based on the Latin translation of the surname of the founder, August Horch. , meaning \"listen\", becomes in Latin. The four rings of the Audi logo each represent one of four car companies that banded together to create Audi's predecessor company, Auto Union. Audi's slogan is , which is translated as \"Progress through Technology\". Audi, along with German brands BMW and Mercedes-Benz, is among the best-selling luxury automobile brands in the world. History Birth of the company and its name Automobile company Wanderer was originally established in 1885, later becoming a branch of Audi AG. Another company, NSU, which also later merged into Audi, was founded during this time, and later supplied the chassis for Gottlieb Daimler's four-wheeler. On 14 November 1899, August Horch (1868–1951) established the company A. Horch & Cie. in the Ehrenfeld district of Cologne. In 1902, he moved with his company to Reichenbach im Vogtland. On 10 May 1904, he founded the August Horch & Cie. Motorwagenwerke AG, a joint-stock company in Zwickau (State of Saxony)."
  },
  {
   "id": "wikipedia-01974",
   "text": "NGC 772 has a diameter of 240,000 light-years and the system is 114 million light-years from Earth. Another spiral galaxy in Aries is NGC 673, a face-on class SAB(s)c galaxy. It is a weakly barred spiral galaxy with loosely wound arms. It has no ring and a faint bulge and is 2.5 by 1.9 arcminutes. It has two primary arms with fragments located farther from the core. 171,000 light-years in diameter, NGC 673 is 235 million light-years from Earth. NGC 678 and NGC 680 are a pair of galaxies in Aries that are only about 200,000 light-years apart. Part of the NGC 691 group of galaxies, both are at a distance of approximately 130 million light-years. NGC 678 is an edge-on spiral galaxy that is 4.5 by 0.8 arcminutes. NGC 680, an elliptical galaxy with an asymmetrical boundary, is the brighter of the two at magnitude 12.9; NGC 678 has a magnitude of 13.35. Both galaxies have bright cores, but NGC 678 is the larger galaxy at a diameter of 171,000 light-years; NGC 680 has a diameter of 72,000 light-years. NGC 678 is further distinguished by its prominent dust lane. NGC 691 itself is a spiral galaxy slightly inclined to our line of sight. It has multiple spiral arms and a bright core. Because it is so diffuse, it has a low surface brightness. It has a diameter of 126,000 light-years and is 124 million light-years away. NGC 877 is the brightest member of an 8-galaxy group that also includes NGC 870, NGC 871, and NGC 876, with a magnitude of 12.53. It is 2.4 by 1.8 arcminutes and is 178 million light-years away with a diameter of 124,000 light-years. Its companion is NGC 876, which is about 103,000 light-years from the core of NGC 877."
  },
  {
   "id": "wikipedia-01420",
   "text": "In 2016, Alberta continued to have the youngest population among the provinces with a median age of 36.7 years, compared with the national median of 41.2 years. Also in 2016, Alberta had the smallest proportion of seniors (12.3%) among the provinces and one of the highest population shares of children (19.2%), further contributing to Alberta's young and growing population. About 81% of the population lives in urban areas and only about 19% in rural areas. The Calgary–Edmonton Corridor is the most urbanized area in the province and is one of the most densely populated areas of Canada. Many of Alberta's cities and towns have experienced very high rates of growth in recent history. Alberta's population rose from 73,022 in 1901 to 3,290,350 according to the 2006 census. According to the 2016 census Alberta has 779,155 residents (19.2%) between the ages of 0–14, 2,787,805 residents (68.5%) between the ages of 15–64, and 500,215 residents (12.3%) aged 65 and over. Additionally, as per the 2016 census, 1,769,500 residents hold a postsecondary certificate, diploma or degree, 895,885 residents have obtained a secondary (high) school diploma or equivalency certificate, and 540,665 residents do not have any certificate, diploma or degree. Municipalities Language As of the 2021 Canadian Census, the ten most spoken languages in the province included English (4,109,720 or 98.37%), French (260,415 or 6.23%), Tagalog (172,625 or 4.13%), Punjabi (126,385 or 3.03%), Spanish (116,070 or 2.78%), Hindi (94,015 or 2.25%), Mandarin (82,095 or 1.97%), Arabic (76,760 or 1.84%), Cantonese (74,960 or 1.79%), and German (65,370 or 1.56%). The question on knowledge of languages allows for multiple responses. As of the 2016 census, English is the most common mother tongue, with 2,991,485 native speakers."
  },
  {
   "id": "wikipedia-01252",
   "text": "Biggest seas in Atlantic Ocean Top large seas: Sargasso Sea3.5 million km2 Caribbean Sea2.754 million km2 Mediterranean Sea2.510 million km2 Gulf of Guinea2.35 million km2 Gulf of Mexico1.550 million km2 Norwegian Sea1.383 million km2 Hudson Bay1.23 million km2 Greenland Sea1.205 million km2 Argentine Sea1 million km2 Labrador Sea841,000 km2 Irminger Sea780,000 km2 Baffin Bay689,000 km2 North Sea575,000 km2 Black Sea436,000 km2 Baltic Sea377,000 km2 Libyan Sea350,000 km2 Levantine Sea320,000 km2 Celtic Sea300,000 km2 Tyrrhenian Sea275,000 km2 Gulf of Saint Lawrence226,000 km2 Bay of Biscay223,000 km2 Aegean Sea214,000 km2 Ionian Sea169,000 km2 Balearic Sea150,000 km2 Adriatic Sea138,000 km2 Gulf of Bothnia116,300 km2 Sea of Crete95,000 km2 Gulf of Maine93,000 km2 Ligurian Sea80,000 km2 English Channel75,000 km2 James Bay68,300 km2 Bothnian Sea66,000 km2 Gulf of Sidra57,000 km2 Sea of the Hebrides47,000 km2 Irish Sea46,000 km2 Sea of Azov39,000 km2 Bothnian Bay36,800 km2 Gulf of Venezuela17,840 km2 Bay of Campeche16,000 km2 Gulf of Lion15,000 km2 Sea of Marmara11,350 km2 Wadden Sea10,000 km2 Archipelago Sea8,300 km2 Bathymetry The bathymetry of the Atlantic is dominated by a submarine mountain range called the Mid-Atlantic Ridge (MAR). It runs from 87°N or south of the North Pole to the subantarctic Bouvet Island at 54°S. Expeditions to explore the bathymertry of the Atlantic include the Challenger expedition and the German Meteor expedition; , Columbia University's Lamont–Doherty Earth Observatory and the United States Navy Hydrographic Office conduct research on the ocean. Mid-Atlantic Ridge The MAR divides the Atlantic longitudinally into two halves, in each of which a series of basins are delimited by secondary, transverse ridges. The MAR reaches above along most of its length, but is interrupted by larger transform faults at two places: the Romanche Trench near the Equator and the Gibbs Fracture Zone at 53°N."
  },
  {
   "id": "wikipedia-00083",
   "text": "In 2009, Bryant–Denny Stadium and Jordan-Hare Stadium became the homes of the Alabama High School Athletic Association state football championship games, after previously being held at Legion Field in Birmingham. Transportation Aviation Major airports with sustained operations in Alabama include Birmingham-Shuttlesworth International Airport (BHM), Huntsville International Airport (HSV), Dothan Regional Airport (DHN), Mobile Regional Airport (MOB), Montgomery Regional Airport (MGM), Northwest Alabama Regional Airport (MSL) and Northeast Alabama Regional Airport (GAD). Rail For rail transport, Amtrak schedules the Crescent, a daily passenger train, running from New York to New Orleans with station stops at Anniston, Birmingham, and Tuscaloosa. Roads Alabama has six major interstate routes: Interstate 65 (I-65) travels north–south roughly through the middle of the state; I-20/I-59 travel from the central west Mississippi state line to Birmingham, where I-59 continues to the north-east corner of the state and I-20 continues east towards Atlanta; I-85 originates in Montgomery and travels east-northeast to the Georgia state line, providing a main thoroughfare to Atlanta; and I-10 traverses the southernmost portion of the state, traveling from west to east through Mobile. I-22 enters the state from Mississippi and connects Birmingham with Memphis, Tennessee. In addition, there are currently five auxiliary interstate routes in the state: I-165 in Mobile, I-359 in Tuscaloosa, I-459 around Birmingham, I-565 in Decatur and Huntsville, and I-759 in Gadsden. A sixth route, I-685, will be formed when I-85 is rerouted along a new southern bypass of Montgomery. A proposed northern bypass of Birmingham will be designated as I-422. Since a direct connection from I-22 to I-422 will not be possible, I-222 has been proposed, as well. Several U.S. Highways also pass through the state, such as U.S. Route 11 (US-11), US-29, US-31, US-43, US-45, US-72, US-78, US-80, US-82, US-84, US-90, US-98, US-231, US-278, US-280, US-331, US-411, and US-431."
  },
  {
   "id": "wikipedia-01421",
   "text": "This is followed by Tagalog, with 99,035 speakers, German, with 80,050 speakers, French, with 72,150 native speakers, and Punjabi, with 68,695 speakers. The 2006 census found that English, with 2,576,670 native speakers, was the most common mother tongue of Albertans, representing 79.99% of the population. The next most common mother tongues were Chinese with 97,275 native speakers (3.02%), followed by German with 84,505 native speakers (2.62%) and French with 61,225 (1.90%). Other mother tongues include: Punjabi, with 36,320 native speakers (1.13%); Tagalog, with 29,740 (0.92%); Ukrainian, with 29,455 (0.91%); Spanish, with 29,125 (0.90%); Polish, with 21,990 (0.68%); Arabic, with 20,495 (0.64%); Dutch, with 19,980 (0.62%); and Vietnamese, with 19,350 (0.60%). The most common aboriginal language is Cree 17,215 (0.53%). Other common mother tongues include Italian with 13,095 speakers (0.41%); Urdu with 11,275 (0.35%); and Korean with 10,845 (0.33%); then Hindi 8,985 (0.28%); Persian 7,700 (0.24%); Portuguese 7,205 (0.22%); and Hungarian 6,770 (0.21%). According to Statistics Canada, Alberta is home to the second-highest proportion (2%) of Francophones in western Canada (after Manitoba). Despite this, relatively few Albertans claim French as their mother tongue. Many of Alberta's French-speaking residents live in the central and northwestern regions of the province, after migration from other areas of Canada or descending from Métis. Ethnicity Alberta has considerable ethnic diversity. In line with the rest of Canada, many are descended from immigrants of Western European nations, notably England, Scotland, Ireland, Wales and France, but large numbers later came from other regions of Europe, notably Germany, Ukraine and Scandinavia."
  },
  {
   "id": "arxiv-00634",
   "text": "in this representation the larger inclination of the ridges mean lower propagation speeds and vice versa . note , that by projecting the velocities , we are able to separate the modes only in the magnetically dominated atmosphere , i.e. above the solid line in fig . [ fig : modes ] . the figure shows how the incident fast mode wave propagates to the equipartition layer and then splits into several components . the alfvn wave is produced by mode conversion above 0.2 mm ( left panel ) and propagates upwards with the ( rapid ) alfvn speed , confirmed by almost vertical inclination of the ridges . conversely , the essentially magnetic fast - mode low- wave produced in the upper atmosphere ( middle panel ) is reflected , and its velocity variations in the upper layers vanish with height . the ( acoustic ) slow - mode low- wave escapes to the upper atmosphere tunnelling over the cut - off layer due to the field inclination of . the amplitudes of the velocity variations of the alfvn wave are comparable to those of the slow wave . left panel : log of the ratio to for projected velocities and magnetic field variations , averaged over all , as a function of . black line : fast mode ( projection ) ; red line : alfvn mode ( ) ; blue line : slow mode ( ) . right panel : phase shift between the projected variations of and , as a function of for selected ."
  }
 ]
}

WORK = "/kaggle/working"
# Placeholders are filled per-model by make_capinfer.py.
MODEL_SLUG = "qwen-lm/qwen-3/transformers/0.6b/1"          # e.g. qwen-lm/qwen-3/transformers/0.6b/1
ADAPTER_NAME = ""      # e.g. qwen3-0.6b
# Dataset slug that mounts the capacity adapters (default idalextan/cap-upload;
# make_capinfer.py may bake an alternative dataset for a specific run).
DATASET = "cap-upload"
MODEL_PATH = f"/kaggle/input/models/{MODEL_SLUG}"
ADAPTER_DIR = f"/kaggle/input/{DATASET}/{ADAPTER_NAME}"
# No-adapter mode (baked True by `make_capinfer.py --no-adapter`): run the BASE
# model alone — no LoRA merge — with the SAME protocol, schema-prefix injection,
# tolerant parser and decode batching as the adapter runs (base-baseline leg).
# When True, ADAPTER_DIR/resolve_adapter_dir are unused and the adapter dataset
# is not required. False keeps the standard capacity-line behaviour.
NO_ADAPTER = True
# Smoke cap: make_capinfer.py bakes --limit into LIMIT (kernel env is not
# settable at push time, so per-run limits are baked at generation).
LIMIT = None
# Batch size for length-sorted decode (measured 22/85/164 tok/s at bs 1/4/8 on
# 0.6B). Per-model override baked by the generator.
BATCH_SIZE = 4
# Prompt wrapper: "chat" applies the Qwen3 chat template (reproduction-line
# protocol); "plain" uses the capacity training format verbatim
# (`STUDENT_PROMPT\n<passage>\n\n`, no chat markers — train_unsloth build_text).
# Schema-prefix injection and sampling are identical in both, so the two differ
# only in the wrapper. See issue 08 Finding (2026-09-06).
PROTOCOL = "chat"
# Stop decoding as soon as the top-level JSON object closes (bracket balance
# returns to zero after the forced `{"entities": [` prefix) instead of decoding
# to max_new_tokens. Plain prompts have no chat stop marker and would otherwise
# generate ~8k tokens per passage (~5.6 h for 200); early stop cuts wall time
# to ~chat levels. Backstop MAX_NEW_TOKENS still applies if JSON never closes.
EARLY_STOP = False
MAX_NEW_TOKENS = 8192


def log(msg: str) -> None:
    print(msg, flush=True)


def _tree(roots=("/kaggle/input",), depth=4) -> list[str]:
    lines = []
    for root in roots:
        if not os.path.isdir(root):
            lines.append(f"MISSING {root}")
            continue
        base = root.count(os.sep)
        for cur, dirs, files in os.walk(root):
            if cur.count(os.sep) - base > depth:
                dirs[:] = []
                continue
            lines.append(f"DIR  {cur}")
            for f in sorted(files)[:8]:
                lines.append(f"  {f}")
    return lines


# Dataset-source mount layouts observed on Kaggle (slug-root and the
# /datasets/<owner>/<slug> forms); model sources mount under /kaggle/input/models.
def _candidates(rel: str) -> list[str]:
    return [
        f"/kaggle/input/{rel}",
        f"/kaggle/input/datasets/idalextan/{rel}",
        f"/kaggle/input/datasets/{rel}",
    ]


def resolve_adapter_dir() -> str:
    """Return the adapter dir the dataset actually mounted at.

    Spawned workers re-import this module and re-evaluate module constants, so
    path resolution must be a pure function both the parent (check_env) and
    each worker call — never a mutated module global.
    """
    for cand in _candidates(f"{DATASET}/{ADAPTER_NAME}"):
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    return ADAPTER_DIR


def check_env() -> None:
    log("=== env ===")
    import torch

    log(f"torch: {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)}  count={torch.cuda.device_count()}")
    log(f"MODEL_PATH={MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"required base model missing: {MODEL_PATH}")
    if NO_ADAPTER:
        log("NO_ADAPTER: base model only (no LoRA merge expected)")
        return
    adapter = resolve_adapter_dir()
    log(f"ADAPTER_DIR={adapter}")
    if not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        log("--- /kaggle/input tree ---\n" + "\n".join(_tree()))
        raise SystemExit(f"no adapter_config.json under {adapter}")


def _array_span(text: str, key: str, stop_key: str | None = None) -> str | None:
    """Return the JSON array value for `key` (with surrounding brackets).

    Respects string/nesting depth. If the array never closes, stop at the
    next `stop_key` (so an unterminated entities array does not swallow the
    relationships that follow) and return the unclosed prefix — the caller
    repairs it by appending ']'. Returns None if the key or its array is
    absent.
    """
    i = text.find('"' + key + '"')
    if i < 0:
        return None
    j = text.find("[", i)
    if j < 0:
        return None
    limit = len(text)
    if stop_key:
        k = text.find('"' + stop_key + '"', i + len(key))
        if k > j:
            limit = k
    depth = 0
    in_str = False
    esc = False
    for idx in range(j, min(limit + 1, len(text))):
        c = text[idx]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[j : idx + 1]
    return text[j:limit]


def _fix_array(seg: str | None, item_key: str) -> list[dict]:
    """Best-effort parse of a possibly-broken array into a list of dicts.

    Tries, in order: strict parse, append ']', prepend '[', per-object salvage
    (each non-nested {...} parsed independently with light char repairs).
    Only dicts carrying `item_key` (e.g. "title" or "source") are kept.
    """
    if not seg:
        return []
    cands = [seg]
    if not seg.endswith("]"):
        cands.append(seg + "]")
    if not seg.startswith("["):
        cands.append("[" + seg)
    cands += [seg.replace(")", "}"), seg.replace("),", "},")]
    for c in cands:
        try:
            v = json.loads(c)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, list):
            return [d for d in v if isinstance(d, dict) and d.get(item_key)]
    # per-object salvage
    out = []
    for o in re.findall(r"\{[^{}]*\}", seg):
        try:
            d = json.loads(o.replace(")", "}"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get(item_key):
            out.append(d)
    return out


def _dict_list(items) -> list[dict]:
    """Return only the dict entries of a parsed JSON array.

    A strict parse can succeed while the model still emits schema-invalid rows
    (e.g. ``"relationships": ["PRISON GOVERNOR", ...]``). Dropping non-dict
    entries deterministically keeps predictions schema-clean so downstream
    referent tooling never crashes on a bare string.
    """
    if not isinstance(items, list):
        return []
    return [d for d in items if isinstance(d, dict)]


def _parse_answer(raw: str) -> tuple[dict, str]:
    """Parse a completion into {entities, relationships} with repair.

    Returns (parsed, status) where status is "ok" (strict JSON parse) or
    "repaired:N" (recovered N of {entities,relationships} arrays by salvage).
    Never raises; on total failure returns empty lists with status "failed".
    """
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return {"entities": _dict_list(obj.get("entities")),
                    "relationships": _dict_list(obj.get("relationships"))}, "ok"
        except Exception:  # noqa: BLE001
            pass
    entities = _fix_array(_array_span(raw, "entities", stop_key="rel"), "title")
    # the model is unstable about the relationship key name: it has emitted
    # "relationships", the misspelt "relationship", and "relations". Try the
    # candidates in order (the standard one first), bounded by the entities
    # array on the left.
    relationships = []
    for key in ("relationships", "relations", "relationship"):
        span = _array_span(raw, key)
        if span:
            relationships = _fix_array(span, "source")
            if relationships:
                break
    if not relationships:
        # degenerate emission seen in the wild:
        # {"relationship": ["ENGLAND", "RUDDER"], "strength": 9} — a two-element
        # array standing in for source/target. Convert to the canonical form.
        rels = []
        for m in re.finditer(
            r'\{\s*"relationship"\s*:\s*\[("[^"]*")\s*,\s*("[^"]*")\]\s*(?:,\s*"strength"\s*:\s*([0-9.]+))?\s*\}',
            raw,
        ):
            src, tgt, strength = m.groups()
            d = {"source": json.loads(src), "target": json.loads(tgt)}
            if strength:
                d["strength"] = float(strength)
            rels.append(d)
        relationships = rels
    if entities or relationships:
        return {"entities": entities, "relationships": relationships}, "repaired"
    return {"entities": [], "relationships": []}, "failed"


def _truncate_at_json_close(raw: str) -> str:
    """Cut `raw` right after the top-level JSON object closes.

    Scans char-by-char tracking bracket depth, string state and escapes. The
    forced schema prefix `{"entities": [` is always the start of `raw`, so a
    return to depth 0 is the balanced end of the whole object; everything after
    it (runaway plain-mode generation) is dropped. If the object never closes,
    returns `raw` unchanged (the tolerant parser still salvages it).
    """
    depth = 0
    in_str = False
    esc = False
    for i, c in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
            if depth <= 0:
                return raw[: i + 1]
    return raw


def install() -> None:
    """Neutralise torchao only; no transformers/peft downgrade.

    torchao 0.10.0 preinstalled in the Kaggle image makes peft's
    `PeftModel.from_pretrained` hard-fail before reading the adapter.
    Uninstalling it is enough (inference does not use torchao).

    Earlier versions downgraded to the v4-validated matrix (transformers
    4.57.x / peft 0.18.1) but the Kaggle pip backend repeatedly hung mid-
    install on those runs. Inference does not need the training matrix, so we
    now try the preinstalled transformers 5.0 / peft 0.19.1 directly and only
    uninstall torchao (with a hard timeout so a hung backend fails fast).
    """
    import importlib.metadata as md
    import subprocess

    log("=== install stack ===")

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "MISSING"

    for p in ("torchao", "transformers", "peft"):
        log(f"pre {p}: {ver(p)}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
            timeout=60, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log("uninstalled torchao")
    except Exception as e:  # noqa: BLE001
        log(f"torchao uninstall failed (continuing): {e}")
    # No transformers/peft downgrade: the Kaggle pip backend has repeatedly
    # hung mid-install, and inference does not actually need the training
    # matrix — try the preinstalled transformers 5.0 / peft 0.19.1 directly.
    for p in ("torchao", "transformers", "peft"):
        log(f"post {p}: {ver(p)}")


def _infer_worker(rank: int, nprocs: int, passages: list[dict]) -> list[dict]:
    """Run batch inference for `passages` on GPU `rank`. Top-level for spawn.

    Each worker process pins its own CUDA_VISIBLE_DEVICES so two Kaggle T4s are
    used (each loads its own fp16 copy of model+adapter and decodes a half of
    the passages). Writes raw_<id>.txt into WORK as it goes; returns the parsed
    rows so the parent can merge them into predictions.jsonl.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import StoppingCriteria
    from peft import PeftModel

    if EARLY_STOP:
        class _JsonClose(StoppingCriteria):
            """Stop each row as soon as its JSON object closes.

            Generation starts inside the forced schema prefix `{"entities": [`
            (depth 2). Each call decodes only the newly generated token per row
            and updates bracket/string/escape state; a row stops when depth
            returns to 0 (its object is balanced). Returns a per-batch bool
            tensor so finished rows stop decoding while others continue.
            """

            def __init__(self, tokenizer, n):
                super().__init__()
                self.tok = tokenizer
                self.depth = [2] * n
                self.in_str = [False] * n
                self.esc = [False] * n
                self.closed = [False] * n
                self.prev = None

            def __call__(self, input_ids, scores):
                cur = input_ids.shape[1]
                start = cur - 1 if self.prev is None else self.prev
                self.prev = cur
                for i in range(input_ids.shape[0]):
                    if self.closed[i]:
                        continue
                    text = self.tok.decode(
                        input_ids[i, start:cur].tolist(), skip_special_tokens=True)
                    d, s, e = self.depth[i], self.in_str[i], self.esc[i]
                    for c in text:
                        if s:
                            if e:
                                e = False
                            elif c == "\\":
                                e = True
                            elif c == '"':
                                s = False
                            continue
                        if c == '"':
                            s = True
                        elif c in "[{":
                            d += 1
                        elif c in "]}":
                            d -= 1
                            if d <= 0:
                                self.closed[i] = True
                                d = 0
                                break
                    self.depth[i], self.in_str[i], self.esc[i] = d, s, e
                return torch.tensor(self.closed, dtype=torch.bool,
                                    device=input_ids.device)

    adapter_dir = None if NO_ADAPTER else resolve_adapter_dir()
    log(f"[gpu{rank}] worker on {torch.cuda.get_device_name(0)}; "
        f"{len(passages)} passages; adapter={adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"[gpu{rank}] tokenizer loaded")
    # FP16 load (bf16 also works — decode is GEMV-bound so dtype does not matter
    # at bs=1 — but fp16 is the training-validated path).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True,
    )
    log(f"[gpu{rank}] base model loaded")
    if adapter_dir is not None:
        # Adapter weights are fp32 (Unsloth saves) while base is fp16; torch_dtype
        # casts them on merge. PeftModel.load accepts unknown peft-0.20 config keys.
        model = PeftModel.from_pretrained(
            model, adapter_dir, torch_dtype=torch.float16
        ).merge_and_unload()
    model.eval()
    log(f"[gpu{rank}] model ready "
        f"(adapter={'merged' if adapter_dir is not None else 'none — base baseline'})")

    prompt = INPUTS["student_prompt"]
    rows: list[dict] = []
    # Batch decode: bs=1 decode is GEMV-bound (~20 tok/s on T4; cublas gemv2T/
    # gemvx kernels, memory-bandwidth limited, fp16 vs bf16 makes no diff).
    # Batching N sequences turns weight application into a GEMM: measured
    # 22 -> 85 -> 164 aggregate tok/s at bs 1/4/8. Passages differ in length,
    # so sort by length and pack consecutive ones to keep padding waste small.
    sorted_ps = sorted(enumerate(passages), key=lambda ep: len(ep[1]["text"]))
    schema_prefix = '{"entities": ['
    prefix_ids = tokenizer.encode(schema_prefix, add_special_tokens=False)
    batch_size = int(os.environ.get("BATCH_SIZE", str(BATCH_SIZE)))

    def run_batch(batch: list[tuple[int, dict]]) -> None:
        texts = []
        for _, p in batch:
            content = f"{prompt}\n{p['text']}"
            if PROTOCOL == "plain":
                # training-faithful: no chat markers, blank line before output
                texts.append(content + "\n\n")
            else:
                texts.append(tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False))
        enc = tokenizer(texts, return_tensors="pt", padding=True)
        prefix_t = torch.tensor([prefix_ids] * len(batch), dtype=torch.long,
                                device="cuda")
        gen_input = torch.cat([enc["input_ids"].to("cuda"), prefix_t], dim=1)
        attn = torch.cat([enc["attention_mask"].to("cuda"),
                          torch.ones(len(batch), prefix_t.shape[1],
                                     dtype=torch.long, device="cuda")], dim=1)
        # True (unpadded) prompt length per row — the generated tail starts at
        # mask.sum(). Slice from prompt_end = mask.sum() - prefix_len so the
        # schema prefix stays in the decoded raw (parser expects it).
        prompt_ends = (attn.sum(dim=1) - len(prefix_ids)).tolist()
        kwargs = {}
        if EARLY_STOP:
            kwargs["stopping_criteria"] = [_JsonClose(tokenizer, len(batch))]
        with torch.no_grad():
            gen = model.generate(
                input_ids=gen_input, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=0.7, top_p=0.95, repetition_penalty=1.15,
                **kwargs,
            )
        for bi, (_orig_i, p) in enumerate(batch):
            out_ids = gen[bi][int(prompt_ends[bi]):]
            raw = tokenizer.decode(out_ids, skip_special_tokens=True)
            if EARLY_STOP:
                raw = _truncate_at_json_close(raw)
            with open(os.path.join(WORK, f"raw_{p['id']}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(raw)
            try:
                parsed, status = _parse_answer(raw)
            except Exception as e:  # noqa: BLE001
                status = f"except:{e}"
                parsed = {"entities": [], "relationships": []}
            if status != "ok":
                log(f"[gpu{rank}][{p['id']}]: PARSE {status} len={len(raw)}")
            rows.append({"id": p["id"],
                         "entities": parsed.get("entities", []),
                         "relationships": parsed.get("relationships", [])})
            log(f"[gpu{rank}][{p['id']}]: entities={len(rows[-1]['entities'])} "
                f"rels={len(rows[-1]['relationships'])} raw_len={len(raw)}")

    for start in range(0, len(sorted_ps), batch_size):
        run_batch(sorted_ps[start:start + batch_size])
    log(f"[gpu{rank}] worker done, {len(rows)} rows")
    return rows


def _load_passages() -> list[dict]:
    """Load the 200-passage sample from the mounted dataset.

    Sources, in priority order: PASSAGES_FILE env (explicit override), then the
    cap-upload / 200-passages datasets under any of the path layouts Kaggle has
    used (slug-root vs /datasets/<owner>/<slug>). Each line must be
    {"id": ..., "text": ...}. Hard-fails (no silent embedded fallback) because a
    wrong passage set would invalidate the eval.
    """
    cands = [os.environ.get("PASSAGES_FILE")] + _candidates(
        "cap-upload/sample_200_passages.jsonl") + _candidates(
        "200-passages/sample_200_passages.jsonl")
    for pf in cands:
        if pf and os.path.exists(pf):
            log(f"reading passages from {pf}")
            out = []
            with open(pf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        out.append({"id": r["id"], "text": r["text"]})
            return out
    raise SystemExit(
        "no sample_200_passages.jsonl under /kaggle/input; tree:\n"
        + "\n".join(_tree()))


def infer(limit: int | None = None) -> None:
    passages = _load_passages()
    if limit:
        passages = passages[:limit]
    log(f"=== inference on {len(passages)} eval passages (max_new_tokens=8192) ===")
    import torch

    n_gpu = torch.cuda.device_count()
    log(f"CUDA devices available: {n_gpu}")
    # Spawn one worker per GPU (each loads its own model copy + decode half the
    # passages -> ~2x wall throughput on two T4s). Single-GPU falls back to one
    # worker (rank 0).
    nprocs = n_gpu if n_gpu >= 2 else 1
    if nprocs == 1:
        rows = _infer_worker(0, 1, passages)
    else:
        from torch import multiprocessing as mp
        chunks = [passages[i::nprocs] for i in range(nprocs)]
        ctx = mp.get_context("spawn")
        procs = []
        results: dict[int, list[dict]] = {}
        q = ctx.Queue()
        for rank in range(nprocs):
            p = ctx.Process(target=_worker_send, args=(rank, chunks[rank], q, nprocs))
            p.start()
            procs.append(p)
        for _ in range(nprocs):
            rank, part = q.get()
            results[rank] = part
        for p in procs:
            p.join(timeout=3600)
        rows = []
        for rank in range(nprocs):
            rows.extend(results.get(rank, []))

    pred_path = os.path.join(WORK, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {pred_path} ({len(rows)} rows)")
    with open(os.path.join(WORK, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("id,entities,relationships\n")
        for r in rows:
            f.write(f"{r['id']},{len(r['entities'])},{len(r['relationships'])}\n")


def _worker_send(rank: int, passages: list[dict], q, nprocs: int) -> None:
    """spawn target: run worker and ship its rows back on the queue."""
    try:
        q.put((rank, _infer_worker(rank, nprocs, passages)))
    except Exception as e:  # noqa: BLE001
        log(f"[gpu{rank}] worker failed: {e}")
        q.put((rank, []))


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="infer only the first N passages (smoke check); "
                             "defaults to baked LIMIT then env TRACE_LIMIT")
    args = parser.parse_args(argv)
    check_env()
    install()
    n = args.limit if args.limit is not None else LIMIT
    if n is None and os.environ.get("TRACE_LIMIT"):
        n = int(os.environ["TRACE_LIMIT"])
    log(f"=== inference {'(limit=%d)' % n if n else '(all passages)'} ===")
    infer(limit=n)
    log("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
