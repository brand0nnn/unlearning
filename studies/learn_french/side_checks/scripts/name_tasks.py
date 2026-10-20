"""Plan 2 prompts and string metrics. Stdlib only, so the local readers can import it.

TASKS        copy in each script, Latin -> katakana / Cyrillic, and back.
TEMPLATES    one fixed instruction per task kind, in each of the five study languages,
             each with the same one-shot example (Clara Hoffmann, in no name set) so the
             answer format is pinned and exact match means something.
METRICS      exact (verbatim), exact_loose (case, NFKC and name separators ・＝=- and
             spaces ignored; any accepted rendering counts), and character error rate
             (Levenshtein / reference length, on the loose form, best accepted rendering).

The ja/ru/id instructions are hand-written and should be checked by a fluent reader.
"""
import re
import unicodedata

INSTR_LANGS = ["en", "fr", "id", "ja", "ru"]
SCRIPTS = ["lat", "ja", "ru"]
# task -> (input script, output script, template kind)
TASKS = {
    "copy_lat": ("lat", "lat", "copy"),
    "copy_ja": ("ja", "ja", "copy"),
    "copy_ru": ("ru", "ru", "copy"),
    "lat2ja": ("lat", "ja", "to_ja"),
    "lat2ru": ("lat", "ru", "to_ru"),
    "ja2lat": ("ja", "lat", "to_lat"),
    "ru2lat": ("ru", "lat", "to_lat"),
}
EXAMPLE = {"lat": "Clara Hoffmann", "ja": "クララ・ホフマン", "ru": "Клара Хоффман"}

TEMPLATES = {
    "en": {
        "copy": "Repeat this name exactly as it is written. Answer with the name only. "
                "Example: {ex_in} -> {ex_out}. Name: {name}",
        "to_ja": "Write this name in Japanese katakana. Answer with the name only. "
                 "Example: {ex_in} -> {ex_out}. Name: {name}",
        "to_ru": "Write this name in Russian Cyrillic letters. Answer with the name only. "
                 "Example: {ex_in} -> {ex_out}. Name: {name}",
        "to_lat": "Write this name in Latin letters. Answer with the name only. "
                  "Example: {ex_in} -> {ex_out}. Name: {name}",
    },
    "fr": {
        "copy": "Recopiez ce nom exactement tel qu'il est écrit. Répondez uniquement par le "
                "nom. Exemple : {ex_in} -> {ex_out}. Nom : {name}",
        "to_ja": "Écrivez ce nom en katakana japonais. Répondez uniquement par le nom. "
                 "Exemple : {ex_in} -> {ex_out}. Nom : {name}",
        "to_ru": "Écrivez ce nom en alphabet cyrillique russe. Répondez uniquement par le "
                 "nom. Exemple : {ex_in} -> {ex_out}. Nom : {name}",
        "to_lat": "Écrivez ce nom en alphabet latin. Répondez uniquement par le nom. "
                  "Exemple : {ex_in} -> {ex_out}. Nom : {name}",
    },
    "id": {
        "copy": "Tulis ulang nama ini persis seperti yang tertulis. Jawab hanya dengan nama "
                "tersebut. Contoh: {ex_in} -> {ex_out}. Nama: {name}",
        "to_ja": "Tulis nama ini dalam katakana Jepang. Jawab hanya dengan nama tersebut. "
                 "Contoh: {ex_in} -> {ex_out}. Nama: {name}",
        "to_ru": "Tulis nama ini dalam aksara Sirilik Rusia. Jawab hanya dengan nama "
                 "tersebut. Contoh: {ex_in} -> {ex_out}. Nama: {name}",
        "to_lat": "Tulis nama ini dalam huruf Latin. Jawab hanya dengan nama tersebut. "
                  "Contoh: {ex_in} -> {ex_out}. Nama: {name}",
    },
    "ja": {
        "copy": "この名前を書かれたとおり正確に繰り返してください。名前だけを答えてください。"
                "例：{ex_in} -> {ex_out}　名前：{name}",
        "to_ja": "この名前をカタカナで書いてください。名前だけを答えてください。"
                 "例：{ex_in} -> {ex_out}　名前：{name}",
        "to_ru": "この名前をロシア語のキリル文字で書いてください。名前だけを答えてください。"
                 "例：{ex_in} -> {ex_out}　名前：{name}",
        "to_lat": "この名前をラテン文字（ローマ字）で書いてください。名前だけを答えてください。"
                  "例：{ex_in} -> {ex_out}　名前：{name}",
    },
    "ru": {
        "copy": "Повторите это имя точно так, как оно написано. Ответьте только именем. "
                "Пример: {ex_in} -> {ex_out}. Имя: {name}",
        "to_ja": "Напишите это имя японской катаканой. Ответьте только именем. "
                 "Пример: {ex_in} -> {ex_out}. Имя: {name}",
        "to_ru": "Напишите это имя кириллицей. Ответьте только именем. "
                 "Пример: {ex_in} -> {ex_out}. Имя: {name}",
        "to_lat": "Напишите это имя латиницей. Ответьте только именем. "
                  "Пример: {ex_in} -> {ex_out}. Имя: {name}",
    },
}


def instruction(instr_lang, task, name):
    src, tgt, kind = TASKS[task]
    return TEMPLATES[instr_lang][kind].format(ex_in=EXAMPLE[src], ex_out=EXAMPLE[tgt],
                                              name=name[src])


def references(name, task):
    """Accepted outputs, reference first. A copy accepts only the input itself."""
    src, tgt, _ = TASKS[task]
    if src == tgt:
        return [name[tgt]]
    return [name[tgt]] + list(name.get("alts", {}).get(tgt, []))


def name_group(name):
    """Analysis group: a / b / c_real / c_real_overlap / c_invented / c_invented_overlap."""
    if name["set"] != "c":
        return name["set"]
    g = "c_real" if name.get("real") else "c_invented"
    return g + ("_overlap" if name.get("overlap") else "")


# ---- metrics ------------------------------------------------------------------------
_LABEL = re.compile(r"^(?:name|nom|nama|名前|имя|answer|réponse|jawaban|答え|ответ)\s*[:：]\s*",
                    re.IGNORECASE)
_WRAP = "\"'“”‘’«»「」『』()（）[]【】*`"
_QUOTED = re.compile(r"[「『“\"«]\s*(.+?)\s*[」』”\"»]")
_SEP = re.compile(r"[\s・･＝=\-‐‑–—_·]+")


def clean_prediction(gen):
    """The name the model gave: first non-empty line, after any '->', minus a label,
    wrapping quotes and trailing punctuation."""
    line = next((ln.strip() for ln in gen.splitlines() if ln.strip()), "")
    for arrow in ("->", "→"):
        if arrow in line:
            line = line.split(arrow)[-1].strip()
    line = _LABEL.sub("", line).strip()
    quoted = _QUOTED.search(line)          # 「バジル」です -> バジル
    if quoted:
        line = quoted.group(1)
    line = line.strip(_WRAP).strip()
    line = line.rstrip(".。,、!！;；").strip().strip(_WRAP).strip()
    return line


def loose(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return _SEP.sub(" ", s).strip()


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score_prediction(pred, refs):
    lp = loose(pred)
    cers = [levenshtein(lp, loose(r)) / max(1, len(loose(r))) for r in refs]
    return {"exact": pred in refs,
            "exact_loose": lp in {loose(r) for r in refs},
            "cer": min(cers)}
