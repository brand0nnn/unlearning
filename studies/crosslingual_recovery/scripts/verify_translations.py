"""PHASE 2, STEP 0 -- translation quality control on the records we will PROBE.

Phase 1 only ever used the non-English data as relearning MATERIAL, so translation
quality sat off the critical path. Phase 2 probes IN-LANGUAGE, which puts it squarely
on it: a broken Hindi translation is indistinguishable from "the fact did not recover
in Hindi". This script separates those two.

Design follows KBL's (translate to English FIRST, then judge English-vs-English), but
not their vendor:
  back-translate : facebook/nllb-200-3.3B      (their choice; ungated, cc-by-nc-4.0)
  judge          : BASE Qwen3-8B               (never a fine-tuned checkpoint)
Different families on purpose -- a model that both back-translates and grades its own
back-translation will accept its own errors.

Why judging with Qwen is not circular here: this step scores DATA (TOFU's English
answer vs NLLB's back-translation). The model under study never enters and neither do
its generations, so any Qwen quirk applies uniformly across all 9 languages and cannot
correlate with whether our fine-tuned Qwen recovered a given fact. (A generation-based
SE over model OUTPUTS would be a different matter -- self-preference bias there points
toward our own hypothesis, so that judge must come from another family.)

Two hardening steps, both free at this size:
  * the translate-first design already BLINDS the judge -- it only ever sees English
    and never learns which language an item came from, so no language prior leaks in;
  * COUNTERBALANCED order -- each pair is judged twice with the two texts swapped and
    both must agree, so order bias becomes an explicit uncertainty flag rather than a
    silent coin-flip.

The judge decides by comparing the log-probability of " Yes" against " No" rather than
by generating text: deterministic, unparseable-output-proof, and it side-steps Qwen3's
thinking mode entirely.

    python studies/crosslingual_recovery/scripts/verify_translations.py \
        --langs ar fa fr hi id iw ja ko ru --splits forget retain --n 40

-> studies/crosslingual_recovery/results/phase2/translation_qc.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

_r = Path(__file__).resolve()
while _r != _r.parent and not (_r / "src").is_dir():
    _r = _r.parent
sys.path.insert(0, str(_r))

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer)

from src.data import load_multilingual_tofu as ml
from src.utils.logging_utils import load_config, get_logger, ensure_dir
from src.utils.paths import results_root

logger = get_logger("verify_translations")

NLLB = "facebook/nllb-200-3.3B"
# FLORES-200 codes -- the same mapping Phase 1 used, which is also exactly what NLLB
# wants for forced_bos_token_id. Coverage for all 9 languages was confirmed there.
FLORES_CODE = {"en": "eng_Latn", "fr": "fra_Latn", "id": "ind_Latn", "ru": "rus_Cyrl",
               "hi": "hin_Deva", "fa": "pes_Arab", "ar": "arb_Arab", "iw": "heb_Hebr",
               "ko": "kor_Hang", "ja": "jpn_Jpan"}

# NLLB is SENTENCE-level. TOFU answers are frequently multi-sentence, so translating a
# whole answer in one shot degrades quality noticeably -- the single most likely bug in
# this step. Terminators must cover the scripts we actually have: ASCII, CJK, Arabic
# question mark, and the Devanagari danda.
# CJK terminators are NOT followed by a space, so requiring whitespace after every
# terminator silently left Japanese answers as one long "sentence" -- exactly the
# degradation this split exists to prevent.
_SENT_END = re.compile(r"(?<=[。！？])\s*|(?<=[.!?؟۔।])\s+")


def split_sentences(text, max_chars=400):
    """Sentence-split for NLLB. Falls back to a hard character chunk for text with no
    recognised terminator (some translations drop punctuation entirely)."""
    parts = [p.strip() for p in _SENT_END.split(text.strip()) if p.strip()]
    out = []
    for p in parts or [text.strip()]:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars) or max_chars
            out.append(p[:cut].strip()); p = p[cut:].strip()
        if p:
            out.append(p)
    return out


class BackTranslator:
    """NLLB <lang> -> English, sentence by sentence."""

    def __init__(self, device):
        logger.info("loading %s (downloads ~17.6GB on first run)", NLLB)
        self.tok = AutoTokenizer.from_pretrained(NLLB)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        eng = self.tok.convert_tokens_to_ids("eng_Latn")
        if eng is None or eng == self.tok.unk_token_id:      # older tokenizer API
            eng = self.tok.lang_code_to_id["eng_Latn"]
        self.eng = eng

    @torch.no_grad()
    def to_english(self, text, lang):
        self.tok.src_lang = FLORES_CODE[lang]
        done = []
        for sent in split_sentences(text):
            enc = self.tok(sent, return_tensors="pt", truncation=True,
                           max_length=256).to(self.device)
            gen = self.model.generate(**enc, forced_bos_token_id=self.eng,
                                      max_new_tokens=256, num_beams=4)
            done.append(self.tok.batch_decode(gen, skip_special_tokens=True)[0].strip())
        return " ".join(done)


JUDGE_PROMPT = """You are a strict semantic equivalence judge for English text.
Two texts are equivalent if they state the same facts about the same entities. Ignore
differences in wording, fluency, and style. They are NOT equivalent if any name, date,
title, place, or claim differs, or if one omits a fact the other states.

Text A: {a}

Text B: {b}

Are Text A and Text B semantically equivalent? Answer with one word, Yes or No.
Answer:"""


class Judge:
    """Base Qwen3-8B, scoring ' Yes' vs ' No' instead of generating."""

    def __init__(self, model_name, device):
        logger.info("loading judge %s", model_name)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.device = next(self.model.parameters()).device
        self.yes = self.tok(" Yes", add_special_tokens=False).input_ids[0]
        self.no = self.tok(" No", add_special_tokens=False).input_ids[0]

    @torch.no_grad()
    def _equivalent(self, a, b):
        enc = self.tok(JUDGE_PROMPT.format(a=a, b=b), return_tensors="pt",
                       truncation=True, max_length=2048).to(self.device)
        logits = self.model(**enc).logits[0, -1]
        return bool(logits[self.yes] > logits[self.no])

    def judge(self, a, b):
        """Counterbalanced: judge both orders, require agreement. Returns
        (pass, disagreed) -- a disagreement is flagged, never silently resolved."""
        ab, ba = self._equivalent(a, b), self._equivalent(b, a)
        return (ab and ba), (ab != ba)


_LATIN_NAME = re.compile(r"\b[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+\b")


def name_survival(en_answer, tgt_answer):
    """KBL deliberately leaves personal names untranslated. Either choice works for us,
    but it must be KNOWN and it must be CONSISTENT between the answer and the perturbed
    answers -- the truth ratio is a within-language ratio, so an inconsistency there is
    a real bug, not a curiosity. Reports whether Latin-script multiword names from the
    English answer survive verbatim in the target-language answer."""
    names = _LATIN_NAME.findall(en_answer)
    if not names:
        return None
    kept = [n for n in names if n in tgt_answer]
    return {"names": names, "kept": kept, "fraction_kept": len(kept) / len(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+",
                    default=["ar", "fa", "fr", "hi", "id", "iw", "ja", "ko", "ru"])
    ap.add_argument("--splits", nargs="+", default=["forget", "retain"])
    ap.add_argument("--n", type=int, default=40, help="records per (lang, split)")
    args = ap.parse_args()

    cfg = load_config()
    fl = cfg["tofu"]["forget_level"]
    ml_dir, cache = cfg["tofu"]["ml_cache_dir"], cfg["tofu"]["cache_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bt = BackTranslator(device)
    judge = Judge(cfg["model"]["name"], device)

    out = {"config": {"back_translator": NLLB, "judge": cfg["model"]["name"],
                      "n": args.n, "splits": args.splits}, "languages": {}}

    for split in args.splits:
        cfgname = f"{fl}_perturbed" if split == "forget" else f"{split}_perturbed"
        en = ml.load_perturbed(cfgname, "en", ml_dir, cache)[:args.n]
        for lang in args.langs:
            tgt = ml.load_perturbed(cfgname, lang, ml_dir, cache)[:args.n]
            recs, npass, ndis = [], 0, 0
            for i, (e, t) in enumerate(zip(en, tgt)):
                back_q = bt.to_english(t["question"], lang)
                back_a = bt.to_english(t["answer"], lang)
                q_ok, q_dis = judge.judge(e["question"], back_q)
                a_ok, a_dis = judge.judge(e["answer"], back_a)
                ok = q_ok and a_ok
                npass += int(ok); ndis += int(q_dis or a_dis)
                recs.append({"i": i, "pass": ok, "q_pass": q_ok, "a_pass": a_ok,
                             "disagreed": q_dis or a_dis,
                             "back_question": back_q, "back_answer": back_a,
                             "en_question": e["question"], "en_answer": e["answer"],
                             "names": name_survival(e["answer"], t["answer"])})
                if (i + 1) % 10 == 0:
                    logger.info("  %s/%s %d/%d  pass=%d", lang, split, i + 1,
                                len(tgt), npass)
            key = f"{lang}@{split}"
            out["languages"][key] = {
                "pass_rate": npass / len(recs) if recs else float("nan"),
                "n": len(recs), "n_disagreed": ndis, "records": recs}
            logger.info(">>> %-14s pass=%.3f  (%d/%d)  order-disagreements=%d",
                        key, out["languages"][key]["pass_rate"], npass, len(recs), ndis)

    d = ensure_dir(str(results_root() / "phase2"))
    f = Path(d) / "translation_qc.json"
    json.dump(out, open(f, "w"), indent=2, ensure_ascii=False)
    logger.info("wrote %s", f)
    logger.info("NEXT: hand-check ~30 stratified items -- everything is English by now, "
                "so they are readable. The dangerous error is a false PASS, so check "
                "precision on the passes specifically.")


if __name__ == "__main__":
    main()
