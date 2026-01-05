import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel, Field


# -------------------------
# Schemas
# -------------------------
class Requirement(BaseModel):
    req_id: str
    req_text: str
    priority: str  # "must" | "preferred" | "other"
    type: str      # "eligibility" | "skill" | "domain" | "data_scale" | "work_style" | "other"
    keywords: List[str] = Field(default_factory=list)
    source_section: str


class Responsibility(BaseModel):
    resp_id: str
    resp_text: str
    keywords: List[str] = Field(default_factory=list)
    source_section: str


class ExtractedJob(BaseModel):
    source_outline: str
    requirements: List[Requirement] = Field(default_factory=list)
    responsibilities: List[Responsibility] = Field(default_factory=list)


# -------------------------
# Helpers
# -------------------------
TECH_PATTERNS = [
    ("SQL", r"\bsql\b"),
    ("Python", r"\bpython\b"),
    ("R", r"\br\b"),  # standalone 'R' only
    ("SAS", r"\bsas\b"),
    ("Matlab", r"\bmatlab\b"),
    ("Hadoop", r"\bhadoop\b"),
    ("Spark", r"\bspark\b"),
    ("Hive", r"\bhive\b"),
    ("MapReduce", r"\bmap[\s\-]?reduce\b"),
    ("Machine Learning", r"\bmachine learning\b"),
    ("Data Mining", r"\bdata mining\b"),
    ("Big Data", r"\bbig data\b"),
    ("ETL", r"\betl\b"),
    ("Kubernetes", r"\bkubernetes\b|\bk8s\b"),
    ("AWS", r"\baws\b"),
    ("Java", r"\bjava\b"),
    ("Scala", r"\bscala\b"),
    ("Modeling", r"\bmodel(s|ing)?\b"),
    ("Statistics", r"\bstatistic(al|s)?\b"),
]


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def normalize_heading(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def split_bullets(text: str) -> List[str]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("-", "•", "*")):
            item = line.lstrip("-•*").strip()
            if item:
                items.append(item)
    return items


def extract_keywords(text: str) -> List[str]:
    low = text.lower()
    found: List[str] = []
    for label, pat in TECH_PATTERNS:
        if re.search(pat, low):
            found.append(label)
    # stable unique
    return list(dict.fromkeys(found))[:12]


def classify_requirement(req_text: str) -> str:
    low = req_text.lower()

    # eligibility / logistics
    if any(k in low for k in ["relocate", "eligible", "available for full-time", "40 hours", "work authorization"]):
        return "eligibility"

    # skills/tools FIRST (avoid misclassifying as domain)
    if any(k in low for k in ["sql", "python", " sas", "matlab", "hadoop", "spark", "hive", "map-reduce", "mapreduce"]):
        return "skill"

    # data scale
    if any(k in low for k in ["big data", "millions", "100k", "rows", "large quantities", "processing, filtering"]):
        return "data_scale"

    # domain
    if any(k in low for k in ["data science", "machine learning", "data mining", "modeling", "statistical"]):
        return "domain"

    # work style
    if any(k in low for k in ["owning", "execute", "present", "partners", "drive improvements", "automate"]):
        return "work_style"

    return "other"


def sentence_split(text: str) -> List[str]:
    text = normalize_ws(text)
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def extract_responsibilities_from_description(desc_text: str) -> List[str]:
    sents = sentence_split(desc_text)
    keep = []
    for s in sents:
        low = s.lower()
        if any(p in low for p in [
            "you will", "during this internship", "you'll", "will need", "need the ability",
            "build tools", "analyze", "present your findings", "drive improvements"
        ]):
            keep.append(s)

    seen = set()
    out = []
    for s in keep:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:12]


def extract_description_requirements(desc_text: str) -> List[str]:
    low = desc_text.lower()
    reqs = []
    if "master" in low or "phd" in low:
        reqs.append("Currently enrolled in a Master’s or PhD program")
    if "strong modeling" in low or ("modeling" in low and "strong" in low):
        reqs.append("Strong modeling skills")
    return reqs


# -------------------------
# Core extraction
# -------------------------
def load_outline(outline_path: Path) -> Dict:
    return json.loads(outline_path.read_text(encoding="utf-8"))


def extract(outline_path: Path) -> ExtractedJob:
    outline = load_outline(outline_path)
    sections = outline.get("sections", [])

    requirements: List[Requirement] = []
    responsibilities: List[Responsibility] = []

    req_counter = 1
    resp_counter = 1

    for sec in sections:
        sec_id = sec.get("sec_id", "S???")
        title = sec.get("title", "")
        text = sec.get("text", "") or ""
        norm_title = normalize_heading(title)

        # --- Requirements from bullet lists ---
        if "basic qualifications" in norm_title:
            bullets = split_bullets(text)
            for b in bullets:
                requirements.append(Requirement(
                    req_id=f"R{req_counter:03d}",
                    req_text=normalize_ws(b),
                    priority="must",
                    type=classify_requirement(b),
                    keywords=extract_keywords(b),
                    source_section=sec_id
                ))
                req_counter += 1

        elif "preferred qualifications" in norm_title:
            bullets = split_bullets(text)
            for b in bullets:
                requirements.append(Requirement(
                    req_id=f"R{req_counter:03d}",
                    req_text=normalize_ws(b),
                    priority="preferred",
                    type=classify_requirement(b),
                    keywords=extract_keywords(b),
                    source_section=sec_id
                ))
                req_counter += 1

        elif "requirements" in norm_title or "qualifications" in norm_title:
            bullets = split_bullets(text)
            for b in bullets:
                requirements.append(Requirement(
                    req_id=f"R{req_counter:03d}",
                    req_text=normalize_ws(b),
                    priority="other",
                    type=classify_requirement(b),
                    keywords=extract_keywords(b),
                    source_section=sec_id
                ))
                req_counter += 1

        # --- Description: responsibilities + extra inferred requirements ---
        if norm_title == "description":
            # responsibilities
            resp_sents = extract_responsibilities_from_description(text)
            for s in resp_sents:
                responsibilities.append(Responsibility(
                    resp_id=f"P{resp_counter:03d}",
                    resp_text=normalize_ws(s),
                    keywords=extract_keywords(s),
                    source_section=sec_id
                ))
                resp_counter += 1

            # optional: requirements hinted in description (e.g., Master's/PhD)
            for rtxt in extract_description_requirements(text):
                requirements.append(Requirement(
                    req_id=f"R{req_counter:03d}",
                    req_text=rtxt,
                    priority="must",
                    type="eligibility" if "enrolled" in rtxt.lower() else "domain",
                    keywords=extract_keywords(rtxt),
                    source_section=sec_id
                ))
                req_counter += 1

    return ExtractedJob(
        source_outline=str(outline_path),
        requirements=requirements,
        responsibilities=responsibilities
    )


def main():
    ap = argparse.ArgumentParser(description="Phase 3: Extract structured requirements + responsibilities from outline.json")
    ap.add_argument("outline_json", help="Path to outputs/outline/<slug>.json")
    ap.add_argument("--out", default=None, help="Output JSON path (default: outputs/requirements/<slug>.json)")
    args = ap.parse_args()

    outline_path = Path(args.outline_json)
    if not outline_path.exists():
        raise FileNotFoundError(outline_path)

    if args.out is None:
        out_path = Path("outputs") / "requirements" / (outline_path.stem + ".json")
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extracted = extract(outline_path)
    out_path.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")

    print(f"[OK] requirements_json: {out_path}")
    print(f"[OK] requirements: {len(extracted.requirements)}")
    print(f"[OK] responsibilities: {len(extracted.responsibilities)}")
    print("\n--- First few requirements ---")
    for r in extracted.requirements[:8]:
        print(f"- {r.req_id} ({r.priority}/{r.type}): {r.req_text}")


if __name__ == "__main__":
    main()
