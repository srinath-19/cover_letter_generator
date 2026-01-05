import argparse
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


# -------------------------
# Schema
# -------------------------
class Section(BaseModel):
    sec_id: str
    title: str
    start_line: int
    end_line: int
    text: str


class Outline(BaseModel):
    source: str
    sections: List[Section] = Field(default_factory=list)


# -------------------------
# Heuristics for headings
# -------------------------
COMMON_HEADINGS = {
    "about the role", "about you", "about us", "about the company",
    "responsibilities", "description", "what you'll do", "what you will do", "role",
    "requirements", "qualifications", "minimum qualifications", "preferred qualifications",
    "preferred", "nice to have", "benefits", "compensation", "salary",
    "location", "work authorization", "equal opportunity", "eeo",
    "how to apply", "application", "interview process", 
}

def normalize_heading(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def looks_like_heading(line: str) -> bool:
    raw = line.strip()
    if not raw:
        return False

    if len(raw) > 80:
        return False

    if re.fullmatch(r"[-•*]+", raw):
        return False

    if raw.endswith(":") and 2 <= len(raw[:-1].strip()) <= 60:
        return True

    low = normalize_heading(raw)
    if low in COMMON_HEADINGS:
        return True

    letters = re.sub(r"[^A-Za-z]", "", raw)
    if letters and letters.isupper() and len(raw.split()) <= 6:
        return True

    if len(raw.split()) <= 6 and not re.search(r"[.!?]$", raw):
        if sum(1 for c in raw if c.isalpha()) >= 6 and raw[0].isalpha():
            words = [w for w in raw.split() if w]
            caps = sum(1 for w in words if w[0].isupper())
            if caps >= max(2, len(words) - 1):
                return True

    return False


def load_lines(txt_path: Path) -> List[str]:
    text = txt_path.read_text(encoding="utf-8")
    return text.splitlines()


# -------------------------
# Boilerplate trimming
# -------------------------
BOILERPLATE_MARKERS = [
    "equal opportunity employer",
    "does not discriminate",
    "inclusive culture",
    "workplace accommodation",
    "hiring process",
    "compensation reflects",
    "base pay for this position",
    "total compensation",
    "employee benefits",
    "position will remain posted",
    "applicants should apply",
]

def trim_boilerplate(section_text: str) -> str:
    lines = section_text.splitlines()
    out = []
    for line in lines:
        low = line.lower()
        if any(m in low for m in BOILERPLATE_MARKERS):
            break
        out.append(line)
    return "\n".join(out).strip()

def keep_only_leading_bullets(section_text: str) -> str:
    """
    Keep only the initial bullet list and stop at the first non-bullet line
    after bullets begin. This prevents EEO/compensation paragraphs from
    being included in Qualifications sections.
    """
    lines = section_text.splitlines()
    kept = []
    started = False
    for line in lines:
        s = line.strip()
        is_bullet = s.startswith(("-", "•", "*"))
        if is_bullet:
            kept.append(line)
            started = True
        else:
            if started:
                break
            # haven't started bullets yet; ignore leading fluff
    return "\n".join(kept).strip()


def build_outline_from_headings(lines: List[str], source: str) -> Outline:
    heading_idxs = []
    for i, line in enumerate(lines, start=1):
        if looks_like_heading(line):
            heading_idxs.append(i)

    if not heading_idxs:
        full_text = "\n".join(lines).strip()
        return Outline(
            source=source,
            sections=[Section(sec_id="S001", title="Full Text", start_line=1, end_line=len(lines), text=full_text)]
        )

    merged = []
    for idx in heading_idxs:
        if not merged or idx - merged[-1] > 2:
            merged.append(idx)

    sections: List[Section] = []
    for s_i, start_idx in enumerate(merged):
        end_idx = (merged[s_i + 1] - 1) if s_i + 1 < len(merged) else len(lines)
        title = lines[start_idx - 1].strip().rstrip(":") or f"Section {s_i+1}"

        body_start = min(start_idx + 1, len(lines))
        body_lines = lines[body_start - 1 : end_idx]
        body_text = "\n".join(body_lines).strip()

        # ### NEW ### Trim junk intelligently
        norm_title = normalize_heading(title)

        # For qualifications sections, keep only bullets first (strong signal)
        if "qualifications" in norm_title:
            body_text = keep_only_leading_bullets(body_text)

        # Then trim boilerplate markers (covers cases where bullets weren't detected)
        body_text = trim_boilerplate(body_text)

        sec_id = f"S{(s_i+1):03d}"
        sections.append(Section(
            sec_id=sec_id,
            title=title,
            start_line=body_start,
            end_line=end_idx,
            text=body_text
        ))

    filtered = []
    for s in sections:
        word_count = len(s.text.split())
        if word_count >= 15 or normalize_heading(s.title) in COMMON_HEADINGS:
            filtered.append(s)

    return Outline(source=source, sections=filtered if filtered else sections)


# -------------------------
# Optional LLM refinement hook
# -------------------------
def refine_with_llm(outline: Outline) -> Outline:
    return outline


def main():
    ap = argparse.ArgumentParser(description="Phase 2: Create section outline JSON from cleaned job-post text.")
    ap.add_argument("clean_txt", help="Path to outputs/clean/<slug>.txt")
    ap.add_argument("--out", default=None, help="Output JSON path (default: outputs/outline/<slug>.json)")
    ap.add_argument("--llm", action="store_true", help="Enable LLM refinement (hook; currently no-op)")
    args = ap.parse_args()

    clean_path = Path(args.clean_txt)
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)

    if args.out is None:
        out_path = Path("outputs") / "outline" / (clean_path.stem + ".json")
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = load_lines(clean_path)
    outline = build_outline_from_headings(lines, source=str(clean_path))

    if args.llm:
        outline = refine_with_llm(outline)

    out_path.write_text(outline.model_dump_json(indent=2), encoding="utf-8")

    print(f"[OK] outline_json: {out_path}")
    print(f"[OK] sections: {len(outline.sections)}")
    print("\n--- Section titles ---")
    for s in outline.sections[:20]:
        print(f"- {s.sec_id}: {s.title} (lines {s.start_line}-{s.end_line})")


if __name__ == "__main__":
    main()
