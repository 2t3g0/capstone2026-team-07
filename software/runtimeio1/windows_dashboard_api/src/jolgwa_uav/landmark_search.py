from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_KOREAN_DIGITS = {
    "영": 0,
    "공": 0,
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_KOREAN_UNITS = {"십": 10, "백": 100, "천": 1000}
_NUMBERED_BUILDING_RE = re.compile(
    r"제?([영공일이삼사오육칠팔구십백천]+)"
    r"(?=(?:공학관|공|호관|생활관|기숙사|연구동|관))"
)
_ARABIC_NUMBER_RE = re.compile(r"\d+")
_ENGINEERING_BUILDING_NUMBER_RE = re.compile(r"(\d+)공학관")


def _parse_korean_number(value: str) -> int | None:
    """Parse digit strings and simple Sino-Korean numbers used in building names."""
    if not value:
        return None
    if all(character in _KOREAN_DIGITS for character in value):
        digits = "".join(str(_KOREAN_DIGITS[character]) for character in value)
        return int(digits)

    total = 0
    pending_digit: int | None = None
    for character in value:
        if character in _KOREAN_DIGITS:
            if pending_digit is not None:
                return None
            pending_digit = _KOREAN_DIGITS[character]
            continue
        unit = _KOREAN_UNITS.get(character)
        if unit is None:
            return None
        total += (1 if pending_digit is None else pending_digit) * unit
        pending_digit = None
    return total + (pending_digit or 0)


def normalize_landmark_name(value: str) -> str:
    """Return a comparison key for Korean campus landmark names.

    The key is case-insensitive, ignores spacing and punctuation, and treats common
    numbered-building forms such as ``제 2 공학관`` and ``제이공학관`` as
    equivalent to ``2공학관``.
    """
    if not isinstance(value, str):
        raise TypeError("landmark name must be a string")

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(character for character in normalized if character.isalnum())

    def replace_korean_number(match: re.Match[str]) -> str:
        parsed = _parse_korean_number(match.group(1))
        return match.group(0) if parsed is None else str(parsed)

    normalized = _NUMBERED_BUILDING_RE.sub(replace_korean_number, normalized)
    normalized = _ARABIC_NUMBER_RE.sub(lambda match: str(int(match.group(0))), normalized)
    return re.sub(r"^제(?=\d)", "", normalized)


@dataclass(frozen=True)
class LandmarkDocument:
    landmark_id: str
    name: str
    aliases: tuple[str, ...] = ()
    building_number: str = ""
    search_terms: tuple[str, ...] = ()
    departments: tuple[str, ...] = ()
    organizations: tuple[str, ...] = ()


@dataclass(frozen=True)
class LandmarkCandidate:
    landmark_id: str
    name: str
    score: float
    match_type: str
    matched_text: str
    is_exact: bool


@dataclass(frozen=True)
class LandmarkSearchResult:
    query: str
    normalized_query: str
    candidates: tuple[LandmarkCandidate, ...]
    is_ambiguous: bool

    @property
    def ambiguous(self) -> bool:
        return self.is_ambiguous

    @property
    def best_match(self) -> LandmarkCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def resolved_id(self) -> str | None:
        if self.is_ambiguous or not self.candidates:
            return None
        return self.candidates[0].landmark_id


@dataclass(frozen=True)
class _SearchTerm:
    text: str
    normalized: str
    match_type: str
    weight: float


class LandmarkSearchIndex:
    """Dependency-free exact, alias, and typo-tolerant landmark lookup."""

    def __init__(
        self,
        landmarks: Iterable[LandmarkDocument | Mapping[str, Any]],
        *,
        minimum_score: float = 0.55,
        ambiguity_margin: float = 0.08,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        if not 0.0 <= ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be between 0 and 1")

        self.minimum_score = minimum_score
        self.ambiguity_margin = ambiguity_margin
        self._documents = tuple(self._coerce_document(item) for item in landmarks)
        if not self._documents:
            raise ValueError("landmark catalog must not be empty")
        ids = [document.landmark_id for document in self._documents]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate landmark_id in landmark catalog")
        self._terms = {
            document.landmark_id: self._build_terms(document)
            for document in self._documents
        }

    @classmethod
    def from_catalog(
        cls,
        catalog: Path | str | Mapping[str, Any],
        **kwargs: float,
    ) -> LandmarkSearchIndex:
        if isinstance(catalog, (str, Path)):
            payload = json.loads(Path(catalog).read_text(encoding="utf-8"))
        else:
            payload = catalog
        landmarks = payload.get("landmarks")
        if not isinstance(landmarks, list):
            raise ValueError("landmark catalog must contain a landmarks list")
        return cls(landmarks, **kwargs)

    def search(self, query: str, *, limit: int = 5) -> LandmarkSearchResult:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        normalized_query = normalize_landmark_name(query)
        if not normalized_query:
            return LandmarkSearchResult(query, normalized_query, (), False)
        requested_engineering_numbers = set(
            _ENGINEERING_BUILDING_NUMBER_RE.findall(normalized_query)
        )

        candidates: list[LandmarkCandidate] = []
        for document in self._documents:
            document_engineering_numbers = {
                number
                for term in self._terms[document.landmark_id]
                for number in _ENGINEERING_BUILDING_NUMBER_RE.findall(term.normalized)
            }
            if (
                requested_engineering_numbers
                and document_engineering_numbers
                and requested_engineering_numbers.isdisjoint(
                    document_engineering_numbers
                )
            ):
                continue
            term, score = max(
                (
                    (term, self._score(normalized_query, term) * term.weight)
                    for term in self._terms[document.landmark_id]
                ),
                key=lambda item: (item[1], self._match_priority(item[0].match_type)),
            )
            if score >= self.minimum_score:
                candidates.append(
                    LandmarkCandidate(
                        landmark_id=document.landmark_id,
                        name=document.name,
                        score=round(min(score, 1.0), 6),
                        match_type=term.match_type,
                        matched_text=term.text,
                        is_exact=normalized_query == term.normalized,
                    )
                )

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                -self._match_priority(candidate.match_type),
                candidate.landmark_id,
            )
        )
        exact_candidates = [candidate for candidate in candidates if candidate.is_exact]
        if exact_candidates:
            candidates = exact_candidates
        visible = tuple(candidates[:limit])
        return LandmarkSearchResult(
            query=query,
            normalized_query=normalized_query,
            candidates=visible,
            is_ambiguous=self._is_ambiguous(candidates),
        )

    @staticmethod
    def _coerce_document(
        item: LandmarkDocument | Mapping[str, Any],
    ) -> LandmarkDocument:
        if isinstance(item, LandmarkDocument):
            document = item
        else:
            landmark_id = str(item.get("id", item.get("landmark_id", ""))).strip()
            name = str(item.get("name_ko", item.get("name", ""))).strip()
            aliases_value = item.get("aliases", ())
            if isinstance(aliases_value, (str, bytes)) or not isinstance(
                aliases_value, Iterable
            ):
                raise ValueError(f"aliases for {landmark_id or '<unknown>'} must be a list")
            document = LandmarkDocument(
                landmark_id=landmark_id,
                name=name,
                aliases=tuple(str(alias).strip() for alias in aliases_value),
                building_number=str(item.get("building_number", "")).strip(),
                search_terms=LandmarkSearchIndex._string_tuple(
                    item.get("search_terms", ())
                ),
                departments=LandmarkSearchIndex._string_tuple(
                    item.get("departments", ())
                ),
                organizations=LandmarkSearchIndex._string_tuple(
                    item.get("organizations", ())
                ),
            )
        if not document.landmark_id or not document.name:
            raise ValueError("every landmark requires an id and name")
        return document

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            return ()
        return tuple(str(item).strip() for item in value if str(item).strip())

    @staticmethod
    def _build_terms(document: LandmarkDocument) -> tuple[_SearchTerm, ...]:
        source = [
            (document.name, "exact_name", 1.0),
            *((alias, "alias", 0.99) for alias in document.aliases),
        ]
        if document.building_number:
            source.append((document.building_number, "building_number", 0.94))
        source.extend(
            (term, "search_term", 0.88) for term in document.search_terms
        )
        source.extend(
            (department, "department", 0.84)
            for department in document.departments
        )
        source.extend(
            (organization, "organization", 0.82)
            for organization in document.organizations
        )
        source.append((document.landmark_id, "landmark_id", 0.94))

        terms: dict[str, _SearchTerm] = {}
        for text, match_type, weight in source:
            normalized = normalize_landmark_name(text)
            if not normalized:
                continue
            term = _SearchTerm(text, normalized, match_type, weight)
            existing = terms.get(normalized)
            if existing is None or weight > existing.weight:
                terms[normalized] = term
        return tuple(terms.values())

    @classmethod
    def _score(cls, query: str, term: _SearchTerm) -> float:
        if query == term.normalized:
            return 1.0

        shorter, longer = sorted((query, term.normalized), key=len)
        containment = 0.0
        if len(shorter) >= 2 and shorter in longer:
            containment = 0.72 + 0.20 * (len(shorter) / len(longer))

        distance = cls._damerau_levenshtein(query, term.normalized)
        edit_similarity = 1.0 - distance / max(len(query), len(term.normalized))
        return max(containment, edit_similarity)

    @staticmethod
    def _damerau_levenshtein(left: str, right: str) -> int:
        """Optimal-string-alignment distance, including adjacent transpositions."""
        rows = len(left) + 1
        columns = len(right) + 1
        matrix = [[0] * columns for _ in range(rows)]
        for row in range(rows):
            matrix[row][0] = row
        for column in range(columns):
            matrix[0][column] = column

        for row in range(1, rows):
            for column in range(1, columns):
                substitution_cost = left[row - 1] != right[column - 1]
                matrix[row][column] = min(
                    matrix[row - 1][column] + 1,
                    matrix[row][column - 1] + 1,
                    matrix[row - 1][column - 1] + substitution_cost,
                )
                if (
                    row > 1
                    and column > 1
                    and left[row - 1] == right[column - 2]
                    and left[row - 2] == right[column - 1]
                ):
                    matrix[row][column] = min(
                        matrix[row][column], matrix[row - 2][column - 2] + 1
                    )
        return matrix[-1][-1]

    def _is_ambiguous(self, candidates: list[LandmarkCandidate]) -> bool:
        if len(candidates) < 2:
            return False
        first, second = candidates[:2]
        if first.is_exact:
            return second.is_exact
        return first.score - second.score <= self.ambiguity_margin

    @staticmethod
    def _match_priority(match_type: str) -> int:
        return {
            "exact_name": 4,
            "alias": 3,
            "building_number": 2,
            "search_term": 2,
            "department": 2,
            "organization": 2,
            "landmark_id": 1,
        }[match_type]


# Shorter alias for callers that do not need to distinguish the index type.
LandmarkSearch = LandmarkSearchIndex
