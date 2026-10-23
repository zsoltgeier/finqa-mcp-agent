"""Strict financial-number parsing and record-local, provenance-aware arithmetic."""
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext

from finqa_mcp.schemas import source_map

FACTORS = {"USD": Decimal(1), "thousand_USD": Decimal(1000),
           "million_USD": Decimal(10**6), "billion_USD": Decimal(10**9)}
TOKEN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"


class NumberError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(detail)


@dataclass
class Number:
    value: Decimal
    unit: str
    sources: list[str]
    original: str = ""

    @property
    def canonical(self):
        with localcontext() as ctx:
            ctx.prec = 80
            return self.value * FACTORS.get(self.unit, Decimal(1))

    def json(self):
        result = {"value": format(self.value, "f"), "unit": self.unit,
                  "sources": self.sources, "original": self.original}
        if self.unit == "ratio":
            with localcontext() as ctx:
                ctx.prec = 80
                result["percent_value"] = format(self.value * 100, "f")
        return result


def parse_number(text, *, currency=False, scale=None, sources=None):
    """Percent -> ratio; money keeps its explicit display scale. Never guess a missing cell."""
    original = str(text)
    text = original.strip().replace("−", "-").replace("\u00a0", " ")
    if len(text) > 180 or text in {"", "-", "—", "–", "n/a", "N/A"}:
        raise NumberError("missing_or_invalid_number", "Choose a cell containing a number; dashes are not zero.")
    percent = "%" in text
    currency = currency or "$" in text or bool(re.search(r"\busd\b", text, re.I))
    for word in ("billion", "million", "thousand"):
        if re.search(r"\b" + word + r"s?\b", text, re.I):
            scale = word
    text = re.sub(r"\b(?:USD|dollars?|billions?|millions?|thousands?)\b", "", text, flags=re.I)
    text = text.replace("$", "")
    # FinQA duplicates formatted values, e.g. -156 (156), 36.3% (36.3 %).
    duplicate = re.fullmatch(r"\s*(" + TOKEN + r")\s*%?\s*\(\s*(" + TOKEN + r")\s*%?\s*\)\s*", text)
    if duplicate:
        a, b = (Decimal(x.replace(",", "")) for x in duplicate.groups())
        if abs(a) != abs(b):
            raise NumberError("ambiguous_number", "The cell contains different numbers; specify a text number_index or another source.")
        text = duplicate.group(1)
    text = re.sub(r"\(\s*%\s*\)", "", text).replace("%", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1].strip()
    if not re.fullmatch(TOKEN, text):
        raise NumberError("invalid_number", "Use a decimal, a table cell ID, calc_N, or {source_id, number_index}.")
    try:
        value = Decimal(text.replace(",", ""))
    except InvalidOperation as exc:
        raise NumberError("invalid_number", "A finite decimal is required.") from exc
    if not value.is_finite() or (value and abs(value.adjusted()) > 100):
        raise NumberError("invalid_number", "A finite decimal with exponent between -100 and 100 is required.")
    if percent:
        value /= 100
        unit = "ratio"
    elif currency:
        unit = f"{scale}_USD" if scale else "USD"
    else:
        unit = "number"
    return Number(value, unit, list(sources or []), original)


def text_numbers(text):
    """Extract ordered numeric occurrences; remove only adjacent FinQA duplicate annotations."""
    text = re.sub(r"\b20ac(?=\d)", "EUR ", text)  # FinQA's encoded euro symbol
    text = re.sub(r"(" + TOKEN + r")\s*%\s*\(\s*\1\s*%\s*\)", r"\1%", text)
    text = re.sub(r"-(\d+(?:\.\d+)?)\s*\(\s*\1\s*\)", r"-\1", text)
    pattern = r"(?<![\w.])(?:\$\s*)?(?:\(\s*" + TOKEN + r"\s*\)|" + TOKEN + r")(?:\s*%)?(?:\s+(?:billion|million|thousand)s?)?"
    return [m.group(0).strip() for m in re.finditer(pattern, text, flags=re.I)]


class Calculator:
    def __init__(self, record):
        self.record = record
        self.sources = source_map(record)
        self.results = {}

    def resolve(self, operand):
        if isinstance(operand, dict):
            if set(operand) - {"source_id", "number_index"} or "source_id" not in operand:
                raise NumberError("invalid_operand", "A reference object has source_id and optional number_index only.")
            sid = operand["source_id"]
            index = operand.get("number_index", 0)
        elif isinstance(operand, str):
            if operand in self.results:
                return self.results[operand]
            if operand.startswith("calc_"):
                raise NumberError("unknown_result", "Use a calc_N returned by an earlier successful call.")
            if operand in self.sources or operand.startswith(("table_", "pre_", "post_")):
                sid, index = operand, 0
            else:
                return parse_number(operand)
        else:
            raise NumberError("invalid_operand", "Operands must be decimal strings or explicit source references.")
        if sid not in self.sources:
            raise NumberError("unknown_source", "This source ID does not exist in the current example.")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise NumberError("invalid_number_index", "number_index must be a nonnegative integer.")
        original = self.sources[sid]
        currency, scale = False, None
        if sid.startswith("table_"):
            r, c = map(int, re.fullmatch(r"table_r(\d+)_c(\d+)", sid).groups())
            row_label = self.record.table[r][0] if self.record.table[r] else ""
            column_label = self.record.table[0][c] if c < len(self.record.table[0]) else ""
            nonmoney = re.search(r"shares|employees|gallons|gwh|\bmw\b|tonnes|\bunits\b|issuance|maturity|interest rate", row_label + " " + column_label, re.I)
            currency = "$" in original or (r > 0 and c > 0 and not nonmoney and any("$" in cell for row in self.record.table for cell in row))
            caption = " ".join(self.record.table[0]) + " " + " ".join(self.record.pre_text[:3])
            for word in ("billion", "million", "thousand"):
                if re.search(r"(?:in|of|\()\s*" + word + r"s?\b", caption, re.I):
                    scale = word
                    break
            try:
                scalar = parse_number(original, currency=currency, scale=scale, sources=[sid])
            except NumberError:
                values = text_numbers(original)
                if index >= len(values):
                    raise NumberError("invalid_number_index", f"This cell has {len(values)} numeric occurrences; inspect numeric_values.")
                original = values[index]
                currency = "$" in original  # Mixed-currency/date cells need explicit occurrence semantics.
            else:
                if index:
                    raise NumberError("invalid_number_index", "This cell is scalar; use number_index 0.")
                return scalar
        else:
            values = text_numbers(original)
            if index >= len(values):
                raise NumberError("invalid_number_index", f"This text has {len(values)} numbers, indexed from 0. Read its numeric_values.")
            original = values[index]
            # An explicit 'in millions' caption applies to dollar amounts lacking a suffix.
            if "$" in original:
                caption = " ".join(self.record.table[0]) if self.record.table else ""
                for word in ("billion", "million", "thousand"):
                    if re.search(r"\bin\s+" + word + r"s?\b", caption, re.I):
                        scale = word
        return parse_number(original, currency=currency, scale=scale, sources=[sid])

    def calculate(self, operation, operands):
        try:
            if operation not in {"add", "subtract", "multiply", "divide"} or len(operands) != 2:
                raise NumberError("invalid_arguments", "Choose one operation and exactly two operands. Chain via calc_N for more values.")
            a, b = [self.resolve(value) for value in operands]
            am, bm = a.unit in FACTORS, b.unit in FACTORS
            av, bv = a.canonical, b.canonical
            if operation in {"add", "subtract"}:
                if am != bm and not ((not am and av == 0) or (not bm and bv == 0)):
                    raise NumberError("unit_mismatch", "Combine matching monetary units using source references; a plain decimal is unitless.")
                unit = a.unit if am else b.unit if bm else "ratio" if "ratio" in (a.unit, b.unit) else "number"
            elif operation == "multiply":
                if am and bm:
                    raise NumberError("unit_mismatch", "Multiplying two monetary amounts is unsupported; check the chosen operands.")
                unit = a.unit if am else b.unit if bm else "ratio" if "ratio" in (a.unit, b.unit) else "number"
            else:
                if bv == 0:
                    raise NumberError("division_by_zero", "The denominator is zero. Check the year, cell and intended denominator.")
                if bm and not am:
                    raise NumberError("unit_mismatch", "A unitless numerator divided by money is unsupported.")
                unit = "ratio" if am and bm else a.unit if am else "ratio"
            with localcontext() as ctx:
                ctx.prec = 40
                result = {"add": lambda: av + bv, "subtract": lambda: av - bv,
                          "multiply": lambda: av * bv, "divide": lambda: av / bv}[operation]()
                value = result / FACTORS.get(unit, Decimal(1))
            result_id = f"calc_{len(self.results) + 1}"
            number = Number(value, unit, sorted(set(a.sources + b.sources)))
            self.results[result_id] = number
            return {"result_id": result_id, **number.json(), "operation": operation,
                    "resolved_operands": [a.json(), b.json()]}
        except NumberError as exc:
            return {"error": exc.code, "detail": str(exc), "retry_hint": "Correct the arguments; do not repeat the identical failed call."}

    def numeric_values(self, sid):
        if sid.startswith("table_"):
            try:
                first = self.resolve({"source_id": sid, "number_index": 0})
            except NumberError:
                return []
            values = [self.sources[sid]] if first.original == self.sources[sid] else text_numbers(self.sources[sid])
        else:
            values = text_numbers(self.sources[sid])
        out = []
        for index in range(len(values)):
            try:
                out.append({"number_index": index, **self.resolve({"source_id": sid, "number_index": index}).json()})
            except NumberError:
                pass
        return out
