#!/usr/bin/env python3
"""Minimal LaTeX-subset -> OMML (native Word math) converter.

Supports exactly the constructs used by the manuscript equations:
  - groups {..}, subscripts _x, superscripts ^x
  - \\frac{num}{den}
  - \\sum_{lo}^{hi}, \\max / \\min / \\median / \\log
  - \\left( \\right), \\left[ \\right], \\left\\{ \\right\\}
  - \\mathrm{..}, \\mathbf{..}, \\mathbb{..}, \\mathcal{..}
  - Greek / symbol commands (\\tau \\mu \\sigma \\lambda \\delta \\kappa
    \\ell \\Lambda \\infty \\le \\ge \\neq \\in \\to \\times \\cdot \\wedge)
  - \\quad \\qquad \\; \\, \\% \\{ \\} \\( \\) \\[ \\]
"""

from html import escape as _hesc


SYMBOLS = {
    "tau": "\u03c4", "mu": "\u03bc", "sigma": "\u03c3", "lambda": "\u03bb",
    "pi": "\u03c0",
    "delta": "\u03b4", "kappa": "\u03ba", "ell": "\u2113", "Lambda": "\u039b",
    "alpha": "\u03b1", "beta": "\u03b2", "gamma": "\u03b3", "varepsilon": "\u03b5",
    "varphi": "\u03c6", "phi": "\u03c6", "Phi": "\u03a6", "psi": "\u03c8",
    "epsilon": "\u03b5",
    "eta": "\u03b7",
    "infty": "\u221e", "le": "\u2264", "leq": "\u2264", "ge": "\u2265",
    "geq": "\u2265", "neq": "\u2260", "in": "\u2208", "to": "\u2192",
    "times": "\u00d7", "cdot": "\u22c5", "wedge": "\u2227", "mid": "\u2223",
    "pm": "\u00b1", "approx": "\u2248",
    "exists": "\u2203", "forall": "\u2200",
    "sim": "\u223c", "lfloor": "\u230a", "rfloor": "\u230b",
    "ddagger": "\u2021", "dagger": "\u2020",
    "colon": ":", "colonapprox": ":",
    "dots": "\u2026", "cdots": "\u22ef",
    "%": "%", "(": "(", ")": ")", "[": "[", "]": "]",
    "{": "{", "}": "}", " ": "\u00a0",
    "|": "\u2225",
}

SPACES = {"quad": "\u2003", "qquad": "\u2003\u2003", ";": "\u2009", ",": "\u2009"}

FUNCS = {"max", "min", "median", "log", "exp", "sin", "cos", "tan", "ln"}
NARY = {"sum": "\u2211", "prod": "\u220f"}
ACCENTS = {"hat": "^", "bar": "\u00af", "tilde": "~"}

BB = {"E": "\U0001d53c", "R": "\u211d", "C": "\u2102", "N": "\u2115"}
CAL = {"C": "\U0001d49e", "N": "\U0001d4a9", "R": "\u211b", "L": "\u2112"}


def _run(text, plain=False):
    rpr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f"<m:r>{rpr}<m:t xml:space=\"preserve\">{_hesc(text)}</m:t></m:r>"


def _s_sub(base, sub):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def _s_sup(base, sup):
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"


def _frac(num, den):
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def _nary(ch, sub, sup, body):
    pr = f'<m:naryPr><m:chr m:val="{ch}"/><m:limLoc m:val="undOvr"/></m:naryPr>'
    s = f"<m:sub>{sub}</m:sub>" if sub is not None else ""
    p = f"<m:sup>{sup}</m:sup>" if sup is not None else ""
    return f"<m:nary>{pr}{s}{p}<m:e>{body}</m:e></m:nary>"


def _delim(open_ch, close_ch, content):
    dpr = (
        f'<m:dPr><m:begChr m:val="{_hesc(open_ch)}"/>'
        f'<m:endChr m:val="{_hesc(close_ch)}"/></m:dPr>'
    )
    return f"<m:d>{dpr}<m:e>{content}</m:e></m:d>"


def _func(name, arg):
    return f"<m:func><m:fName>{_run(name, plain=True)}</m:fName><m:e>{arg}</m:e></m:func>"


def tokenize(src):
    """Return list of tokens: ('cmd', name) | ('lit', text) | '{' | '}' | '_' | '^'."""
    toks = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\\":
            j = i + 1
            if j < n and src[j].isalpha():
                k = j
                while k < n and src[k].isalpha():
                    k += 1
                toks.append(("cmd", src[j:k]))
                i = k
            else:
                toks.append(("cmd", src[j] if j < n else " "))
                i = j + 1
        elif c == "{":
            toks.append("{")
            i += 1
        elif c == "}":
            toks.append("}")
            i += 1
        elif c == "_":
            toks.append("_")
            i += 1
        elif c == "^":
            toks.append("^")
            i += 1
        else:
            toks.append(("lit", src[i]))
            i += 1
    return toks


class Parser:
    def __init__(self, src):
        self.t = tokenize(src)
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def parse(self):
        out, self.i = self._seq(None)
        if self.i != len(self.t):
            raise ValueError(f"trailing tokens: {self.t[self.i:]}")
        return out

    def _group(self):
        assert self.peek() == "{"
        self.i += 1
        content, self.i = self._seq("}")
        assert self.peek() == "}"
        self.i += 1
        return content

    def _group_text(self):
        """Read a {..} group as plain text (for \\mathrm/\\mathbb/\\mathcal)."""
        assert self.peek() == "{"
        self.i += 1
        out = []
        while self.peek() is not None and self.peek() != "}":
            tok = self.peek()
            self.i += 1
            if tok[0] == "lit":
                out.append(tok[1])
            elif tok[0] == "cmd":
                out.append(SYMBOLS.get(tok[1], tok[1]))
            elif tok == "{":
                out.append(self._group_text())
            # "_" / "^" are not expected inside a plain-text group
        assert self.peek() == "}"
        self.i += 1
        return "".join(out)

    def _script_arg(self):
        if self.peek() == "{":
            return self._group()
        return self._atom()

    def _seq(self, stop):
        parts = []
        while self.i < len(self.t):
            tok = self.peek()
            if tok == stop:
                break
            node = self._atom()
            parts.append(node)
        return "".join(parts), self.i

    def _atom(self):
        tok = self.peek()
        if tok == "{":
            self.i += 1
            base, self.i = self._seq("}")
            assert self.peek() == "}"
            self.i += 1
        elif tok == ("cmd", "frac"):
            self.i += 1
            num = self._group()
            den = self._group()
            base = _frac(num, den)
        elif tok[0] == "cmd" and tok[1] == "sqrt":
            self.i += 1
            inner = self._script_arg()
            base = f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{inner}</m:e></m:rad>'
        elif tok[0] == "cmd" and tok[1] in NARY:
            self.i += 1
            sub = sup = None
            if self.peek() == "_":
                self.i += 1
                sub = self._script_arg()
            if self.peek() == "^":
                self.i += 1
                sup = self._script_arg()
            body = self._atom()
            base = _nary(NARY[tok[1]], sub, sup, body)
        elif tok == ("cmd", "left"):
            base = self._delimiter()
        elif tok[0] == "cmd" and tok[1] in FUNCS:
            base = self._operator()
        elif tok[0] == "cmd" and tok[1] in ("mathrm", "mathbf", "operatorname", "text", "rm"):
            self.i += 1
            base = _run(self._group_text(), plain=True)
        elif tok[0] == "cmd" and tok[1] == "mathbb":
            self.i += 1
            base = _run("".join(BB.get(ch, ch) for ch in self._group_text()))
        elif tok[0] == "cmd" and tok[1] == "mathcal":
            self.i += 1
            base = _run("".join(CAL.get(ch, ch) for ch in self._group_text()))
        elif tok[0] == "cmd" and tok[1] in ACCENTS:
            self.i += 1
            inner = self._atom()
            base = f'<m:acc><m:accPr><m:chr m:val="{ACCENTS[tok[1]]}"/></m:accPr>{inner}</m:acc>'
        elif tok[0] == "cmd" and tok[1] in SYMBOLS:
            self.i += 1
            base = _run(SYMBOLS[tok[1]])
        elif tok[0] == "cmd" and tok[1] in SPACES:
            self.i += 1
            base = _run(SPACES[tok[1]])
        elif tok == ("cmd", "right"):
            raise ValueError("stray \\right")
        elif tok[0] == "lit":
            self.i += 1
            base = _run(tok[1])
        else:
            raise ValueError(f"unexpected token: {tok!r}")

        # optional scripts
        while True:
            nxt = self.peek()
            if nxt == "_":
                self.i += 1
                base = _s_sub(base, self._script_arg())
            elif nxt == "^":
                self.i += 1
                base = _s_sup(base, self._script_arg())
            else:
                break
        return base

    def _operator(self):
        tok = self.peek()
        name = tok[1]
        self.i += 1
        nxt = self.peek()
        if nxt in ("_", "^"):
            # limit-style operator: max_{..} R_t
            base = _run(name, plain=True)
        elif nxt == ("cmd", "left"):
            arg = self._delimiter()
            return _func(name, arg)
        else:
            return _run(name, plain=True)
        return base

    def _delimiter(self):
        # consume \left
        assert self.peek() == ("cmd", "left")
        self.i += 1
        open_ch = self._delim_char()
        content, self.i = self._seq(("cmd", "right"))
        assert self.peek() == ("cmd", "right")
        self.i += 1
        close_ch = self._delim_char()
        return _delim(open_ch, close_ch, content)

    def _delim_char(self):
        tok = self.peek()
        self.i += 1
        if tok[0] == "lit":
            return tok[1].strip() or tok[1]
        if tok[0] == "cmd":
            return SYMBOLS.get(tok[1], tok[1])
        return tok


def to_omml(latex):
    """Convert a LaTeX-subset expression to an <m:oMath>...</m:oMath> string."""
    body = Parser(latex).parse()
    return f"<m:oMath>{body}</m:oMath>"


if __name__ == "__main__":
    import sys
    src = sys.argv[1]
    print(to_omml(src))
