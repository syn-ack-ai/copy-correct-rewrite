"""Render paper.md (the subset it uses: headings, paragraphs, two-level lists, tables, images with a following "*Figure N.*" caption,
**bold**, *italic*, `code`) into one self-contained HTML page with the figures inlined as data URIs, for review.
  python to_html.py PAPER_MD OUT_HTML"""
import base64, html, re, sys
from pathlib import Path

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]); ROOT = SRC.parent


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:48]


lines = SRC.read_text().split("\n"); out, toc, i, title, sub, byline = [], [], 0, "", "", ""
while i < len(lines):
    ln = lines[i]
    if not ln.strip():
        i += 1; continue
    if ln.startswith("# "):
        title = ln[2:].strip(); i += 1; continue
    if ln.startswith("*") and not title == "" and not sub and not out:
        sub = ln.strip("*"); i += 1; continue
    if title and not byline and not sub and not out and not ln.startswith(("#", "*", "!", "|")):
        byline = ln.strip(); i += 1; continue                           # the author line between the title and the subtitle
    m = re.match(r"^(#{2,3}) (.*)", ln)
    if m:
        lvl, txt = len(m.group(1)), m.group(2); sid = slug(txt)
        if lvl == 2:
            toc.append((sid, txt))
        out.append(f'<h{lvl} id="{sid}">{inline(txt)}</h{lvl}>'); i += 1; continue
    m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", ln)
    if m:
        alt, src = m.groups(); data = base64.b64encode((ROOT / src).read_bytes()).decode()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        cap = ""
        if j < len(lines) and lines[j].startswith(("*Figure", "Figure ")):   # plain "Figure N." captions (style of arXiv 2610.15095) or the older italic form
            k = j; buf = []
            while k < len(lines) and lines[k].strip():
                buf.append(lines[k]); k += 1
            cap = inline(" ".join(buf)); i = k
        else:
            i += 1
        out.append(f'<figure><div class="plate"><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"></div>{f"<figcaption>{cap}</figcaption>" if cap else ""}</figure>'); continue
    if ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
        head, body = rows[0], [r for r in rows[2:]]
        num = [all(re.match(r"^[\d.,%<>≤≥+\-–— ()*a-z]*\d", c.replace("**", "")) for c in (r[k] for r in body) if c) for k in range(len(head))]
        t = '<div class="tablewrap"><table><thead><tr>' + "".join(f'<th class="{"num" if num[k] and k else ""}">{inline(c)}</th>' for k, c in enumerate(head)) + "</tr></thead><tbody>"
        t += "".join("<tr>" + "".join(f'<td class="{"num" if num[k] and k else ""}">{inline(c)}</td>' for k, c in enumerate(r)) + "</tr>" for r in body) + "</tbody></table></div>"
        out.append(t); continue
    if re.match(r"^(\s*)([-*]|\d+\.) ", ln):
        items = []                                                     # (indent, ordered, text)
        while i < len(lines) and (re.match(r"^(\s*)([-*]|\d+\.) ", lines[i]) or (lines[i].startswith("   ") and lines[i].strip() and items)):
            m2 = re.match(r"^(\s*)([-*]|\d+\.) (.*)", lines[i])
            if m2:
                items.append([len(m2.group(1)), m2.group(2)[0].isdigit(), m2.group(3)])
            else:
                items[-1][2] += " " + lines[i].strip()
            i += 1
        h, stack = [], []
        for ind, ordd, txt in items:
            while stack and stack[-1][0] > ind:
                h.append("</li></" + stack.pop()[1] + ">")
            if not stack or ind > stack[-1][0]:
                tag = "ol" if ordd else "ul"; stack.append((ind, tag)); h.append(f"<{tag}><li>{inline(txt)}")
            else:
                h.append(f"</li><li>{inline(txt)}")
        while stack:
            h.append("</li></" + stack.pop()[1] + ">")
        out.append("".join(h)); continue
    buf = []
    while i < len(lines) and lines[i].strip() and not re.match(r"^(#|!\[|\||\s*([-*]|\d+\.) )", lines[i]):
        buf.append(lines[i]); i += 1
    if buf:
        p = " ".join(buf); cls = ' class="note"' if p.startswith("*[") else ""
        if re.match(r"^Table [A-Z]?\d+\. ", p):   # body "Table 3." and appendix "Table D1."
            cls = ' class="tcap"'                                       # a table caption, above its table
        elif re.search(r"\s\(\d+\)$", p) and len(p) < 140:
            cls = ' class="eq"'                                         # a numbered equation
        elif re.match(r"^[A-Z][^()]{0,200}\(\d{4}\)\. ", p):
            cls = ' class="ref"'                                        # a reference entry
        out.append(f"<p{cls}>{inline(p)}</p>")
    else:
        i += 1

css = r"""
:root{--bg:#fbfbf9;--fg:#1a1a18;--fg2:#4d4c48;--muted:#86857e;--rule:#e2e1da;--accent:#2a78d6;--plate:#fcfcfb;--tbl:#f4f4f1;--note:#fff6e0;--notefg:#5a4300;
--serif:"Literata",Georgia,"Times New Roman",serif;--sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#eceae4;--fg2:#c3c1b8;--muted:#8f8d85;--rule:#2f2e2b;--accent:#5b9cea;--plate:#fcfcfb;--tbl:#1f1f1d;--note:#2c2510;--notefg:#f1d58a;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#151514;--fg:#eceae4;--fg2:#c3c1b8;--muted:#8f8d85;--rule:#2f2e2b;--accent:#5b9cea;--plate:#fcfcfb;--tbl:#1f1f1d;--note:#2c2510;--notefg:#f1d58a;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--serif);font-size:17px;line-height:1.62;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:40px 80px}
main{max-width:46rem;margin:0 auto}
.status{font-family:var(--sans);font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 16px}
h1{font-family:var(--sans);font-weight:600;font-size:clamp(28px,4.4vw,40px);line-height:1.15;letter-spacing:-.01em;margin:14px 0 10px;text-wrap:balance}
.sub{font-size:19px;color:var(--fg2);font-style:italic;margin:0 0 26px;text-wrap:balance}
h2{font-family:var(--sans);font-weight:600;font-size:23px;margin:48px 0 10px;padding-top:14px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-family:var(--sans);font-weight:600;font-size:18px;margin:32px 0 6px;text-wrap:balance}
p{margin:0 0 14px}
ul,ol{margin:0 0 14px;padding-left:1.4em}li{margin:3px 0}li>ul,li>ol{margin:4px 0 4px}
strong{font-weight:600}code{font-family:var(--mono);font-size:.86em;background:var(--tbl);padding:1px 5px;border-radius:4px}
a{color:var(--accent)}
nav.toc{font-family:var(--sans);font-size:14px;border:1px solid var(--rule);border-radius:8px;padding:12px 16px;margin:0 0 30px}
nav.toc b{display:block;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;font-weight:600}
nav.toc ol{margin:0;padding-left:0;list-style:none;display:grid;grid-template-columns:repeat(auto-fill,minmax(15rem,1fr));gap:2px 18px}
nav.toc a{color:var(--fg2);text-decoration:none}nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent);text-decoration:underline}
figure{margin:24px calc(-1 * min(9rem, max(0px, (100vw - 46rem) / 2 - 24px))) 26px}
.plate{background:var(--plate);border:1px solid var(--rule);border-radius:10px;padding:10px;overflow:hidden}
.plate img{display:block;width:100%;height:auto}
figcaption{font-family:var(--sans);font-size:14px;line-height:1.5;color:var(--fg2);margin:10px 4px 0;max-width:46rem}
.tcap{font-family:var(--sans);font-size:14px;line-height:1.5;color:var(--fg2);margin:22px 0 6px}
.byline{font-family:var(--sans);font-size:15px;color:var(--fg2);margin:-6px 0 4px}
.eq{font-family:var(--serif);font-style:italic;text-align:center;margin:14px 0}
.ref{font-size:14.5px;line-height:1.5;padding-left:1.6em;text-indent:-1.6em;margin:0 0 8px;overflow-wrap:anywhere}
.tablewrap{overflow-x:auto;margin:6px 0 18px}
table{border-collapse:collapse;font-family:var(--sans);font-size:14.5px;min-width:60%}
th{text-align:left;font-weight:600;color:var(--fg2);border-bottom:1.5px solid var(--rule);padding:7px 14px 7px 0;vertical-align:bottom}
td{border-bottom:1px solid var(--rule);padding:7px 14px 7px 0;vertical-align:top}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}
p.note{font-family:var(--sans);font-size:14.5px;background:var(--note);color:var(--notefg);border-radius:8px;padding:10px 14px}
@media (max-width:600px){body{font-size:16px}figure{margin-left:0;margin-right:0}}
"""
toc_html = '<nav class="toc" aria-label="Contents"><b>Contents</b><ol>' + "".join(f'<li><a href="#{s}">{inline(t)}</a></li>' for s, t in toc) + "</ol></nav>"
page = f"""<title>Copy, Correct, or Rewrite?</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,400&display=swap">
<style>{css}</style>
<main>
<div class="status"><span>Kaggle Gemma 4 paper track</span><span>Draft, not submitted</span><span>2026-11-10</span></div>
<h1>{inline(title)}</h1>
{f'<p class="byline">{inline(byline)}</p>' if byline else ""}
<p class="sub">{inline(sub)}</p>
{toc_html}
{chr(10).join(out)}
</main>"""
OUT.write_text(page); print("wrote", OUT, f"{len(page) / 1e6:.2f} MB")
