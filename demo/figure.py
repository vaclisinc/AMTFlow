"""Draw the AMTFlow framework figure: demo/framework.svg.

The pipeline from left to right, one card per stage listing the modules it can use.
Run `python demo/figure.py`; see demo/README.md for the PNG used in the README.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

INK, PAPER, CARD, SOFT = "#1d1a17", "#f4efea", "#ffffff", "#6b645c"
FELT, SKY_WASH, SKY, SUN, TANG = "#2b4f8f", "#e1f1ff", "#8fd0ff", "#ffd84d", "#ff9f4a"
WIDTH, HEIGHT = 1600, 650

out: list[str] = []
add = out.append


def text(x, y, s, size=14, weight=500, fill=INK, anchor="start", mono=False, spacing=0):
    fam = "var(--mono)" if mono else "var(--sans)"
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{spacing}">{escape(s)}</text>')


def label(x, y, s, fill):
    w = 8.4 * len(s) + 16
    add(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="22" fill="{fill}" stroke="{INK}" stroke-width="1.5"/>')
    text(x + 8, y + 15.5, s.upper(), 11.5, 600, mono=True, spacing=0.8)


def box(x, y, w, h):
    add(f'<rect x="{x + 6}" y="{y + 6}" width="{w}" height="{h}" fill="{INK}"/>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{CARD}" stroke="{INK}" stroke-width="2"/>')


def chip(x, y, w, s, kind=""):
    fill, color, weight, dash = CARD, INK, 500, ""
    if kind == "on":
        fill, color, weight = FELT, "#ffffff", 600
    elif kind == "plug":
        fill, color, dash = "none", SOFT, ' stroke-dasharray="5 4"'
    add(f'<rect x="{x}" y="{y}" width="{w}" height="31" fill="{fill}" stroke="{INK}" stroke-width="1.5"{dash}/>')
    text(x + 11, y + 20.5, s, 14, weight, color)
    if kind == "on":
        text(x + w - 9, y + 19.5, "DEFAULT", 9.5, 600, "#cfe0ff", "end", True, 1)


def option(x, y, w, name, desc, kind="", example=None):
    """A selectable option: radio dot, name, a description and an optional example line."""
    fill, dash = ("#e9eef7", "") if kind == "on" else ("none", ' stroke-dasharray="5 4"') if kind == "plug" else (CARD, "")
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{62 if example else 46}" fill="{fill}" stroke="{INK}" stroke-width="1.5"{dash}/>')
    add(f'<circle cx="{x + 15}" cy="{y + 16}" r="6" fill="{CARD}" stroke="{INK}" stroke-width="1.5"/>')
    if kind == "on":
        add(f'<circle cx="{x + 15}" cy="{y + 16}" r="3.2" fill="{FELT}"/>')
    text(x + 29, y + 21, name, 14, 600, INK if kind != "plug" else SOFT)
    text(x + 29, y + 38, desc, 12, 500, SOFT)
    if example:
        text(x + 29, y + 54, "e.g. " + example, 12, 600, FELT if kind == "on" else INK)


def note(x, y, s):
    text(x, y, s.upper(), 11, 600, SOFT, mono=True, spacing=0.7)


def arrow(x0, x1, y, color=INK, width=3):
    add(f'<line x1="{x0}" y1="{y}" x2="{x1 - 10}" y2="{y}" stroke="{color}" stroke-width="{width}"/>'
        f'<polygon points="{x1},{y} {x1 - 12},{y - 7} {x1 - 12},{y + 7}" fill="{color}"/>')


# ------------------------------------------------------------------ layout
TOP, H = 196, 360                 # stage cards
FLOW_Y = TOP + 58                 # the main flow line, through the card titles
GAP = 32
cols = {"in": 108, "sep": 248, "rec": 300, "route": 318, "amt": 248, "out": 150}
xs, x = {}, 44
for k, w in cols.items():
    xs[k] = x
    x += w + GAP

add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAPER}"/>')

# header
ly = 64
add(f'<circle cx="56" cy="{ly}" r="6" fill="{CARD}" stroke="{INK}" stroke-width="1.5"/><circle cx="56" cy="{ly}" r="3.2" fill="{FELT}"/>')
text(68, ly + 4.5, "default", 13, 600)
add(f'<circle cx="138" cy="{ly}" r="6" fill="{CARD}" stroke="{INK}" stroke-width="1.5"/>')
text(150, ly + 4.5, "also available, chosen in the YAML config", 13, 500, SOFT)
box(1176, 40, 370, 96)
note(1192, 64, "MulTTiPop test · family F1")
for k, (name, val, w, fill, weight) in enumerate((("MuScriptor, no tags", "29.4", 140, "#e2dcd4", 500),
                                                   ("AMTFlow", "32.5", 155, FELT, 700))):
    y = 88 + k * 30
    text(1192, y + 5, name, 14, weight)
    add(f'<rect x="1328" y="{y - 8}" width="{w}" height="15" fill="{fill}" stroke="{INK}" stroke-width="1.5"/>')
    text(1534, y + 5, val, 14.5, 700, anchor="end", mono=True)

# instrument conditioning: a wash behind steps 2 and 3
cx0, cx1 = xs["rec"] - 16, xs["route"] + cols["route"] + 16
add(f'<rect x="{cx0}" y="{TOP - 50}" width="{cx1 - cx0}" height="{H + 74}" fill="{SKY_WASH}" stroke="{INK}" stroke-width="1.5"/>')
label(cx0 + 12, TOP - 38, "Instrument conditioning · optional", SKY)

# bypass under the cards: skip conditioning, run the AMT model unconstrained
bx0, bx1, by = xs["sep"] + cols["sep"] / 2, xs["amt"] + cols["amt"] / 2, TOP + H + 56
add(f'<path d="M{bx0},{TOP + H + 6} V{by} H{bx1} V{TOP + H + 18}" fill="none" stroke="#8b847c" stroke-width="2.2" stroke-dasharray="7 6"/>'
    f'<polygon points="{bx1},{TOP + H + 8} {bx1 - 7},{TOP + H + 20} {bx1 + 7},{TOP + H + 20}" fill="#8b847c"/>')
add(f'<rect x="{(bx0 + bx1) / 2 - 150}" y="{by - 12}" width="300" height="24" fill="{PAPER}"/>')
text((bx0 + bx1) / 2, by + 4.5, "NO CONDITIONING → UNCONSTRAINED", 11.5, 600, "#8b847c", "middle", True, 0.8)

# main flow arrows between the stages
keys = list(cols)
for a, b in zip(keys, keys[1:]):
    arrow(xs[a] + cols[a] + 6, xs[b] - 2, FLOW_Y)

# a note travelling along the flow, behind the cards (animated on the demo page only)
path = f"M{xs['in'] + cols['in'] + 6},{FLOW_Y} H{xs['out'] - 2}"
add(f'<circle class="pulse" r="6" fill="{FELT}" stroke="{INK}" stroke-width="1.5" visibility="hidden">'
    f'<animateMotion dur="5s" repeatCount="indefinite" path="{path}"/></circle>')

# input
x = xs["in"]
box(x, TOP + 6, cols["in"], 104)
for k, h in enumerate((10, 22, 14, 32, 18, 36, 22, 26, 12, 20)):
    lx = x + 12 + k * 9.5
    add(f'<line x1="{lx}" y1="{FLOW_Y - h / 2}" x2="{lx}" y2="{FLOW_Y + h / 2}" stroke="{INK}" stroke-width="3" stroke-linecap="round"/>')
text(x + cols["in"] / 2, TOP + 136, "AUDIO", 12, 600, anchor="middle", mono=True, spacing=1)
text(x + cols["in"] / 2, TOP + 156, "a full song", 13.5, 500, SOFT, "middle")


def stage(key, step, title, question, tag_fill):
    x, w = xs[key], cols[key]
    box(x, TOP, w, H)
    label(x + 14, TOP + 14, step, tag_fill)
    text(x + 14, TOP + 66, title, 19, 700, spacing=-0.2)
    text(x + 14, TOP + 88, question, 13, 500, SOFT)
    return x + 14, w - 28


# 1 separation
x, w = stage("sep", "Step 1 · optional", "Source separation", "Split the mix into stems first?", PAPER)
option(x, TOP + 104, w, "None", "transcribe the full mix", "on")
option(x, TOP + 156, w, "BS-RoFormer", "6 stems: bass, drums, guitar, …")
option(x, TOP + 208, w, "Precomputed stems", "reuse an earlier separation")

# 2 recognition
x, w = stage("rec", "Step 2", "Instrument recognition", "Which instruments are playing?", SUN)
option(x, TOP + 104, w, "MATPAC++ ∪ Music Flamingo", "an audio encoder + an audio LLM", "on")
option(x, TOP + 156, w, "Audio encoders", "MATPAC++, MERT, MuQ, … (8 in total)")
option(x, TOP + 208, w, "Audio-language models", "Music Flamingo, Qwen3-Omni")
option(x, TOP + 260, w, "Ensembles", "union · majority vote · at least k")

# 3 routing: detected families -> the instrument types the AMT model may write
x, w = stage("route", "Step 3", "Family routing", "Which instrument families to transcribe?", SUN)
note(x, TOP + 116, "7 families + voice")
fams = ("Piano", "Guitar", "Bass", "Drums", "Strings", "Winds", "Synth", "Voice")
fw = (w - 15) / 4
for k, fam in enumerate(fams):
    fx, fy = x + (k % 4) * (fw + 5), TOP + 124 + (k // 4) * 30
    add(f'<rect x="{fx:.1f}" y="{fy}" width="{fw:.1f}" height="24" fill="{PAPER}" stroke="{INK}" stroke-width="1.3"/>')
    text(fx + fw / 2, fy + 16.5, fam, 12.5, 600, anchor="middle")
option(x, TOP + 194, w, "Top-N per family", "types the audio supports, up to 3", "on",
       "guitar → acoustic + distorted")
option(x, TOP + 260, w, "One per family", "always the same type (MIDI-REPA)", "",
       "guitar → clean electric")
note(x, TOP + 342, "+ finer families (brass, reeds, …)")

# 4 AMT model
x, w = stage("amt", "Step 4", "Pretrained AMT model", "Transcribes your music using only", TANG)
text(x, TOP + 106, "the instruments it is given.", 13, 500, SOFT)
option(x, TOP + 122, w, "MuScriptor", "small · medium · large", "on")
option(x, TOP + 174, w, "Your AMT model", "takes an instrument set", "plug")

# output
x = xs["out"]
box(x, TOP + 6, cols["out"], 104)
lanes = ((FELT, "#e8efff", ((8, 22), (36, 14), (56, 30), (92, 24))),
         ("#2a9d84", "#e3f6f1", ((8, 34), (48, 20), (76, 44))),
         ("#e07a2a", "#fdf0e2", ((10, 4), (30, 4), (50, 4), (70, 4), (90, 4), (110, 4))))
for k, (col, bg, notes) in enumerate(lanes):
    ly = TOP + 22 + k * 26
    add(f'<rect x="{x + 10}" y="{ly}" width="{cols["out"] - 20}" height="26" fill="{bg}" stroke="{INK}" stroke-width="1.2"/>')
    for nx, nw in notes:
        add(f'<rect x="{x + 10 + nx}" y="{ly + 10}" width="{nw}" height="6" fill="{col}"/>')
text(x + cols["out"] / 2, TOP + 136, "MIDI", 12, 600, anchor="middle", mono=True, spacing=1)
text(x + cols["out"] / 2, TOP + 156, "notes with instruments", 13, 500, SOFT, "middle")
text(x + cols["out"] / 2, TOP + 174, "+ JSON · piano-roll cube", 13, 500, SOFT, "middle")

STYLE = """
  @import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&amp;family=JetBrains+Mono:wght@500;600&amp;display=swap');
  svg { --sans: 'Hanken Grotesk', system-ui, sans-serif; --mono: 'JetBrains Mono', ui-monospace, monospace; }
  @media (prefers-reduced-motion: no-preference) { .animated .pulse { visibility: visible; } }
"""

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
       'role="img" aria-label="AMTFlow framework, left to right: audio, optional source separation, instrument '
       'recognition and family routing (together the optional instrument conditioning), a pretrained AMT model, '
       'MIDI with an instrument on every note.">'
       f'<style>{STYLE}</style>' + "\n".join(out) + "</svg>\n")
Path(__file__).with_name("framework.svg").write_text(svg)
print("wrote", Path(__file__).with_name("framework.svg"))
