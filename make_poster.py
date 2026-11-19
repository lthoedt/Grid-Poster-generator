#!/usr/bin/env python3
"""Build a collage poster (default 40x30 cm) from ./pictures, guided by an optional manifest.yaml.

Renders at the highest DPI that (a) does not upscale any source photo and
(b) yields a JPEG (quality 100, 4:4:4 chroma) no larger than MAX_BYTES.

Manifest (every key optional; dates default to EXIF, so only write them to override):

  title: Poster title
  size: [40, 30]         # [width, height] in cm; sets the aspect ratio, e.g. [30, 40] for portrait
  order: date            # or "manifest"; only sorts the top level, nested items keep list order
  numbering: true        # false hides the "N." prefix in top-level captions
  dates: true            # false hides all caption dates; `date: false` on an item hides only its date
  margin: 0              # default item margin in cm (see per-item margin below)
  columns: 6             # force the poster grid
  rows: 5
  items:
    - IMG_1.jpeg                      # single picture
    - IMG_2.jpeg: Bought it           # single picture with title
    - IMG_3.jpeg: {title: ..., date: 2024-07-06, col: 1, row: 1, colspan: 2, rowspan: 2}
    - IMG_8.jpeg:
        fit: contain                  # cover (default): fill cell and crop; contain: show whole photo
        position: [50, 20]            # x/y in percent: crop focus (cover) or placement (contain)
        zoom: 1.5                     # >1 zooms in, <1 shrinks; default 1
        margin: [0.5, 1]              # inset within its cell in cm: n, [v, h] or [top, right, bottom, left]
    - title: Water pump               # group: any mapping with an "items" key
      date: 2024-07-06
      margin: 0.3                     # groups take a margin too (not inherited by their items)
      colspan: 3                      # cells covered in the parent grid (default: cols x rows)
      rowspan: 2
      cols: 3                         # the group's own sub-grid inside that area
      rows: 2
      col: 1                          # 1-based pinned position in the parent grid
      row: 1
      items:                          # same syntax as the top level, nestable
        - IMG_4.jpeg
        - IMG_5.jpeg: Old pump        # titles of nested pictures are drawn on the photo
        - rows: 2                     # untitled subgroup: two photos stacked in one cell
          rowspan: 1
          items: [IMG_6.jpeg, IMG_7.jpeg]

Unset group cols/rows are chosen as compact as possible.
Pictures in ./pictures not listed are skipped with a warning.
Without a manifest every picture is a single top-level item.
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Union

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).parent
PICTURES_DIR = ROOT / "pictures"
MANIFEST = ROOT / "manifest.yaml"
OUTPUT = ROOT / "poster.jpg"

POSTER_CM = (40, 30)
MAX_DPI = 1200
MIN_DPI = 300
MAX_BYTES = 50 * 1000 * 1000
DEFAULT_TITLE = "BMW E46 320i Touring (1999) - Maintenance log"
BG = (18, 18, 18)
FG = (235, 235, 235)
EXTS = {".jpg", ".jpeg", ".png"}
TILE_ASPECT = 4 / 3

MARGIN_CM = 1.0
GUTTER_CM = 0.25
TITLE_CM = 2.0
LABEL_CM = 0.5


@dataclass
class Picture:
    path: Path
    taken: datetime
    title: str = ""
    date: date | None = None
    pos: tuple[int, int] | None = None
    colspan: int = 1
    rowspan: int = 1
    fit: str = "cover"
    position: tuple[float, float] = (0.5, 0.5)
    zoom: float = 1.0
    show_date: bool = True
    margin: tuple[float, float, float, float] | None = None

    @property
    def day(self) -> date:
        return self.date or self.taken.date()

    def pictures(self) -> list[Picture]:
        return [self]


@dataclass
class Group:
    children: list[Node]
    placed: list[tuple[int, int]]
    cols: int
    rows: int
    title: str = ""
    date: date | None = None
    pos: tuple[int, int] | None = None
    colspan: int = 1
    rowspan: int = 1
    show_date: bool = True
    margin: tuple[float, float, float, float] | None = None

    def pictures(self) -> list[Picture]:
        return [p for child in self.children for p in child.pictures()]


Node = Union[Picture, Group]


def sort_key(node: Node) -> tuple[date, datetime]:
    pics = node.pictures()
    return node.date or min(p.day for p in pics), min(p.taken for p in pics)


def date_label(node: Node) -> str:
    if node.date:
        return f"{node.date:%d-%m-%Y}"
    days = sorted({p.day for p in node.pictures()})
    return f"{days[0]:%d-%m-%Y}" if len(days) == 1 else f"{days[0]:%d-%m-%Y} - {days[-1]:%d-%m-%Y}"


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Layout:
    size: tuple[float, float]
    tiles: list[tuple[Picture, Rect, str]] = field(default_factory=list)
    captions: list[tuple[Rect, str]] = field(default_factory=list)


@dataclass
class LayoutOptions:
    numbering: bool
    dates: bool
    margin: tuple[float, float, float, float]


def taken_at(path: Path) -> datetime:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
            if raw:
                return datetime.strptime(raw.strip("\x00"), "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def oriented_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
        # EXIF orientations 5-8 swap width and height.
        return (h, w) if im.getexif().get(0x0112, 1) in (5, 6, 7, 8) else (w, h)


def parse_date(value: object, where: str) -> date | None:
    # `date: false` hides the date in the caption; sorting still falls back to EXIF.
    if value is None or value is False:
        return None
    # YAML already turns unquoted YYYY-MM-DD into a date.
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        sys.exit(f"manifest: invalid date {value!r} in {where}; use YYYY-MM-DD")


def parse_position(spec: dict, where: str) -> tuple[int, int] | None:
    if "col" not in spec and "row" not in spec:
        return None
    if "col" not in spec or "row" not in spec:
        sys.exit(f"manifest: {where} needs both col and row to be pinned")
    return spec["col"] - 1, spec["row"] - 1


def parse_span(spec: dict, where: str, default: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    span = spec.get("colspan", default[0]), spec.get("rowspan", default[1])
    if not all(isinstance(s, int) and s >= 1 for s in span):
        sys.exit(f"manifest: {where} colspan/rowspan must be positive integers")
    return span


def parse_margin(spec: dict, where: str) -> tuple[float, float, float, float] | None:
    """CSS-style: a number, [vertical, horizontal] or [top, right, bottom, left], in cm."""
    value = spec.get("margin")
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    if len(values) not in (1, 2, 4) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in values):
        sys.exit(f"manifest: {where} margin must be a non-negative number, [v, h] or [top, right, bottom, left] (cm)")
    if len(values) == 1:
        values = values * 4
    elif len(values) == 2:
        values = [values[0], values[1], values[0], values[1]]
    return tuple(float(v) for v in values)


def parse_framing(meta: dict, where: str) -> tuple[str, tuple[float, float], float]:
    fit = meta.get("fit", "cover")
    if fit not in ("cover", "contain"):
        sys.exit(f"manifest: {where} fit must be 'cover' or 'contain'")
    position = meta.get("position", [50, 50])
    if not (isinstance(position, list) and len(position) == 2
            and all(isinstance(v, (int, float)) for v in position)):
        sys.exit(f"manifest: {where} position must be [x, y] in percent")
    zoom = meta.get("zoom", 1)
    if not (isinstance(zoom, (int, float)) and zoom > 0):
        sys.exit(f"manifest: {where} zoom must be a positive number")
    return fit, (position[0] / 100, position[1] / 100), float(zoom)


def framing(pic: Picture, src: tuple[int, int], cell: tuple[float, float]) -> tuple[float, float, float]:
    """Return the scale (cell units per source pixel) and the photo's offset within the cell."""
    (w, h), (cw, ch) = src, cell
    scale = (max if pic.fit == "cover" else min)(cw / w, ch / h) * pic.zoom
    return scale, (cw - w * scale) * pic.position[0], (ch - h * scale) * pic.position[1]


def load_items(manifest: dict, files: dict[str, Path]) -> list[Node]:
    seen: set[str] = set()

    def picture(entry: object, where: str) -> Picture:
        """Accept `name`, `name: title` or `name: {title, date, col, row, colspan, rowspan}`."""
        if isinstance(entry, str):
            name, meta = entry, {}
        elif isinstance(entry, dict) and len(entry) == 1:
            name, meta = next(iter(entry.items()))
            meta = {"title": meta} if isinstance(meta, str) else (meta or {})
        else:
            hint = "; groups list their children under 'items'" if isinstance(entry, dict) and "pictures" in entry else ""
            sys.exit(f"manifest: {where} is not a picture entry: {entry!r}{hint}")
        if name not in files:
            sys.exit(f"manifest: {where} references unknown picture {name!r}")
        if name in seen:
            sys.exit(f"manifest: picture {name!r} is listed more than once")
        seen.add(name)
        return Picture(files[name], taken_at(files[name]), meta.get("title") or "",
                       parse_date(meta.get("date"), f"{where} ({name})"), parse_position(meta, where),
                       *parse_span(meta, where), *parse_framing(meta, f"{where} ({name})"),
                       show_date=meta.get("date") is not False, margin=parse_margin(meta, f"{where} ({name})"))

    def group(spec: dict, where: str) -> Group:
        children = [node(e, f"{where}.items[{j}]") for j, e in enumerate(spec["items"] or [])]
        if not children:
            sys.exit(f"manifest: {where} has no items")
        cols, rows, placed = group_grid(children, spec.get("cols"), spec.get("rows"), where)
        return Group(children, placed, cols, rows, spec.get("title") or "",
                     parse_date(spec.get("date"), where), parse_position(spec, where),
                     *parse_span(spec, where, default=(cols, rows)), show_date=spec.get("date") is not False,
                     margin=parse_margin(spec, where))

    def node(entry: object, where: str) -> Node:
        return group(entry, where) if isinstance(entry, dict) and "items" in entry else picture(entry, where)

    nodes = [node(e, f"items[{i}]") for i, e in enumerate(manifest.get("items") or [])]

    for name in sorted(set(files) - seen):
        print(f"warning: {name} is not in the manifest; skipped", file=sys.stderr)

    if manifest.get("order", "date") == "date":
        nodes.sort(key=sort_key)
    return nodes


def group_grid(children: list[Node], cols: int | None, rows: int | None,
               where: str) -> tuple[int, int, list[tuple[int, int]]]:
    """Resolve a group's grid size from its children; unspecified dimensions are as compact as possible."""
    min_cols = max(c.colspan + (c.pos[0] if c.pos else 0) for c in children)
    area = sum(c.colspan * c.rowspan for c in children)
    if cols is None and rows is None:
        cols = max(math.ceil(math.sqrt(area)), min_cols)
    if cols is None:
        cols = next((n for n in range(min_cols, min_cols + area + 1) if pack(children, n, rows)), None)
        if cols is None:
            sys.exit(f"manifest: {where} items do not fit in {rows} rows")
    placed = pack(children, cols, rows)
    if placed is None:
        sys.exit(f"manifest: {where} items do not fit in a {cols}x{rows or '?'} grid")
    return cols, rows or used_rows(children, placed), placed


def pack(nodes: list[Node], cols: int, max_rows: int | None) -> list[tuple[int, int]] | None:
    """Place nodes on a cols-wide cell grid: pinned nodes first, then first-fit in order."""
    occupied: set[tuple[int, int]] = set()
    placed: list[tuple[int, int] | None] = [None] * len(nodes)

    def cells(c: int, r: int, n: Node) -> list[tuple[int, int]]:
        return [(c + dc, r + dr) for dc in range(n.colspan) for dr in range(n.rowspan)]

    for i, n in enumerate(nodes):
        if n.pos:
            c, r = n.pos
            block = cells(c, r, n)
            if c < 0 or r < 0 or c + n.colspan > cols or occupied.intersection(block):
                return None
            occupied.update(block)
            placed[i] = n.pos

    for i, n in enumerate(nodes):
        if placed[i]:
            continue
        if n.colspan > cols:
            return None
        r = 0
        while placed[i] is None:
            for c in range(cols - n.colspan + 1):
                block = cells(c, r, n)
                if not occupied.intersection(block):
                    occupied.update(block)
                    placed[i] = (c, r)
                    break
            r += 1

    if max_rows is not None and used_rows(nodes, placed) > max_rows:
        return None
    return placed


def used_rows(nodes: list[Node], placed: list[tuple[int, int]]) -> int:
    return max(r + n.rowspan for (_, r), n in zip(placed, nodes))


def cell_size(area_w: float, area_h: float, cols: int, rows: int) -> tuple[float, float]:
    return (area_w - GUTTER_CM * (cols - 1)) / cols, (area_h - GUTTER_CM * (rows - 1)) / rows


def choose_grid(nodes: list[Node], area_w: float, area_h: float,
                force_cols: int | None, force_rows: int | None) -> tuple[int, int, list[tuple[int, int]]]:
    total = sum(n.colspan * n.rowspan for n in nodes)
    candidates = [force_cols] if force_cols else range(1, total + 1)
    best = None
    for cols in candidates:
        placed = pack(nodes, cols, force_rows)
        if placed is None:
            continue
        rows = force_rows or used_rows(nodes, placed)
        cell_w, cell_h = cell_size(area_w, area_h, cols, rows)
        # Score by the largest 4:3 photo that fits, so the grid avoids extreme crop ratios.
        score = min(cell_w, (cell_h - LABEL_CM) * TILE_ASPECT)
        if best is None or score > best[0]:
            best = (score, cols, rows, placed)
    if best is None:
        sys.exit("manifest: items do not fit the requested poster grid")
    return best[1], best[2], best[3]


def last_row_shift(nodes: list[Node], placed: list[tuple[int, int]], cols: int) -> float:
    """Columns to shift the bottom row by to center it, if it only holds single-row, unpinned nodes."""
    bottom = used_rows(nodes, placed)
    in_last = [(c, n) for (c, r), n in zip(placed, nodes) if r + n.rowspan == bottom]
    if any(n.rowspan > 1 or n.pos for _, n in in_last):
        return 0.0
    used = sorted((c, c + n.colspan) for c, n in in_last)
    if used[0][0] != 0 or any(a[1] != b[0] for a, b in zip(used, used[1:])):
        return 0.0
    return (cols - used[-1][1]) / 2


def layout_grid(layout: Layout, nodes: list[Node], placed: list[tuple[int, int]], cols: int, rows: int,
                area: Rect, top_level: bool, opts: LayoutOptions) -> None:
    cell_w, cell_h = cell_size(area.w, area.h, cols, rows)
    shift = last_row_shift(nodes, placed, cols)
    bottom = used_rows(nodes, placed)
    for i, ((c, r), n) in enumerate(zip(placed, nodes), start=1):
        dx = shift if r + n.rowspan == bottom else 0.0
        rect = Rect(area.x + (c + dx) * (cell_w + GUTTER_CM), area.y + r * (cell_h + GUTTER_CM),
                    n.colspan * cell_w + (n.colspan - 1) * GUTTER_CM, n.rowspan * cell_h + (n.rowspan - 1) * GUTTER_CM)
        top, right, bottom_m, left = n.margin or opts.margin
        rect = Rect(rect.x + left, rect.y + top, rect.w - left - right, rect.h - top - bottom_m)
        if rect.w <= 0 or rect.h <= LABEL_CM:
            name = n.path.name if isinstance(n, Picture) else (n.title or "untitled group")
            sys.exit(f"manifest: margin leaves no room for {name}")
        # Top-level items are always captioned; nested groups only when they carry a title or date,
        # so untitled subgroups act as pure layout blocks.
        if top_level or (isinstance(n, Group) and (n.title or n.date)):
            text = "  ·  ".join(filter(None, [n.title, date_label(n) if opts.dates and n.show_date else ""]))
            if top_level and opts.numbering:
                text = f"{i}.  {text}".rstrip()
            layout.captions.append((Rect(rect.x, rect.y + rect.h - LABEL_CM, rect.w, LABEL_CM), text))
            rect = Rect(rect.x, rect.y, rect.w, rect.h - LABEL_CM)
        if isinstance(n, Picture):
            layout.tiles.append((n, rect, "" if top_level else n.title))
        else:
            layout_grid(layout, n.children, n.placed, n.cols, n.rows, rect, False, opts)


def parse_size(manifest: dict) -> tuple[float, float]:
    value = manifest.get("size", list(POSTER_CM))
    min_w, min_h = 2 * MARGIN_CM + 1, 2 * MARGIN_CM + TITLE_CM + LABEL_CM + 1
    if not (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
            and value[0] >= min_w and value[1] >= min_h):
        sys.exit(f"manifest: size must be [width, height] in cm, at least [{min_w:g}, {min_h:g}]")
    return float(value[0]), float(value[1])


def build_layout(nodes: list[Node], manifest: dict) -> tuple[Layout, int, int]:
    size = parse_size(manifest)
    area = Rect(MARGIN_CM, MARGIN_CM + TITLE_CM, size[0] - 2 * MARGIN_CM, size[1] - TITLE_CM - 2 * MARGIN_CM)
    cols, rows, placed = choose_grid(nodes, area.w, area.h, manifest.get("columns"), manifest.get("rows"))
    opts = LayoutOptions(manifest.get("numbering", True), manifest.get("dates", True),
                          parse_margin(manifest, "top-level") or (0.0, 0.0, 0.0, 0.0))
    layout = Layout(size)
    layout_grid(layout, nodes, placed, cols, rows, area, True, opts)
    return layout, cols, rows


def source_limited_dpi(layout: Layout) -> int:
    """Highest DPI at which every visible source region is still rendered at >= 1:1."""
    limit = math.inf
    for pic, rect, _ in layout.tiles:
        scale, _, _ = framing(pic, oriented_size(pic.path), (rect.w, rect.h))
        limit = min(limit, 2.54 / scale)
    return int(limit)


_fonts: dict[int, ImageFont.FreeTypeFont] = {}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _fonts:
        for name in ("/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf", "Arial.ttf"):
            try:
                _fonts[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _fonts[size] = ImageFont.load_default(size=size)
    return _fonts[size]


def fit_font(draw: ImageDraw.ImageDraw, text: str, size: int, max_w: float) -> ImageFont.FreeTypeFont:
    while size > 6 and draw.textlength(text, font=load_font(size)) > max_w:
        size -= 1
    return load_font(size)


def render(layout: Layout, title: str, dpi: int) -> Image.Image:
    def px(v: float) -> int:
        return round(v / 2.54 * dpi)

    def box(r: Rect) -> tuple[int, int, int, int]:
        return px(r.x), px(r.y), px(r.x + r.w), px(r.y + r.h)

    poster = Image.new("RGB", (px(layout.size[0]), px(layout.size[1])), BG)
    draw = ImageDraw.Draw(poster)
    title_font = fit_font(draw, title, px(TITLE_CM * 0.5), poster.width - 2 * px(MARGIN_CM))
    draw.text((poster.width / 2, px(MARGIN_CM + TITLE_CM / 2)), title, font=title_font, fill=FG, anchor="mm")
    label_size = px(LABEL_CM * 0.6)

    for pic, rect, overlay in layout.tiles:
        x0, y0, x1, y1 = box(rect)
        cw, ch = x1 - x0, y1 - y0
        tile = Image.new("RGB", (cw, ch), BG)
        with Image.open(pic.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            scale, ox, oy = framing(pic, im.size, (cw, ch))
            vx0, vy0 = round(max(0.0, ox)), round(max(0.0, oy))
            vx1, vy1 = round(min(cw, ox + im.width * scale)), round(min(ch, oy + im.height * scale))
            if vx1 > vx0 and vy1 > vy0:
                src_box = (max(0.0, (vx0 - ox) / scale), max(0.0, (vy0 - oy) / scale),
                           min(im.width, (vx1 - ox) / scale), min(im.height, (vy1 - oy) / scale))
                tile.paste(im.resize((vx1 - vx0, vy1 - vy0), Image.LANCZOS, box=src_box), (vx0, vy0))
        if overlay:
            band_h = px(LABEL_CM)
            band = Image.new("RGBA", tile.size, (0, 0, 0, 0))
            bdraw = ImageDraw.Draw(band)
            bdraw.rectangle((0, tile.height - band_h, tile.width, tile.height), fill=(0, 0, 0, 150))
            font = fit_font(bdraw, overlay, label_size, tile.width * 0.95)
            bdraw.text((tile.width / 2, tile.height - band_h / 2), overlay, font=font, fill=FG, anchor="mm")
            tile = Image.alpha_composite(tile.convert("RGBA"), band).convert("RGB")
        poster.paste(tile, (x0, y0))

    for rect, text in layout.captions:
        x0, y0, x1, y1 = box(rect)
        font = fit_font(draw, text, label_size, (x1 - x0) * 0.98)
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=font, fill=FG, anchor="mm")
    return poster


def encode(poster: Image.Image, dpi: int) -> bytes:
    buf = io.BytesIO()
    poster.save(buf, "JPEG", quality=100, subsampling=0, optimize=True, dpi=(dpi, dpi))
    return buf.getvalue()


def init_manifest(files: dict[str, Path]) -> None:
    if MANIFEST.exists():
        sys.exit(f"{MANIFEST} already exists")

    def scalar(value: str) -> str:
        return yaml.safe_dump(value, default_style=None).removesuffix("\n...\n").strip()

    items = sorted(files.items(), key=lambda kv: taken_at(kv[1]))
    lines = [f"title: {scalar(DEFAULT_TITLE)}", "items:"]
    # EXIF dates go in comments so they guide grouping without being duplicated as data.
    lines += [f"  - {scalar(name)}  # {taken_at(p):%Y-%m-%d}" for name, p in items]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {MANIFEST} with {len(items)} pictures")


def build() -> None:
    files = {p.name: p for p in sorted(PICTURES_DIR.iterdir()) if p.suffix.lower() in EXTS}
    manifest = (yaml.safe_load(MANIFEST.read_text()) or {}) if MANIFEST.exists() else {"items": list(files)}
    nodes = load_items(manifest, files)
    if not nodes:
        sys.exit("nothing to render")
    layout, cols, rows = build_layout(nodes, manifest)
    title = manifest.get("title") or DEFAULT_TITLE

    dpi = max(MIN_DPI, min(MAX_DPI, source_limited_dpi(layout)))
    print(f"{len(nodes)} top-level items, {len(layout.tiles)} pictures, grid {cols}x{rows}, starting at {dpi} DPI")

    while True:
        poster = render(layout, title, dpi)
        data = encode(poster, dpi)
        print(f"  {dpi} DPI -> {poster.width}x{poster.height}px, {len(data) / 1e6:.1f} MB")
        if len(data) <= MAX_BYTES:
            break
        if dpi <= MIN_DPI:
            sys.exit(f"Poster exceeds {MAX_BYTES / 1e6:.0f} MB even at {MIN_DPI} DPI.")
        # File size scales roughly with pixel count, i.e. with DPI squared; 0.98 adds headroom.
        dpi = max(MIN_DPI, int(dpi * math.sqrt(MAX_BYTES / len(data)) * 0.98))

    tmp = OUTPUT.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(OUTPUT)
    print(f"Saved {OUTPUT} ({poster.width}x{poster.height}px, {dpi} DPI, {len(data) / 1e6:.1f} MB)")


def watched_state() -> tuple[float, float]:
    manifest_mtime = MANIFEST.stat().st_mtime if MANIFEST.exists() else 0.0
    return manifest_mtime, PICTURES_DIR.stat().st_mtime


def watch(interval: float) -> None:
    """Re-render whenever the manifest or the pictures folder changes; errors are reported, not fatal."""
    print(f"Watching {MANIFEST.name} and {PICTURES_DIR.name}/ (Ctrl+C to stop)")
    state = None
    try:
        while True:
            current = watched_state()
            if current != state:
                # Editors may write in several steps; wait until the file has settled.
                time.sleep(0.3)
                if watched_state() != current:
                    continue
                state = current
                print(f"\n[{datetime.now():%H:%M:%S}] rendering")
                try:
                    build()
                except SystemExit as e:
                    print(f"error: {e.code}", file=sys.stderr)
                except yaml.YAMLError as e:
                    print(f"error: invalid YAML in {MANIFEST.name}: {e}", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", action="store_true", help="write a starter manifest.yaml and exit")
    parser.add_argument("--watch", action="store_true", help="keep running and re-render on manifest/picture changes")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between change checks in --watch mode")
    args = parser.parse_args()

    if args.init:
        init_manifest({p.name: p for p in sorted(PICTURES_DIR.iterdir()) if p.suffix.lower() in EXTS})
    elif args.watch:
        watch(args.interval)
    else:
        build()


if __name__ == "__main__":
    main()
