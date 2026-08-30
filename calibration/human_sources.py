"""Collectors for the human side of the FR calibration corpus (PRD §10.1).

All sources are freely accessible (no consent required):

- ``wikipedia-fr``  : encyclopedic extracts (fr.wikipedia action API),
- ``blog-maudet``   : personal editorial, blog.maudet.cloud (Ghost rss.xml) —
  replaces the PRD LinkedIn strate (consent) with the blog owner agreement,
- ``linuxfr``       : technical blog posts (Atom feed + post pages),
- ``presse-fr``     : IT press (Le Monde Informatique RSS/listing, LeMagIT),
- ``litterature``   : public-domain literature (fr.wikisource action API).

Every fetcher returns records ``{"title", "url", "text", "source"}`` with the
text HTML-stripped, whitespace-normalised and trimmed near ``TARGET_CHARS``
(cut at a sentence boundary, minimum ``MIN_CHARS``); records that are too
short after trimming are dropped.
"""

from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

UA = {"User-Agent": "binoculars-eu-corpus/0.1 (research corpus; contact mmaudet@linagora.com)"}
MIN_CHARS = 400
TARGET_CHARS = 1200
PATIENCE_S = 0.4  # polite rate limiting between page fetches


def get(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


class _TextExtractor(HTMLParser):
    """Keeps the text of block tags, skipping chrome (script/style/nav...)."""

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self._in_block = False
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "nav", "footer", "header", "aside"):
            self._skip_depth += 1
        if tag in self.BLOCK_TAGS and self._skip_depth == 0:
            self._in_block = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header", "aside") and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.BLOCK_TAGS:
            self._in_block = False
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_block and self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def html_to_text(page: str) -> str:
    parser = _TextExtractor()
    parser.feed(page)
    return html.unescape(parser.text())


def trim(text: str) -> str | None:
    """Trim near TARGET_CHARS at a sentence boundary; None if too short."""
    text = text.strip()
    if len(text) <= TARGET_CHARS:
        return text if len(text) >= MIN_CHARS else None
    cut = text[:TARGET_CHARS]
    boundaries = [m.end() for m in re.finditer(r"[.!?…»]\s", cut)]
    boundaries = [b for b in boundaries if b >= MIN_CHARS]
    short = cut[: boundaries[-1]].strip() if boundaries else cut.strip()
    return short if len(short) >= MIN_CHARS else None


def _record(title: str, url: str, raw_text: str, source: str) -> dict | None:
    text = trim(raw_text)
    if text is None:
        return None
    return {"title": title.strip(), "url": url.strip(), "text": text, "source": source}


# --------------------------------------------------------------------------
# MediaWiki action API (Wikipedia / Wikisource)
# --------------------------------------------------------------------------
def _wiki_extract(api: str, title: str, chars: int = 1500) -> str:
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "exchars": chars, "redirects": 1, "format": "json", "titles": title,
    })
    data = json_loads(get(f"{api}?{params}"))
    for page in data["query"]["pages"].values():
        return page.get("extract", "").strip()
    return ""


def json_loads(payload: str) -> dict:
    import json

    return json.loads(payload)


WIKIPEDIA_TITLES = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Toulouse", "Nantes", "Strasbourg",
    "Lille", "Rennes", "Montpellier", "Dijon", "Grenoble", "Annecy", "Avignon",
    "Arles", "Rouen", "Caen", "Carcassonne", "Le Mont-Saint-Michel", "Versailles",
    "Pont du Gard", "Viaduc de Millau", "Canal du Midi", "Tour de France",
    "Championnat de France de rugby", "Pétanque", "Jeux olympiques", "Tour Eiffel",
    "Musée du Louvre", "Musée d'Orsay", "Centre Pompidou", "Cathédrale Notre-Dame de Paris",
    "Grotte de Lascaux", "Alignements de Carnac", "Révolution française",
    "Premier Empire", "Résistance intérieure française", "Débarquement de Normandie",
    "Construction européenne", "Francophonie", "Académie française",
    "Histoire de l'imprimerie", "Histoire de la photographie", "Jazz", "Cinéma français",
    "Bande dessinée franco-belge", "Cuisine française", "Fromage", "Vin de Bordeaux",
    "Champagne (AOC)", "Lavande", "Mer Méditerranée", "Massif alpin", "Massif central",
    "Forêt", "Vulcain", "Météorologie", "Astronomie", "Mathématiques", "Logique",
    "Électricité", "Télécommunications", "Internet", "Informatique", "Intelligence artificielle",
    "Robotique", "Cryptographie", "Espace (cosmologie)", "Système solaire", "Volcan",
    "Séisme", "Biodiversité", "Parc national", "Énergie solaire", "Énergie éolienne",
    "Voiture électrique", "TGV", "Métro de Paris", "Aviation civile", "Port de Marseille",
    "Économie circulaire", "Droit du travail (France)", "Laïcité", "Code civil (France)",
    "Sécurité sociale en France", "Éducation en France", "Université française",
]

WIKISOURCE_TITLES = [
    "Les Fleurs du mal", "Les Misérables", "Le Rouge et le Noir", "Madame Bovary",
    "Les Trois Mousquetaires", "Le Comte de Monte-Cristo", "Candide", "L'Île mystérieuse",
    "Vingt mille lieues sous les mers", "Germinal", "Le Père Goriot", "La Peau de chagrin",
    "Les Contemplations", "Hernani", "Cyrano de Bergerac", "Phèdre (Racine)", "Le Cid",
    "Tartuffe", "Dom Juan", "Les Femmes savantes", "Le Bourgeois gentilhomme", "L'Avare",
    "Le Malade imaginaire", "Les Chansons de Bilitis", "Demain il fera jour",
    "Le Rouge et le Noir/Partie 1", "Notre-Dame de Paris", "Les Travailleurs de la mer",
    "Salammbô", "L'Éducation sentimentale", "Thérèse Raquin", "Au Bonheur des Dames",
    "Le Tour du monde en quatre-vingts jours", "Cinq semaines en ballon",
    "De la Terre à la Lune", "Les Aventures du capitaine Hatteras", "Michel Strogoff",
    "Le Château des Carpathes", "Poésies (Baudelaire)", "Spleen de Paris",
    "Les Diaboliques (Barbey d'Aurevilly)", "Aphorismes (La Rochefoucauld)",
    "Quatrevingt-treize", "L'Homme qui rit", "Les Châtiments",
    "La Légende des siècles", "L'Art d'être grand-père",
    "Une saison en enfer", "Illuminations", "Alcools", "Calligrammes",
    "Sagesse (Verlaine)", "Romances sans paroles", "Poèmes saturniens",
    "L'Assommoir", "Nana", "La Bête humaine", "Le Ventre de Paris",
    "Bouvard et Pécuchet", "Trois Contes", "Bel-Ami", "Une vie", "Pierre et Jean",
    "La Chartreuse de Parme", "Eugénie Grandet", "Illusions perdues",
    "La Reine Margot", "Vingt ans après", "L'Aiglon",
    "Lorenzaccio", "On ne badine pas avec l'amour", "Andromaque (Racine)", "Britannicus",
    "Cinna (Corneille)", "Le Mariage de Figaro", "Le Barbier de Séville",
    "Le Jeu de l'amour et du hasard", "Zadig", "Micromégas", "L'Ingénu",
    "Julie ou la Nouvelle Héloïse", "Jacques le fataliste", "La Religieuse",
    "Manon Lescaut", "Gil Blas", "Lettres persanes", "Les Rêveries du promeneur solitaire",
    "Fables de La Fontaine", "Contes de Perrault", "Le Grand Meaulnes",
    "Le Diable au corps (Radiguet)", "Le Horla",
]


def fetch_wikipedia(n: int = 80) -> list[dict]:
    return _fetch_mediawiki(
        "https://fr.wikipedia.org/w/api.php", WIKIPEDIA_TITLES, n, "wikipedia-fr"
    )


def _wikisource_candidates(title: str) -> list[str]:
    """Mainspace page first, then its subpages (novels live in Tome/ parts)."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "allpages", "apprefix": f"{title}/",
        "aplimit": 20, "format": "json",
    })
    try:
        data = json_loads(get(f"https://fr.wikisource.org/w/api.php?{params}"))
        subpages = [p["title"] for p in data.get("query", {}).get("allpages", [])]
        return [title] + subpages
    except Exception:
        return [title]


def fetch_wikisource(n: int = 40) -> list[dict]:
    """Public-domain literature via rendered (transcluded) page text.

    Mainspace Wikisource pages are often edition portals, so each title is
    tried on its main page then on its subpages until a usable extract
    (>= MIN_CHARS after trimming) is found.
    """
    records = []
    for title in WIKISOURCE_TITLES:
        if len(records) >= n:
            break
        candidates = _wikisource_candidates(title)
        time.sleep(PATIENCE_S)
        for candidate in candidates[:8]:
            params = urllib.parse.urlencode({
                "action": "parse", "page": candidate, "prop": "text",
                "format": "json", "redirects": 1,
            })
            try:
                data = json_loads(get(f"https://fr.wikisource.org/w/api.php?{params}"))
                rendered = data["parse"]["text"]["*"]
                time.sleep(PATIENCE_S)
            except Exception:
                continue
            url = f"https://fr.wikisource.org/wiki/{urllib.parse.quote(candidate)}"
            rec = _record(title, url, html_to_text(rendered), "litterature")
            if rec:
                records.append(rec)
                break
    return records


def _fetch_mediawiki(api: str, titles: list[str], n: int, source: str) -> list[dict]:
    records = []
    for title in titles:
        if len(records) >= n:
            break
        try:
            extract = _wiki_extract(api, title)
            time.sleep(PATIENCE_S)
        except Exception:
            continue
        url = f"{api.split('/w/')[0]}/wiki/{urllib.parse.quote(title)}"
        rec = _record(title, url, extract, source)
        if rec:
            records.append(rec)
    return records


# --------------------------------------------------------------------------
# Feeds (Ghost RSS, Atom) and listing-based press scraping
# --------------------------------------------------------------------------
CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}encoded"


def fetch_ghost_blog(n: int = 70, feed: str = "https://blog.maudet.cloud/rss.xml") -> list[dict]:
    """Personal editorial blog (Ghost). Paginates ?page= until n records."""
    records: list[dict] = []
    page = 1
    while len(records) < n and page <= 10:
        try:
            root = ET.fromstring(get(f"{feed}?page={page}" if page > 1 else feed))
        except Exception:
            break
        items = root.findall(".//item")
        if not items:
            break
        for item in items:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            content = item.findtext(CONTENT_NS) or item.findtext("description") or ""
            rec = _record(title, link, html_to_text(content), "blog-maudet")
            if rec:
                records.append(rec)
            if len(records) >= n:
                break
        page += 1
        time.sleep(PATIENCE_S)
    return records


def fetch_linuxfr(n: int = 40) -> list[dict]:
    """Technical blog posts from LinuxFr journals (Atom + page fetch)."""
    entries: list[tuple[str, str]] = []
    for feed in ("https://linuxfr.org/journaux.atom", "https://linuxfr.org/forums.atom"):
        try:
            root = ET.fromstring(get(feed))
        except Exception:
            continue
        for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title = entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
            link_el = entry.find("{http://www.w3.org/2005/Atom}link")
            link = link_el.get("href", "") if link_el is not None else ""
            if link:
                entries.append((title, link))
        time.sleep(PATIENCE_S)
    records: list[dict] = []
    for title, link in entries:
        if len(records) >= n:
            break
        try:
            rec = _record(title, link, html_to_text(get(link)), "linuxfr")
            time.sleep(PATIENCE_S)
        except Exception:
            continue
        if rec:
            records.append(rec)
    return records


def _scrape_listing(listing_urls: list[str], link_pattern: str, source: str, n: int) -> list[dict]:
    """Generic press collector: harvest article links from listing pages."""
    article_re = re.compile(link_pattern)
    links: list[str] = []
    for listing in listing_urls:
        try:
            page = get(listing)
        except Exception:
            continue
        for href in article_re.findall(page):
            full = href if href.startswith("http") else urllib.parse.urljoin(listing, href)
            if full not in links:
                links.append(full)
        time.sleep(PATIENCE_S)
        if len(links) >= n * 2:
            break
    records: list[dict] = []
    for link in links:
        if len(records) >= n:
            break
        try:
            page = get(link)
        except Exception:
            continue
        title_m = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S | re.I)
        title = html_to_text(title_m.group(1)) if title_m else link.rsplit("/", 1)[-1]
        rec = _record(title, link, html_to_text(page), source)
        time.sleep(PATIENCE_S)
        if rec:
            records.append(rec)
    return records


def fetch_presse(n: int = 60) -> list[dict]:
    """IT press: Le Monde Informatique first, LeMagIT tops up if short."""
    lmi = _scrape_listing(
        [f"https://www.lemondeinformatique.fr/actualites/toute-l-actualite-page-{i}.html"
         for i in range(1, 5)] + ["https://www.lemondeinformatique.fr/rss/"],
        r'href="(https://www\.lemondeinformatique\.fr/[^"]*lire-[^"]+-\d+\.html)"',
        "presse-fr", n,
    )
    if len(lmi) >= n:
        return lmi[:n]
    magit = _scrape_listing(
        ["https://www.lemagit.fr/actualites",
         "https://www.lemagit.fr/rubriques/infrastructure"],
        r'href="(https://www\.lemagit\.fr/(?:actualites|rubriques)/[^"]+-\d+)"',
        "presse-fr", n - len(lmi),
    )
    return (lmi + magit)[:n]
