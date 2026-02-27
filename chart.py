import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def fetch_page_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and len(response.text) > 100:
            from html.parser import HTMLParser
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip = False
                def handle_starttag(self, tag, attrs):
                    if tag in ['script', 'style', 'nav', 'header', 'footer']:
                        self.skip = True
                def handle_endtag(self, tag):
                    if tag in ['script', 'style', 'nav', 'header', 'footer']:
                        self.skip = False
                def handle_data(self, data):
                    if not self.skip and data.strip():
                        self.text.append(data.strip())
            parser = TextExtractor()
            parser.feed(response.text)
            full_text = " ".join(parser.text)
            return full_text[:3000]
    except:
        pass
    return None

def search_wine_reviews(wine_name):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    url = "https://google.serper.dev/search"
    queries = [
        f"{wine_name} tasting notes cellartracker",
        f"{wine_name} tasting notes vivino",
        f"{wine_name} tasting notes review",
        f"{wine_name} site:decanter.com tasting notes",
        f"{wine_name} barnyard brett funky earthy tasting notes",
        f"{wine_name} avis degustation notes dégustation",
    ]
    snippets = []
    sources = []
    for query in queries:
        payload = {"q": query, "num": 5}
        response = requests.post(url, headers=headers, json=payload)
        results = response.json()
        for r in results.get("organic", []):
            link = r.get("link", "")
            if link and link not in [s["link"] for s in sources]:
                # Always include the snippet first
                if "snippet" in r and r["snippet"]:
                    snippets.append(r["snippet"])
                # Try to fetch full page for non-blocked sites
                if not any(blocked in link for blocked in ["vivino.com", "cellartracker.com"]):
                    full_text = fetch_page_text(link)
                    if full_text:
                        snippets.append(full_text)
                sources.append({
                    "title": r.get("title", "Unknown"),
                    "link": link
                })
    return "\n\n".join(snippets), sources

def analyse_wine(wine_name, review_text):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are an expert wine critic and sommelier with deep knowledge of natural wines,
brett (brettanomyces), and funky/oxidative wine styles.

IMPORTANT: Reviews may be in any language including Danish, French, German, Spanish etc.
You must recognise and score based on descriptors in ALL languages.

Analyse the following reviews of "{wine_name}" and score each parameter from 0-10.
Be sensitive to subtle language — reviewers rarely use technical terms directly,
so you must interpret descriptive language carefully across all languages.

SCORING GUIDANCE:

Quality (0-10): Overall quality and pleasure. Use any mention of scores (e.g. 90 points = ~8,
95 points = ~9), words like "exceptional", "outstanding", "disappointing", "simple" etc.

Acidity (0-10): Look for words like "fresh", "crisp", "tart", "bright", "zippy", "lively",
"sharp", "sour", "low acid", "flat", "flabby". Danish: "syre", "frisk", "syrlig".

Tannins (0-10): Look for "grippy", "firm", "chewy", "silky", "smooth", "velvety", "astringent",
"drying", "fine-grained", "dusty", "soft tannins", "tannic". Danish: "tannin", "garvesyre".

Body (0-10): Look for "full-bodied", "medium-bodied", "light-bodied", "weighty", "substantial",
"lean", "thin", "concentrated", "dense". Danish: "fyldig", "let", "kraftig".

Fruitiness (0-10): Look for fruit descriptors like "red fruits", "black fruits", "cherry",
"plum", "raspberry", "citrus", "tropical", "stone fruit". Danish: "frugt", "kirsebær",
"jordbær", "hindbær", "solbær", "blomme". Low score if described as "savory" or "earthy"
with little fruit.

Finish (0-10): Look for "long finish", "persistent", "lingering", "short finish".
Danish: "eftersmag", "lang", "kort".

Complexity (0-10): Look for "complex", "layered", "multidimensional", "nuanced", "evolving",
"simple", "straightforward". Danish: "kompleks", "lagdelt", "enkel".

Funky (0-10): Score based on SMELL AND TASTE ONLY - not production methods.
Look for these words in ANY language:
English: "funky", "wild", "natural", "earthy", "smoke", "gunpowder", "reduction",
"reductive", "sulfur", "struck match", "volatile", "VA", "brett-like", "sweaty",
"unconventional", "acquired taste", "natural wine character".
Danish: "funky", "vild", "naturvin", "jordagtig", "røg", "reduktiv", "ustabil",
"elektrisk", "beskidt", "udfordrende", "naturlig".
French: "sauvage", "terreux", "fume", "reduit", "volatile".
IMPORTANT: If ANY reviewer says the wine has "too much" of these characteristics,
or explicitly complains about gunpowder, funkiness or natural wine taste, score 8 or higher.
If a descriptor is mentioned negatively or as a flaw, that means it is very present - score high.
Scoring:
- 0-2: None of these descriptors present. Clean, conventional style.
- 3-4: One or two mild hints mentioned positively.
- 5-6: Clearly present but not dominant.
- 7-8: Strongly present, mentioned by multiple reviewers OR mentioned as excessive/a flaw.
- 9-10: Overwhelming, repeatedly flagged as too much or a major characteristic.

Brett (0-10): Look for these words in ANY language:
English: "brett", "brettanomyces", "barnyard", "horse", "horse saddle", "stable",
"wet dog", "band-aid", "earthy", "leather", "animal", "farm", "manure".
Danish: "hestestald", "stald", "hest", "lo", "staldlugt", "laeder", "dyrisk",
"landlig", "bondegaard", "beskidt".
French: "ecurie", "cheval", "cuir", "animal", "etable", "fumier".
Also score higher if the wine is described as having strong terroir character
in a specifically animal or earthy sense in any language.
IMPORTANT: If ANY reviewer explicitly mentions brett by name or says there is "lots of brett",
score 8 or higher immediately. A complaint about brett means it is very present.
Scoring:
- 0-2: No brett descriptors present whatsoever.
- 3-4: One mild hint mentioned once.
- 5-6: Clearly present, mentioned positively or neutrally.
- 7-8: Strongly present, explicitly named or mentioned as excessive by any reviewer.
- 9-10: Brett is dominant, repeatedly mentioned or flagged as a major flaw by multiple reviewers.

Alcohol (0-10): Look for actual % if mentioned (under 12% = 3, 12-13% = 5, 13-14% = 7,
14-15% = 8, over 15% = 9-10). Also words like "hot", "warming", "heady", "low alcohol",
"light", "easy drinking". Danish: "alkohol", "let drikkelig", "varm".

Sweetness (0-10): Look for "dry", "bone dry", "off-dry", "hint of sweetness", "semi-sweet",
"sweet", "residual sugar". Bone dry = 1, dry = 2, off-dry = 4, semi-sweet = 6, sweet = 8+.
Danish: "tor", "sod", "restsodme".

Reviews:
{review_text}

Respond ONLY with a JSON object in this exact format, no other text, no markdown, no backticks:
{{
  "wine_type": "Red",
  "scores": {{
    "Quality": 8,
    "Acidity": 7,
    "Tannins": 9,
    "Body": 8,
    "Fruitiness": 7,
    "Finish": 9,
    "Complexity": 9,
    "Funky": 2,
    "Brett": 1,
    "Alcohol": 7,
    "Sweetness": 2
  }}
}}

wine_type must be one of: Red, White, Rose, Orange"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    result = json.loads(raw)
    return result

def draw_chart(wine_name, wine_type, scores):
    color_map = {
        "Red":    {"line": "#8B0000", "fill": "#8B0000"},
        "White":  {"line": "#7A8C00", "fill": "#7A8C00"},
        "Rose":   {"line": "#DC6478", "fill": "#DC6478"},
        "Orange": {"line": "#C87814", "fill": "#C87814"},
    }
    alpha_map = {"Red": 0.35, "White": 0.2, "Rose": 0.2, "Orange": 0.2}
    line_color = color_map.get(wine_type, color_map["Red"])["line"]
    fill_color = color_map.get(wine_type, color_map["Red"])["fill"]
    alpha = alpha_map.get(wine_type, 0.35)
    categories = ['Quality', 'Acidity', 'Tannins', 'Body', 'Fruitiness',
                  'Finish', 'Complexity', 'Funky', 'Brett', 'Alcohol', 'Sweetness']
    score_values = [scores[c] for c in categories]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    scores_closed = score_values + [score_values[0]]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.plot(angles_closed, scores_closed, color=line_color, linewidth=2)
    ax.fill(angles_closed, scores_closed, color=fill_color, alpha=alpha)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color='lightgrey', linestyle='-', linewidth=0.8)
    ax.xaxis.grid(True, color='lightgrey', linestyle='-', linewidth=0.8)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)
    ax.tick_params(axis='x', pad=12)
    ax.spines['polar'].set_color('lightgrey')
    ax.set_rlabel_position(0)
    ax.set_title(wine_name, pad=20, fontsize=13, fontweight='bold')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    return fig

# --- App ---
st.title("🍷 Wine Reviewer")

with st.form("wine_form"):
    wine_name = st.text_input("Wine Name", placeholder="e.g. Chateau Margaux 2018")
    submitted = st.form_submit_button("Analyse Wine")

if submitted and wine_name:
    with st.spinner("Searching for reviews..."):
        review_text, sources = search_wine_reviews(wine_name)

    if not review_text:
        st.error("No reviews found. Try a different wine name.")
    else:
        with st.spinner("Analysing reviews with AI..."):
            try:
                result = analyse_wine(wine_name, review_text)
                wine_type = result["wine_type"]
                scores = result["scores"]
                fig = draw_chart(wine_name, wine_type, scores)
                st.pyplot(fig, use_container_width=False)

                with st.expander("📝 Raw review snippets"):
                    st.write(review_text)

                with st.expander("📚 Sources used for this analysis"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**{i}. [{source['title']}]({source['link']})**")

            except Exception as e:
                st.error(f"Something went wrong analysing the wine: {e}")

elif submitted and not wine_name:
    st.warning("Please enter a wine name.")