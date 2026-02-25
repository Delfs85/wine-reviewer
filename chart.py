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

def search_wine_reviews(wine_name):
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    url = "https://google.serper.dev/search"

    queries = [
        f"{wine_name} site:cellartracker.com tasting notes",
        f"{wine_name} site:vivino.com tasting notes",
        f"{wine_name} wine review tasting notes"
    ]

    snippets = []
    sources = []

    for query in queries:
        payload = {"q": query, "num": 3}
        response = requests.post(url, headers=headers, json=payload)
        results = response.json()
        for r in results.get("organic", []):
            if "snippet" in r and r.get("link") not in [s["link"] for s in sources]:
                snippets.append(r["snippet"])
                sources.append({
                    "title": r.get("title", "Unknown"),
                    "link": r.get("link", "")
                })

    return "\n\n".join(snippets), sources

def analyse_wine(wine_name, review_text):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    prompt = f"""You are a wine expert. Based on the following reviews of "{wine_name}", 
extract scores from 0-10 for each of these parameters. Also determine the wine type.

Parameters to score:
- Quality (overall quality)
- Acidity
- Tannins
- Body
- Fruitiness
- Finish
- Complexity
- Funky (how funky/natural does it smell and taste - gunpowder, barnyard, natural wine character in aroma and taste only, not production method)
- Brett (brettanomyces character)
- Alcohol (how high is the alcohol level)
- Sweetness

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

wine_type must be one of: Red, White, Rosé, Orange"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
        "Rosé":   {"line": "#DC6478", "fill": "#DC6478"},
        "Orange": {"line": "#C87814", "fill": "#C87814"},
    }
    alpha_map = {"Red": 0.35, "White": 0.2, "Rosé": 0.2, "Orange": 0.2}

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
    wine_name = st.text_input("Wine Name", placeholder="e.g. Château Margaux 2018")
    submitted = st.form_submit_button("Analyse Wine")

if submitted and wine_name:
    with st.spinner("Searching Cellartracker, Vivino and other sources..."):
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