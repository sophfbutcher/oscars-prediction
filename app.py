"""
Oscars Best Picture predictor – Streamlit app
Shows the predictions of the logistic regression model from
Sophie Butcher's IEEE MPCON 2026 paper.
"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Oscars Best Picture predictor", page_icon="🎬")

# ----------------------------------------------------------------
# Settings – fill in your links (leave as "" to hide them)
# ----------------------------------------------------------------
PAPER_URL = "https://ieeexplore.ieee.org/document/11508236"   # link to your IEEE paper
REPO_URL = "https://github.com/sophfbutcher/oscars-prediction"    # link to your GitHub repo
VARADI_URL = "https://github.com/MateVaradi/OscarPrediction"

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "best_picture_model.joblib"
PREDICTIONS_PATH = HERE / "data" / "best_picture_predictions.csv"

WINNER_COLOUR = "#2a78d6"   # blue: the film that actually won
NOMINEE_COLOUR = "#a3a29c"  # grey: the other nominees
UP_COLOUR = "#2a78d6"       # blue: feature increases the odds
DOWN_COLOUR = "#eb6834"     # orange: feature decreases the odds

# Friendlier names for the "what drives the model" chart
READABLE = {
    "Total_Oscar_noms": "Total Oscar nominations",
    "Oscar_nom_director": "Nominated for Best Director",
    "Rating_IMDB": "IMDb rating",
    "Rating_rtaudience": "Rotten Tomatoes audience score",
    "Rating_rtcritic": "Rotten Tomatoes critic score",
    "BAFTA_Nom": "BAFTA nomination", "BAFTA_Win": "BAFTA win",
    "GG_comedy_nom": "Golden Globe nomination (comedy)",
    "GG_comedy_win": "Golden Globe win (comedy)",
    "GG_drama_nom": "Golden Globe nomination (drama)",
    "GG_drama_win": "Golden Globe win (drama)",
    "DGA_Nom": "Directors Guild nomination", "DGA_Win": "Directors Guild win",
    "PGA_Nom": "Producers Guild nomination", "PGA_Win": "Producers Guild win",
    "Best_Editing_Nom": "Nominated for Best Editing",
    "Runtime": "Runtime",
}


def readable(name: str) -> str:
    if name in READABLE:
        return READABLE[name]
    if name.startswith("Genre_"):
        return "Genre: " + name.removeprefix("Genre_")
    if name.startswith("MPAA_"):
        return "Rated " + name.removeprefix("MPAA_")
    if name.startswith("Release_"):
        return "Released in " + name.removeprefix("Release_")
    return name


# ----------------------------------------------------------------
# Load the files once and keep them in memory (caching)
# ----------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


bundle = load_model()
preds = load_predictions()

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.title("🎬 Oscars Best Picture predictor")
st.write(
    "A logistic regression model that estimates each Best Picture nominee's "
    "chance of winning, using its other nominations, precursor awards (BAFTA, "
    "Golden Globes, Directors and Producers Guilds), ratings, genre and release timing."
)
links = [f"[Read the paper]({PAPER_URL})" if PAPER_URL else "",
         f"[Code on GitHub]({REPO_URL})" if REPO_URL else ""]
if any(links):
    st.markdown(" · ".join(l for l in links if l))

# Headline results on the test set (the years the model never saw)
test = preds[preds["In_test_set"] == 1]
picks = test.loc[test.groupby("Oscar_Year")["Win_Probability"].idxmax()]
n_years, n_right = len(picks), int(picks["Winner"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Test-set ceremonies", n_years)
c2.metric("Top pick won", f"{n_right} of {n_years}")
c3.metric("Top-pick accuracy", f"{n_right / n_years:.0%}")
st.caption(
    "The test set is the 20% of ceremonies held back from training, "
    "so the model never saw these results."
)

st.divider()

# ----------------------------------------------------------------
# Pick a ceremony
# ----------------------------------------------------------------
show_all = st.toggle(
    "Show all years, including the ones the model was trained on",
    value=False,
)
pool = preds if show_all else test
years = sorted(pool["Oscar_Year"].unique(), reverse=True)
test_years = set(test["Oscar_Year"].unique())


def year_label(y):
    return f"{y}" if y in test_years else f"{y} (training year)"


year = st.selectbox("Ceremony year", years, format_func=year_label)

if year not in test_years:
    st.info(
        "The model saw this year's result during training. "
        "Treat this as a check on how well it fits, not a true prediction."
    )

# ----------------------------------------------------------------
# Result for that year
# ----------------------------------------------------------------
year_df = preds[preds["Oscar_Year"] == year].sort_values("Win_Probability", ascending=False).copy()
pick = year_df.iloc[0]
winner = year_df[year_df["Winner"] == 1].iloc[0]

if pick["Film"] == winner["Film"]:
    st.success(f"✅ The model's pick, **{pick['Film']}** ({pick['Win_Probability']:.0%}), won.")
else:
    st.error(
        f"❌ The model picked **{pick['Film']}** ({pick['Win_Probability']:.0%}), "
        f"but **{winner['Film']}** won ({winner['Win_Probability']:.0%})."
    )

year_df["Result"] = year_df["Winner"].map({1: "Actual winner", 0: "Nominee"})
year_df["Label"] = year_df.apply(
    lambda r: f"🏆 {r['Film']}" if r["Winner"] == 1 else r["Film"], axis=1)
year_df["Chance"] = year_df["Win_Probability"].map(lambda p: f"{p:.0%}")

bars = alt.Chart(year_df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
    x=alt.X("Win_Probability:Q", title="Predicted chance of winning",
            axis=alt.Axis(format="%", grid=True), scale=alt.Scale(domain=[0, 1])),
    y=alt.Y("Label:N", sort=list(year_df["Label"]), title=None, axis=alt.Axis(labelLimit=320)),
    color=alt.Color("Result:N",
                    scale=alt.Scale(domain=["Actual winner", "Nominee"],
                                    range=[WINNER_COLOUR, NOMINEE_COLOUR]),
                    legend=alt.Legend(title=None, orient="top")),
    tooltip=[alt.Tooltip("Film:N"), alt.Tooltip("Chance:N", title="Predicted chance"),
             alt.Tooltip("Result:N")],
)
labels = bars.mark_text(align="left", dx=4).encode(
    text="Chance:N", color=alt.value("gray"))

st.altair_chart((bars + labels).properties(height=40 * len(year_df) + 40),
                width="stretch")
st.caption("Chances are scaled so each year's nominees add up to 100%.")

# ----------------------------------------------------------------
# All test-set years at a glance
# ----------------------------------------------------------------
with st.expander("See every test-set ceremony"):
    winners = test[test["Winner"] == 1].set_index("Oscar_Year")["Film"]
    summary = pd.DataFrame({
        "Year": picks["Oscar_Year"].values,
        "Model's pick": picks["Film"].values,
        "Predicted chance": picks["Win_Probability"].map(lambda p: f"{p:.0%}").values,
        "Actual winner": winners.reindex(picks["Oscar_Year"]).values,
        "Correct": picks["Winner"].map({1: "✅", 0: "❌"}).values,
    }).sort_values("Year", ascending=False)
    st.dataframe(summary, hide_index=True, width="stretch")

# ----------------------------------------------------------------
# What drives the model – read straight from the saved model
# ----------------------------------------------------------------
with st.expander("What drives the model's predictions?"):
    coefs = pd.DataFrame({
        "Feature": [readable(f) for f in bundle["features"]],
        "Coefficient": bundle["model"].coef_[0],
    })
    coefs["abs"] = coefs["Coefficient"].abs()
    top = coefs.nlargest(10, "abs")
    top["Effect"] = top["Coefficient"].map(lambda c: "Increases odds" if c > 0 else "Decreases odds")

    coef_chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Coefficient:Q", title="Coefficient (logistic regression)"),
        y=alt.Y("Feature:N", sort=alt.EncodingSortField("abs", order="descending"),
                title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Effect:N",
                        scale=alt.Scale(domain=["Increases odds", "Decreases odds"],
                                        range=[UP_COLOUR, DOWN_COLOUR]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Feature", alt.Tooltip("Coefficient:Q", format=".2f"), "Effect"],
    ).properties(height=360)
    st.altair_chart(coef_chart, width="stretch")
    st.caption(
        "The ten features with the largest coefficients. Continuous features were "
        "scaled to 0–1 before training, so the sizes are roughly comparable."
    )

# ----------------------------------------------------------------
# About
# ----------------------------------------------------------------
st.divider()
st.markdown(
    f"""
**About the model.** Logistic regression trained on Best Picture nominees from
the 1961 to 2026 ceremonies, with an 80/20 train/test split grouped by ceremony
year so that every nominee from a given year sits on the same side of the split.

**Data.** Built on [Máté Váradi's Oscar prediction dataset]({VARADI_URL}),
extended with data cleaning, additional features and 2026 ceremony.
"""
)
