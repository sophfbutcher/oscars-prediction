# A Machine Learning Approach to Predicting the Oscars

Code and data for **"A Machine Learning Approach to Predicting the Oscars"** (Sophie Butcher and Jad Abbass, Kingston University), presented at IEEE MPCON 2026. [DOI: 10.1109/MPCON69668.2026.11508236; https://ieeexplore.ieee.org/document/11508236]

## Overview

Can machine learning predict who wins at the Oscars, aka. the biggest awards ceremony in Hollywood? 

I've spent years watching movies and writing about them, working as a film journalist at EMPIRE magazine before moving into data science, and wanted to test whether the Academy Awards are as predictable as critics assume – as well as some of my own theories.

This project models winners in the six major Academy Award categories – Best Picture, Best Director, and the four acting awards – using every nominee from 1961 to 2025. This was my dissertation project for my MSc Data Science, studied at Kingston University, for which I received a Distinction grade. Jad Abbass was my dissertation supervisor, and assisted me with preparing and submitting the paper for publication.

(The datasets were updated post-publishing to include the 2026 nominees, and the results in the modelling notebook include these 2026 data points. The README below references results from the paper, which includes data up to 2025 only).

My central hypothesis was that different categories need different features and modelling approaches, and that the features that contribute most to predictability will also vary. In particular, I chose to model male and female acting categories separately to look for gender-based differences, which earlier Oscar prediction research hasn't done. This split turned out to be key: the factors that predict a win differ noticeably between male and female performers.

**Headline results (top pick accuracy):** Best Picture 85%, Best Director 92%, Male Acting 69%, Female Acting 69%. These meet or beat benchmarks from previous research.

## Data

Three datasets cover all nominees from the 1961 to 2025 ceremonies:

| Dataset | Nominees |
|---|---|
| Best Picture | 390 |
| Best Director | 325 |
| Acting (Lead and Supporting Actor and Actress) | 1,300 |

The starting point was Máté Váradi's Oscar prediction dataset ([GitHub](https://github.com/MateVaradi/OscarPrediction)), which includes precursor award results (BAFTA, Golden Globes, Critics' Choice, SAG, DGA, PGA), Oscar history, genre, release timing and ratings sourced from IMDb and Rotten Tomatoes.

I cleaned and extended it with:
- **Missing data filled:** 19 missing ages and 15 entirely missing acting nominees, identified by checking expected nominee counts per year, then filled manually from the Academy's official ceremony pages and IMDb
- **New features for all categories:** film runtime and whether the film was produced in the USA, collected via a TMDb API scraper (99.9% match rate, with mismatches checked and corrected by hand)
- **Best Picture:** whether the film was also nominated for Best Film Editing - a feature I suspected would be highly predictive from my experience in the industry
- **Best Director:** previous nominee/winner flags, age at the ceremony, US birthplace and gender (IMDb non-commercial datasets and Wikidata)
- **Acting:** US birthplace (Wikidata), and whether the role portrays a real person (entirely researched and verified by hand)

The final datasets are also archived on Zenodo: https://zenodo.org/records/17494990

## Methods

- **Framing:** each nominee gets a predicted probability of winning, and these are ranked within each ceremony year. The highest-ranked nominee is the predicted winner ("top pick accuracy"). A true multinomial model would have left only around 65 rows per dataset (one per year), which is too few to train on.
- **Train/test split:** `GroupShuffleSplit` grouped by Oscar year, so nominees from the same ceremony never appear in both training and test sets, while years are still shuffled to avoid bias from trends over time.
- **Feature analysis:** features were removed one at a time to test their contribution. Rating features (IMDb, Rotten Tomatoes) were also tested for removal, because they're mostly collected after the ceremony and so wouldn't be available for genuine pre-ceremony prediction.
- **Models compared:** Logistic Regression, Support Vector Machines and Random Forest.
- **Evaluation:** top pick accuracy, AUC-ROC and F1 score, validated with 5-fold cross-validation. F1 was reported alongside AUC-ROC because AUC-ROC alone can be misleading on imbalanced data (there are far more losing nominees than winners).

## Results

| Category | Best model | Top pick accuracy | AUC-ROC | F1 score |
|---|---|---|---|---|
| Best Picture | Logistic Regression | 85% | 0.930 | 0.769 |
| Best Director | Logistic Regression | 92% | 0.949 | 0.892 |
| Male Acting | Logistic Regression | 69% | 0.847 | 0.656 (Lead) / 0.662 (Supporting) |
| Female Acting | SVM | 69% | 0.817 | 0.692 (Lead) / 0.523 (Supporting) |

Key findings:
- **A Directors Guild of America win** was the strongest predictor for both Best Picture and Best Director.
- **The new Best Editing nomination feature** was the second most important predictor for Best Picture.
- **Male and female acting categories behave differently.** Male winners were driven by SAG wins and age, with previous Oscar wins *lowering* the odds. Female winners were more affected by genre (war, crime, biography) and release timing. SVM outperformed Logistic Regression only for female acting, which suggests non-linear patterns in how female performances are judged.
- **Random Forest overfitted heavily** on every dataset (perfect training scores, poor generalisation), likely due to small sample sizes, so it wasn't used in the final models.

## Repository structure

```
├── notebooks/
│   ├── 01_data_cleaning.ipynb   # loading, cleaning, scraping and feature engineering
│   └── 02_modelling.ipynb       # model training, evaluation and feature importance
├── data/                        # final cleaned datasets
├── paper/                       # accepted manuscript
└── LICENSE
```

## Running the notebooks

The notebooks were developed in Google Colab. They load data from Google Drive, so the file paths would need updating to run elsewhere. The cleaning notebook's TMDb scraper needs your own free [TMDb API key](https://www.themoviedb.org/settings/api).

Main packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`, `beautifulsoup4`.

## Limitations

- The datasets are small (65 ceremonies), which limits model complexity and makes results sensitive to the train/test split. This is especially true for female acting, which showed the highest variance in cross-validation.
- The binary classification approach produces per-nominee probabilities that don't sum to 1 within a year. It ranks nominees well but isn't a calibrated probability distribution.
- Some features come from community-maintained sources (TMDb, Wikidata), so occasional inaccuracies are possible. These were mitigated through spot-checking and manual correction.

## Future work

Multinomial probability modelling, analysing how voting trends have shifted across decades, extending to more categories, and testing deep learning models for the female acting categories.

## Citation

> S. Butcher and J. Abbass, "A Machine Learning Approach to Predicting the Oscars," IEEE MPCON 2026. [DOI:10.1109/MPCON69668.2026.11508236]

## Acknowledgements

Builds on the dataset and approach of Máté Váradi, *Predicting the Oscars with machine learning* (2018), with additional data from [TMDb](https://www.themoviedb.org/), [IMDb](https://developer.imdb.com/non-commercial-datasets/), [Wikidata](https://www.wikidata.org/) and the [Academy's official ceremony records](https://www.oscars.org/oscars/ceremonies). This product uses the TMDb API but is not endorsed or certified by TMDb.
