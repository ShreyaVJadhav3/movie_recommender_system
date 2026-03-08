# 🎬 Movie Recommender System — Content-Based Filtering | Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-CountVectorizer%20%7C%20Cosine%20Similarity-orange)
![NLP](https://img.shields.io/badge/NLP-Porter%20Stemming%20%7C%20BoW-green)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red)
![Dataset](https://img.shields.io/badge/Dataset-TMDB%205000%20Movies-purple)

---

## 📌 Project Overview

This project builds a **content-based movie recommendation engine** that suggests the **top 5 most similar films** for any given movie title. The system works by understanding *what a movie is about* — its genres, plot keywords, top cast members, and director — and finding other films that share the most similar content profile.

Unlike collaborative filtering (which needs user behaviour data), this system works from film metadata alone, making it robust and explainable.

**Test query used:** `recommend('Batman Begins')` → returns 5 contextually similar films based on content vectors.

---

## 🗂️ Project Pipeline

```
movies.csv  +  credits.csv  (TMDB 5000 Dataset)
        ↓
Merge both datasets on 'title'
        ↓
Column Selection — 23 columns reduced to 7 relevant columns
        ↓
Data Cleaning — Null removal (3 rows dropped), duplicate check (0 found)
        ↓
Nested JSON Parsing — ast.literal_eval on genres, keywords, cast, crew
        ↓
Feature Engineering — Top 3 cast extracted + director only from crew
        ↓
Space Removal — Eliminate spaces within names to prevent token splitting
        ↓
TAGS Column Creation — overview + genres + keywords + cast + crew merged
        ↓
Lowercasing + Porter Stemming — NLP preprocessing
        ↓
CountVectorizer — 5,000-feature Bag-of-Words vectorization
        ↓
Cosine Similarity Matrix — Pairwise similarity across all 4,806 films
        ↓
recommend() function — Top 5 closest vectors returned
        ↓
Pickle Serialization — movies.pkl + movie_dict.pkl + similarity.pkl
        ↓
Streamlit App (app.py) — Live deployment
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python (Pandas, NumPy)** | Data loading, merging, cleaning |
| **ast.literal_eval** | Safe parsing of nested JSON-like string columns |
| **Scikit-Learn CountVectorizer** | Bag-of-Words text vectorization (max 5,000 features) |
| **Scikit-Learn cosine_similarity** | Pairwise similarity computation across all film vectors |
| **NLTK PorterStemmer** | Word stemming — `loving` → `love`, `dancing` → `danc` |
| **Pickle** | Model serialization for fast Streamlit loading |
| **Streamlit** | Interactive web app for live recommendations |

---

## 📊 Dataset

**Source:** [TMDB 5000 Movie Dataset — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Two CSV files merged on `title`:

| File | Columns | Description |
|---|---|---|
| `movies.csv` | 19 columns | Title, overview, genres, keywords, budget, revenue etc. |
| `credits.csv` | 3 columns | Title, cast, crew (director info) |

**After merge:** 23 columns → reduced to 7 working columns → final model uses 3 columns only (`movie_id`, `title`, `tags`).

---

## 🔍 Feature Engineering — Building the TAGS Column

The entire recommendation logic rests on one engineered column: **`tags`**.

### Step 1 — Column Selection
From 23 merged columns, only 7 were kept:
`movie_id`, `title`, `overview`, `genres`, `keywords`, `cast`, `crew`

**Removed 15 columns:** `vote_count`, `vote_average`, `tagline`, `status`, `spoken_languages`, `runtime`, `revenue`, `release_date`, `production_countries`, `production_companies`, `popularity`, `original_title`, `original_language`, `homepage`, `budget`

### Step 2 — Nested JSON Parsing with ast.literal_eval

Columns like `genres`, `keywords`, and `cast` were stored as stringified Python lists of dictionaries:

```python
# Raw format in dataset
"[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]"

# Custom convert() function to extract only names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
# Result → ['Action', 'Adventure']
```

Applied to both `genres` and `keywords` columns.

### Step 3 — Top 3 Cast Members Only

```python
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter + 1
        else:
            break
    return L
```

Only the top 3 billed cast members were extracted — minor roles beyond 3 add noise without improving recommendation quality.

### Step 4 — Director Extraction from Crew

```python
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
```

The director is the single most influential creative signal in a film's identity — extracted from the full crew dictionary by filtering `job == 'Director'`.

### Step 5 — Space Removal from Names

```python
# "Sam Mendes" → "SamMendes" — treated as one token, not two separate words
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
```

Without this, `Christopher Nolan` would be split into `Christopher` and `Nolan` — causing false matches with other people named Christopher.

### Step 6 — TAGS Column Creation

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# All 5 lists concatenated per row → joined into a single paragraph string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```

Each movie is now represented as one paragraph of its most meaningful content signals.

---

## 🤖 Machine Learning — Model Building

### Vectorization — Bag of Words (CountVectorizer)

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# Output shape: (4806, 5000) — each film as a 5,000-dimensional vector
```

Common English stop words (`the`, `is`, `and`, `in`) are automatically removed. The top 5,000 most frequent meaningful words across the corpus form the feature space.

### NLP — Porter Stemming

Applied to reduce words to their root form before the final model:

```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
```

Ensures `loved`, `loving`, `lover` all map to `love` — improving similarity matching across morphological variants of the same word.

### Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
# Output shape: (4806, 4806) — similarity score for every film pair
```

**Why cosine over Euclidean distance?** Cosine similarity measures the *angle* between vectors, not their magnitude — making it robust to differences in overview length between films. A short tagline and a long synopsis can still be accurately compared.

- Smaller angle (θ) between two vectors = **more similar** movies
- Larger angle (θ) = **more dissimilar** movies

### Recommendation Function

```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]  # fetch index
    distances = similarity[movie_index]                        # similarity row
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)
```

1. Fetch the index of the input film in the dataframe
2. Pull its entire cosine similarity row (4,806 scores)
3. Sort all scores descending
4. Skip rank 0 (the film itself) → return titles at ranks 1–5

---

## 💾 Pickle Serialization for Deployment

Three files serialized to avoid recomputation on every app load:

```python
import pickle

pickle.dump(new_df, open('movies.pkl', 'wb'))                    # cleaned dataframe
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))      # dict for dropdown
pickle.dump(similarity, open('similarity.pkl', 'wb'))            # similarity matrix
```

The Streamlit app loads these files once at startup → enabling sub-second recommendations across 4,806 films.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ShreyaVJadhav3/movie_recommender_system.git
cd movie_recommender_system
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn nltk streamlit
```

### 3. Run the notebook
Open `movie_recommender_system.ipynb` and run all cells — this regenerates the pickle files.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
movie_recommender_system/
│
├── movie_recommender_system.ipynb   # Full ML pipeline — EDA + model building
├── app.py                           # Streamlit web app
├── movies.pkl                       # Serialized cleaned dataframe
├── movie_dict.pkl                   # Dictionary format for Streamlit dropdown
├── credits.zip                      # Raw TMDB credits dataset (compressed)
└── README.md
```

> **Note:** `similarity.pkl` is not in the repo due to file size (~95MB). Regenerate it by running the notebook locally.

---

## 📌 Related Projects

- 🛡️ [Project Sentinel — E-Commerce Fraud Detection](https://github.com/ShreyaVJadhav3/project---sentinel---ecommerce---fraud---detection-) — Detected 586 ghost orders & R$80,860 revenue leakage across 99K transactions using SQL, Python & Power BI
- 🤖 [RAG Pipeline — Personal Document Chatbot](https://github.com/ShreyaVJadhav3/rag-pipeline) — Production-style GenAI pipeline using LangChain, FAISS & ChromaDB
- 🎬 [IMDB Movie Market Analysis](https://github.com/ShreyaVJadhav3/databricks-pyspark-movie-market-analysis) — Large-scale EDA using PySpark on Databricks with window functions

---

## 👩‍💻 Author

**Shreya Jadhav**
Data Analyst | Python · SQL · Power BI · Machine Learning · GenAI
📧 shreyajune03pune@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/jadhavshreya03pune) | [GitHub](https://github.com/ShreyaVJadhav3)
