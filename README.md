# 🎧 Spotify Million Playlist Challenge — Playlist Continuation

## 1. Problem Statement & General Approach

The goal of this project is to implement a **data processing and recommendation pipeline** to tackle the **Spotify Million Playlist Dataset Challenge**, available and fully describe here: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

Given a **partially known playlist** — which may include:

* a playlist name,
* a set of seed tracks,
* or both —

the objective is to **recommend 500 tracks**, ranked by relevance, that best match the playlist’s characteristics.
Recommendations are derived from a **training dataset of playlists**, leveraging user listening behavior captured at scale.

A key challenge lies in:

* capturing meaningful **similarity between tracks and playlists**, and
* designing a solution that is **computationally efficient** and scalable in principle.

For development and experimentation purposes, a **10k playlist subset** was used as the training dataset.
All results presented here are based on this subset.

The algorithm was carried out in the spirit of the Computing Foundations of Data Science course from the Specialized Master in Data Science – Big Data curriculum (ULB).

---

## 🛠️ Technologies & Libraries

* **Python**
* **Pandas**
* **NumPy**
* **JSON**
* **scikit-learn** (TF-IDF, cosine similarity)
* **thefuzz** (fuzzy string matching)

---

## 📊 Datasets

Two datasets were used:

### 1. Training Dataset

* A subset of the **Spotify Million Playlist Dataset**
* Composed of nested JSON files, each containing multiple playlists
* Each playlist contains:

  * playlist metadata
  * a list of tracks

### 2. Challenge Dataset

* Contains playlists to be completed
* Depending on the case, a playlist may contain:

  * only a name
  * only seed tracks
  * both name and seed tracks

---

## 2. Similarity Modelling

### Track Similarity

To capture similarity between tracks, several approaches were considered. The selected approach relies exclusively on **co-occurrence information** from playlists.

#### Track Similarity Based on Shared Playlists

The core idea is:

> **If two tracks frequently appear together in playlists, they are similar.**

For a track **( t )**, define:

* **( P(t) )**: the set of playlists that contain track ( t )

The similarity between two tracks **( t_1 )** and **( t_2 )** is computed using **cosine similarity**:

```math
\text{sim}(t_1, t_2) =
\frac{|P(t_1) \cap P(t_2)|}
{\sqrt{|P(t_1)| \cdot |P(t_2)|}}
```

This produces a similarity score in ([0,1]).

#### Implementation Details

* Pre-compute:

  * a dictionary mapping each `track_uri` to its set of playlist IDs
  * the cardinality of each playlist set
* Use dictionary lookups to avoid repeated DataFrame operations
* Handle missing tracks safely by returning a similarity score of 0

This results in a fast and reusable `track_sim_fast` function.

---

### Playlist Similarity Based on Tracks

To compute the similarity between two playlists based on their tracks:

1. For each track in playlist **( P_1 )**, compute its similarity with **all tracks** in playlist **( P_2 )**.
2. Average these similarities to obtain a **per-track similarity score**.
3. Average the per-track scores across all tracks in **( P_1 )**.

Formally, the similarity between two playlists is defined as:

```math
SIM(P_1, P_2) =
\frac{1}{|P_1|}
\sum_{t \in P_1}
\left(
\frac{1}{|P_2|}
\sum_{s \in P_2} \text{sim}(t, s)
\right)
```

---

#### Implementation Details

* Pre-compute a dictionary mapping each playlist ID to its set of tracks
* Implement a `playlist_sim_fast` function using loops and list aggregation
* Optimized for repeated similarity calls

---

### Playlist Similarity Based on Name

Some challenge playlists contain **only a name**, requiring a text-based similarity metric.

#### Name Preprocessing

Playlist names were normalized by:

* converting to lowercase
* removing special characters
* removing stop words (English and Spanish)
* removing playlist-specific filler words (e.g. *playlist*, *mix*, *songs*)

#### Name Similarity Metric

A **hybrid similarity metric** was implemented by averaging:

1. **TF-IDF + Cosine Similarity**

   * Captures word overlap while down-weighting frequent terms
2. **Fuzzy String Matching**

   * Captures near-exact matches and is tolerant to word order and spelling variations

The final similarity score is the average of both methods.

This deterministic approach was chosen to balance **robustness and computational efficiency**.

---

## 3. Playlist Continuation Strategy

The challenge dataset presents multiple scenarios. The continuation strategy adapts accordingly.

### Case 1: Playlist with Seed Tracks

When seed tracks are available:

* Treat each candidate track as a **single-track playlist**
* Compute playlist similarity between the seed playlist and each candidate
* Rank tracks by similarity
* Remove:

  * duplicate tracks
  * seed tracks themselves
* Return the top **500 recommendations**

---

### Case 2: Playlist with Name Only

When only the playlist name is available, a **two-step hybrid approach** is used:

1. **Name-based seeding**

   * Compute name similarity between the target playlist and all training playlists
   * Select the top **N most similar playlists**
   * Extract the **most frequently occurring tracks** among them
   * Use these tracks as a synthetic seed playlist

2. **Track-based continuation**

   * Apply the standard track-based continuation approach using the generated seed tracks
   * Complete the recommendation list up to 500 tracks

This approach mitigates the lower reliability of name-only similarity by grounding recommendations in observed listening behavior.

---

## 📤 Output

For each playlist in the challenge dataset:

* A CSV file is generated containing:

  * artist name
  * track name
  * track URI
  * artist URI
  * similarity score
* Each file contains **exactly 500 recommended tracks**, ranked by relevance

