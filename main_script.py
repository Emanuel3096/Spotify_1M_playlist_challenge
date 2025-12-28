import pandas as pd
import numpy as np
import json
import glob
from thefuzz import fuzz
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


####### UPLOADING OUR TRAINING DATASET ########

#1. Uploading our subeset of dat 
playlist_aux_1=[]
tracks_aux_1 =[]

for file in glob.glob('dataset/10kdata/*.json'): #Chhange the path according to your local

    with open(file) as f:
        data = json.load(f)


    for p in data['playlists']:
        playlist_aux_1.append(p)
        for t in p['tracks']:
            tracks_aux_1.append({**t, "pid": p["pid"]})



#2. Saving our traning set on these two dataframes
playlists = pd.DataFrame(playlist_aux_1)
tracks = pd.DataFrame(tracks_aux_1)

####### UPLOADING CHALLENGE DATASET ########

#1. Uploading now the challenge dataset

playlist_aux_2=[]
track_aux_2 =[]

for file in glob.glob('challenge/challenge_set.json'):


    with open(file) as f:
        data = json.load(f)

    for p in data['playlists']:
        playlist_aux_2.append(p)
        for t in p['tracks']:
            track_aux_2.append({**t, "pid": p["pid"]})

#2. Saving our traning set on these two dataframes
chal_playlists = pd.DataFrame(playlist_aux_2)
chal_tracks = pd.DataFrame(track_aux_2)


######## PREPOCESSING ACTIVITY: NORMALIZE NAME COLUMN ########

# 1. Remove stop words, special characters and convert all into lower case

spot_words = ["i", "me","mi","mi","mío","nuestro","nuestra","my","mine", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"] 
spot_words_es = ["yo", "me", "mi", "mío", "mía", "míos", "mías", "conmigo","nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras","tú", "te", "ti", "tuyo", "tuya", "tuyos", "tuyas","usted", "ustedes","él", "ella", "ello", "ellos", "ellas","lo", "la", "los", "las", "le", "les", "se", "sí", "consigo","el", "la", "los", "las", "un", "una", "unos", "unas","ser", "es", "soy", "eres", "somos", "son", "era", "eran", "fui", "fue", "sido","estar", "estoy", "estás", "está", "estamos", "están", "estaba", "estaban","haber", "he", "has", "ha", "hemos", "han", "había", "habían","tener", "tengo", "tienes", "tiene", "tenemos", "tienen","hacer", "hago", "hace", "hacen", "hacerlo","y", "o", "pero", "sino", "porque", "como", "aunque", "si", "mientras", "cuando","donde", "que", "quien", "quienes", "cual", "cuales","a", "ante", "bajo", "con", "contra", "de", "desde", "durante","en", "entre", "hacia", "hasta", "mediante", "para", "por", "según","sin", "sobre", "tras","no", "sí", "ya", "aquí", "ahí", "allí", "ahora", "antes", "después","muy", "más", "menos", "mucho", "poco", "también", "tampoco","siempre", "nunca", "todo", "nada", "algo", "alguien","solo", "solamente", "mismo", "misma", "mismos", "mismas","otro", "otra", "otros", "otras"]
all_spot_words = set(spot_words) | set(spot_words_es)

#For playlists:

normalized =playlists["name"].fillna("").astype(str).str.lower()
normalized = normalized.str.replace(r"[^A-Za-z0-9 ]", " ", regex=True)
normalized = normalized.str.split().apply(lambda words: " ".join(w for w in words if w not in all_spot_words))
normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()

# 2. Remove "filler words" when it comes to playlist lingo

filler_words = ["new","playlist","songs","music","mix","stuff","time","tunes","1","2", "3", "one","two", "top","list","lists","favs","favorites","favorite","musica","gold","classics","classic"]
filler_words_es = ["lista", "listas", "canciones", "temas", "temazos","musica", "música", "sonidos", "audio", "tracks", "track","cosas", "cositas", "varios", "varias", "mix", "mezcla", "todo","uno", "dos", "tres""favoritos", "favoritas", "favorito", "favorita", "nuevo", "nueva","bueno", "buena", "buenos", "buenas","oro"]
all_filler_words = set(filler_words) | set(filler_words_es)


normalized = normalized.str.split().apply(lambda words: " ".join(w for w in words if w not in all_filler_words))
playlists["normalize_name"] = normalized


#For challenge playlist as well:

normalized = chal_playlists["name"].fillna("").astype(str).str.lower()
normalized = normalized.str.replace(r"[^A-Za-z0-9 ]", " ", regex=True)
normalized = normalized.str.split().apply(lambda words: " ".join(w for w in words if w not in all_spot_words))
normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()

filler_words = ["la","playlist","songs","music","mix","stuff","time","tunes","1","2", "3", "one","two", "top","list","lists","favs","favorites","favorite","tbt","musica","gold"]
normalized = normalized.str.split().apply(lambda words: " ".join(w for w in words if w not in all_filler_words))
chal_playlists["normalize_name"] = normalized

################# Slicing the current challenge dataset not to do everything


######## FUNCTION FOR TRACK SIMILARITY ########

# Performed a similarity between two tracks based on how many playlists they both show up. Used a cosine similarity. 

# Pre-Compute sets to improve the computational time

pids_by_track = (tracks.groupby("track_uri")["pid"].apply(set).to_dict())
track_deg = {uri: len(pids) for uri, pids in pids_by_track.items()}

def track_sim_fast(a, b):
    A = pids_by_track.get(a)    #Use get to return none insteand of an error
    B = pids_by_track.get(b)    #Use get to return none insteand of an error
    if A is None or B is None:  #If none then the similarity is zero since they don't share any playlists
        return 0
    return len(A & B) / (track_deg[a] * track_deg[b])**0.5


######## FUNCTION FOR PLAYLIST SIMILARITY BASED ON TRACKS ########

# Performed a similarity between two playlists based on the track similarity, by computing the averages of similarity between tracks of the two playlists, by the following formula:

# Pre-Compute sets to improve the computational time

tracks_by_pid = (tracks.groupby("pid")["track_uri"].apply(set).to_dict())

def playlist_sim_fast(main, second):
    second = list(second)
    sim_vector=[]
    for m in main:
        vals = [track_sim_fast(m, s) for s in second]
        sim_vector.append(sum(vals) / len(vals))  
    return sum(sim_vector) / len(sim_vector)


######## FUNCTION FOR PLAYLIST SIMILARITY BASED ON NAME ########

# Performed name similarity based on two different methods and average the results of the two, to mitigate each other's highs and lows

# Precompute TF IDF model based on our data

tfidf = TfidfVectorizer().fit(playlists["normalize_name"])

def playlist_name_similarity(main_name, name2,tfidf_vectorizer):
    
    # First method:
    q = tfidf_vectorizer.transform([name2])
    t = tfidf_vectorizer.transform([main_name])
    sim_1 = cosine_similarity(q, t)[0,0]

    # Second method:
    sim_2 = fuzz.token_set_ratio(main_name,name2)/100

    return (sim_1+sim_2)/2  


# Run our cycle for each playlist in the challenge playlist

for pid in chal_playlists["pid"]:
    
    # Define the name and the seed tracks from the playlist we are in

    name = chal_playlists.loc[chal_playlists.pid == pid, "normalize_name"].iloc[0]
    track_set = pd.Series(chal_tracks["track_uri"].loc[chal_tracks.pid == pid])

    # Values we are using for the name similarity approach that can be tuned

    number_of_playlist_in_name = 100
    top_tracks_in_name = 50

    #If track set is empty then apply the hybrid approach: name to get seed tracks then playlist similarity

    if len(track_set) == 0:

        playlists["Similarity"] = playlists["normalize_name"].apply(lambda x: playlist_name_similarity(name,x,tfidf))
        top_playlists = playlists.nlargest(number_of_playlist_in_name, "Similarity")                
        pids = top_playlists["pid"]

        # 2. Top tracks shared among the top playlists

        tracks_in_pdis = tracks[tracks["pid"].isin(pids)]
        counts = tracks_in_pdis.groupby("track_uri")["pid"].nunique()
        top50 = counts.nlargest(top_tracks_in_name).index.tolist()
        suggestion_50 = (tracks[tracks["track_uri"].isin(top50)].drop_duplicates("track_uri").assign(Similarity=1.0))

        # 2.1 Order these based on the top 50 order

        suggestion_50["track_uri"] = pd.Categorical(suggestion_50["track_uri"], categories=top50, ordered=True)
        suggestion_50 = suggestion_50.sort_values("track_uri")
        suggestion_50 = suggestion_50[["artist_name", "track_name", "track_uri", "artist_uri", "Similarity"]]

        # 3. Use the top as seed and apply the common approach like if these were 

        sim_series = tracks["track_uri"].apply(lambda x: playlist_sim_fast([x], top50))

        suggestions_450 = tracks.assign(Similarity=sim_series)
        suggestions_450 = suggestions_450[~suggestions_450["track_uri"].isin(top50)]
        suggestions_450 = suggestions_450.drop_duplicates(subset="track_uri")
        suggestions_450 = suggestions_450[["artist_name","track_name","track_uri","artist_uri","Similarity"]].sort_values(by="Similarity", ascending=False).head(500-top_tracks_in_name)

        #Merge them together
        suggestions_500= pd.concat([suggestion_50,suggestions_450], ignore_index=True)
        suggestions_500.to_csv(f"{pid}_suggestions.csv", index=False)   #Export our 500 suggestions with the PID as identifier
    
    # Else, if we have tracks then we apply our approach with tracks and playlist similarity function:
    
    else:    
        sim_series = tracks["track_uri"].apply(lambda x: playlist_sim_fast([x], track_set))
        suggestions = tracks.assign(Similarity=sim_series)
        suggestions = suggestions[~suggestions["track_uri"].isin(track_set)]
        suggestions = suggestions.drop_duplicates(subset="track_uri")
        suggestions = suggestions[["artist_name","track_name","track_uri","artist_uri","Similarity"]].sort_values(by="Similarity", ascending=False).head(500)
        suggestions.to_csv(f"{pid}_suggestions.csv", index=False)       #Export our 500 suggestions with the PID as identifier 
    
