# Music_Recommendation
A ml model to understand clustering algorithm in machine learning. This is based on content filtering. It will recommend similar songs on the basis of the song you wanna hear.
## Dataset
https://www.kaggle.com/datasets/rakkesharv/spotify-top-10000-streamed-songs🔗
Have added comments in the project to help you understand it one step at a time😁
```
# import requried dependencies
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# read the data
df = pd.read_csv("D:\song-dataset.csv", low_memory=False)[:1000]
# remove duplicates
df = df.drop_duplicates(subset="Song Name")

# drop Null values
df = df.dropna(axis=0)

# Drop the non-required columns
df = df.drop(df.columns[3:], axis=1)
# Removing space from "Artist Name" column
df["Artist Name"] = df["Artist Name"].str.replace(" ", "")
# Combine all columns and assgin as new column
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)
# models
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df["data"])
similarities = cosine_similarity(vectorized)

# Assgin the new dataframe with `similarities` values
df_tmp = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()
true = True
while true:
    print("The Top 10 Song Recommendation System")
    print("-------------------------------------")
    print("This will generate the 10 songs from the database thoese are similar to the song you entered.")

    # Asking the user for a song, it will loop until the song name is in our database.
    while True:
        input_song = input("Please enter the name of song: ")

        if input_song in df_tmp.columns:
            recommendation = df_tmp.nlargest(11, input_song)["Song Name"]
            break
        
        else:
            print("Sorry, there is no song name in our database. Please try another one.")
    
    print("You should check out these songs: \n")
    for song in recommendation.values[1:]:
        print(song)

    print("\n")
    # Asking the user for the next command, it will loop until the right command.
    while True:
        next_command = input("Do you want to generate again for the next song? [yes, no] ")

        if next_command == "yes":
            break

        elif next_command == "no":
            # `true` will be false. It will stop the whole script
            true = False
            break

        else:
            print("Please type 'yes' or 'no'")
```
## How actually the model works ?
1. Feature Engineering: The code combines the "Song Name" and "Artist Name" columns into a single "data" column. This allows the model to consider both song title and artist style when calculating similarities.
2. Vectorization: The code uses CountVectorizer to convert the text in the "data" column into numerical vectors. Each vector represents a song, where each element corresponds to the frequency of a word in the song title and artist name.
3. Cosine Similarity: It quantifies how similar two songs are based on their word frequencies. Songs with similar patterns and word usage get higher cosine similarity scores.
4.  Recommendation Engine: $ The system creates a similarity matrix (DataFrame) representing cosine similarity scores between all song pairs.
$ When the user provides a song name, the system retrieves its corresponding vector from the matrix.
$ The system then finds the 10 songs with the highest cosine similarity scores to the user's chosen song, excluding the song itself. These are the recommended songs.
   
