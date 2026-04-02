import pandas as pd
#Loading Dataset
data = pd.read_csv('movie_data.csv')
df = pd.DataFrame(data)
print(df.head())
print(df.shape)
print("------------------------------")

from sklearn.feature_extraction.text import CountVectorizer
#Converting into vectors
cv = CountVectorizer()
vec = cv.fit_transform(df['Genre']).toarray()
#print(vec)
#print(vec.shape)

from sklearn.metrics.pairwise import cosine_similarity
#Finding the similarity probability b/w movies
similarity = cosine_similarity(vec)
#print(similarity)

"""
df[df['Title'] == 'Inception']
dist = sorted(list(enumerate(similarity[0])), reverse = True, key = lambda vec : vec[1])
print(dist)

for i in dist[0:5]:
    print(df.iloc[i[0]].Title)
"""
#Recommendation Function
def recommend(movie):
    print(f"Since You've Watched {movie}, You May also like: ")
    index = df[df['Title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse = True, key = lambda vec : vec[1])
    for i in distance[1:6]:
        print(df.iloc[i[0]].Title)
    print("==================================")

recommend('Titanic')
recommend('Iron Man')