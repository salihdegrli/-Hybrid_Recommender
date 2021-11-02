import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


#############################################
# Görev 1: Veri Hazırlama işlemlerini gerçekleştiriniz.
#############################################
def read_movieData():
    return pd.read_csv('HAFTA_04/1-Notes/movie_lens_dataset/movie.csv')


def read_ratingData():
    return pd.read_csv('HAFTA_04/1-Notes/movie_lens_dataset/rating.csv')


def create_user_movie_df():
    movies = read_movieData()
    ratings = read_ratingData()
    df = movies.merge(ratings, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movies_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movies_df


user_movie_df = create_user_movie_df()

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

#############################################
# Görev 2: Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.
#############################################

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)

#############################################
# Görev 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz
#############################################

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count.head()

perc = len(movies_watched) * 60 / 100
user_movie_count[user_movie_count["movie_count"] > perc].sort_values("movie_count", ascending=False)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#############################################
# Görev 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz
#############################################

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# random user = random userın izlediği filmler
# movies_watched_df= random user ile aynı filmi izleyen idler
# users_same_movies= random user ile 20'den fazla  ortak film izleyenler

final_df.head()

final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df.head()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]]. \
    reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users.head()

rating = read_ratingData()
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings.head(50)

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Görev 5: Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.head()

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df.head()

recommendation_df[["movieId"]].nunique()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5]. \
    sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('HAFTA_04/1-Notes/movie_lens_dataset/movie.csv')
movies_to_be_recommend = movies_to_be_recommend.merge(movie[["movieId", "title"]])

user_based_top5 = movies_to_be_recommend.loc[:5, "title"]

user_based_top5.head()

#############################################
# Görev 6: Kullanıcının izlediği filmlerden en son en yüksek puan verdiği  filmin adına göre item-based öneri yapınız
#############################################

rating = pd.read_csv('HAFTA_04/1-Notes/movie_lens_dataset/rating.csv')

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)]. \
               sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_name = movie[movie["movieId"] == movie_id]["title"][0:1].values[0]

movie_name = user_movie_df[movie_name]

item_based_top5 = user_movie_df.corrwith(movie_name).sort_values(ascending=False).iloc[1:6]
