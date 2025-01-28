# from flask import Flask, request, render_template
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import csr_matrix
# import chardet


# app = Flask(__name__)

# # Load datasets
# books_filepath = "./books_data/books.csv"
# ratings_filepath = "./books_data/ratings.csv"
# users_filepath = "./books_data/users.csv"

# # books_df = pd.read_csv(books_filepath ,  encoding='latin1')
# # with open("./books_data/books.csv", 'rb') as f:
# #     result = chardet.detect(f.read())
# #     print(result)
# books_df = pd.read_csv(books_filepath, encoding='utf-8')
# ratings_df = pd.read_csv(ratings_filepath ,  encoding='utf-8')
# users_df = pd.read_csv(users_filepath,  encoding='utf-8')

# # Preprocess books and ratings datasets
# books_df['ISBN'] = books_df['ISBN'].astype(str)
# ratings_df['ISBN'] = ratings_df['ISBN'].astype(str)

# # Merge datasets for analysis
# ratings_with_books = ratings_df.merge(books_df, on='ISBN')

# # Create a user-item matrix
# user_item_matrix = ratings_with_books.pivot_table(
#     index='User-ID',
#     columns='Book-Title',
#     values='Book-Rating'
# ).fillna(0)

# # Compute cosine similarity
# user_similarity = cosine_similarity(user_item_matrix)
# user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# # Recommendation function
# def recommend_books(user_id, top_n=5):
#     if user_id not in user_similarity_df.index:
#         return []

#     # Find similar users
#     similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).index[1:]

#     # Aggregate ratings from similar users
#     similar_users_ratings = user_item_matrix.loc[similar_users].mean(axis=0)

#     # Exclude books the user has already rated
#     user_ratings = user_item_matrix.loc[user_id]
#     recommendations = similar_users_ratings[user_ratings == 0].sort_values(ascending=False).head(top_n)

#     # Fetch book details
#     recommended_books = books_df[books_df['Book-Title'].isin(recommendations.index)]
#     return recommended_books[['Book-Title', 'Book-Author', 'Publisher']].to_dict(orient='records')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     recommendations = []
#     user_id = ''
#     if request.method == 'POST':
#         user_id = request.form.get('user_id')
#         top_n = int(request.form.get('top_n', 5))
#         try:
#             user_id = int(user_id)
#             recommendations = recommend_books(user_id, top_n)
#         except ValueError:
#             pass
#     return render_template('index2.html', user_id=user_id, recommendations=recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import chardet

app = Flask(__name__)

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

# Load datasets with dynamic encoding detection
books_filepath = "./books_data/books.csv"
ratings_filepath = "./books_data/ratings.csv"
users_filepath = "./books_data/users.csv"

books_encoding = detect_encoding(books_filepath)
ratings_encoding = detect_encoding(ratings_filepath)
users_encoding = detect_encoding(users_filepath)

books_df = pd.read_csv(books_filepath, encoding=books_encoding)
ratings_df = pd.read_csv(ratings_filepath, encoding=ratings_encoding)
users_df = pd.read_csv(users_filepath, encoding=users_encoding)

# Preprocess books and ratings datasets
books_df['ISBN'] = books_df['ISBN'].astype(str)
ratings_df['ISBN'] = ratings_df['ISBN'].astype(str)

# Merge datasets for analysis
ratings_with_books = ratings_df.merge(books_df, on='ISBN')

# Create a user-item matrix
user_item_matrix = ratings_with_books.pivot_table(
    index='User-ID',
    columns='Book-Title',
    values='Book-Rating'
).fillna(0)

# Compute cosine similarity
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Recommendation function
def recommend_books(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return []

    # Find similar users
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).index[1:]

    # Aggregate ratings from similar users
    similar_users_ratings = user_item_matrix.loc[similar_users].mean(axis=0)

    # Exclude books the user has already rated
    user_ratings = user_item_matrix.loc[user_id]
    recommendations = similar_users_ratings[user_ratings == 0].sort_values(ascending=False).head(top_n)

    # Fetch book details
    recommended_books = books_df[books_df['Book-Title'].isin(recommendations.index)]
    return recommended_books[['Book-Title', 'Book-Author', 'Publisher']].to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user_id = ''
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        top_n = int(request.form.get('top_n', 5))
        try:
            user_id = int(user_id)
            recommendations = recommend_books(user_id, top_n)
        except ValueError:
            pass
    return render_template('index2.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
