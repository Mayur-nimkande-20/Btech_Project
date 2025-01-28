# # from flask import Flask, request, jsonify
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # import pandas as pd

# # app = Flask(__name__)

# # # Load dataset and prepare recommendation model
# # books_df = pd.read_csv("BooksDataset.csv")
# # books_df.fillna("", inplace=True)
# # books_df['combined_features'] = books_df.apply(
# #     lambda row: f"'{row['Title']}' is written by {row['Authors']}. "
# #                 f"It falls under the {row['Category']} genre. "
# #                 f"Description: {row['Description']}", axis=1)

# # vectorizer = TfidfVectorizer(stop_words='english')
# # tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'])

# # def recommend_books(input_query, top_n=5):
# #     query_vector = vectorizer.transform([input_query])
# #     similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
# #     similar_indices = similarity_scores.argsort()[-top_n:][::-1]
# #     recommendations = books_df.iloc[similar_indices][['Title', 'Authors', 'Category', 'Description']]
# #     return recommendations.to_dict(orient='records')

# # @app.route('/recommend', methods=['POST'])
# # def recommend():
# #     data = request.get_json()
# #     user_query = data.get('query', '')
# #     top_n = int(data.get('top_n', 5))
# #     recommendations = recommend_books(user_query, top_n)
# #     return jsonify(recommendations)

# # if __name__ == '__main__':
# #     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# app = Flask(__name__)

# # Load dataset and prepare recommendation model
# books_df = pd.read_csv("BooksDataset.csv")
# books_df.fillna("", inplace=True)
# books_df['combined_features'] = books_df.apply(
#     lambda row: f"'{row['Title']}' is written by {row['Authors']}. "
#                 f"It falls under the {row['Category']} genre. "
#                 f"Description: {row['Description']}", axis=1)

# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'])

# def recommend_books(input_query, top_n=5):
#     query_vector = vectorizer.transform([input_query])
#     similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     similar_indices = similarity_scores.argsort()[-top_n:][::-1]
#     recommendations = books_df.iloc[similar_indices][['Title', 'Authors', 'Category', 'Description']]
#     return recommendations.to_dict(orient='records')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     user_query = request.form.get('query', '')
#     top_n = int(request.form.get('top_n', 5))
#     recommendations = recommend_books(user_query, top_n)
#     return render_template('recommendations.html', query=user_query, recommendations=recommendations)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load dataset and prepare recommendation model
books_df = pd.read_csv("BooksDataset.csv")
books_df.fillna("", inplace=True)
books_df['combined_features'] = books_df.apply(
    lambda row: f"'{row['Title']}' is written by {row['Authors']}. "
                f"It falls under the {row['Category']} genre. "
                f"Description: {row['Description']}", axis=1)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'])

def recommend_books(input_query, top_n=5):
    query_vector = vectorizer.transform([input_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = books_df.iloc[similar_indices][['Title', 'Authors', 'Category', 'Description']]
    return recommendations.to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    user_query = ''
    if request.method == 'POST':
        user_query = request.form.get('query', '')
        top_n = int(request.form.get('top_n', 5))
        recommendations = recommend_books(user_query, top_n)
    return render_template('index.html', query=user_query, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
