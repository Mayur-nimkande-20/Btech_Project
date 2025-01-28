# import pandas as pd
# from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

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

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     recommendations = []
# #     user_query = ''
# #     if request.method == 'POST':
# #         user_query = request.form.get('query', '')
# #         top_n = int(request.form.get('top_n', 5))
# #         recommendations = recommend_books(user_query, top_n)
# #     return render_template('index4.html', query=user_query, recommendations=recommendations)

# # if __name__ == '__main__':
# #     app.run(debug=True)


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     collab_recommendations = []
#     content_recommendations = []
#     book_title = ''

#     if request.method == 'POST':
#         if request.form.get('action') == 'collaborative':
#             mystery_books = books_df[books_df['Category'] == 'Mystery'].head(10) # Convert the filtered DataFrame to a dictionary with 'records' orientation 
#             collab_recommendations = mystery_books.to_dict(orient='records')
#             # collab_recommendations = books_df["Category"]  # Display top 10 mystery books

#         elif request.form.get('action') == 'content-based':
#             book_title = request.form.get('book_title', '')
#             top_n = int(request.form.get('top_n', 5))
#             content_recommendations = recommend_books(book_title, top_n)

#     return render_template(
#         'index4.html',
#         collab_recommendations=collab_recommendations,
#         content_recommendations=content_recommendations,
#         book_title=book_title
#     )


import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset and prepare the recommendation model
books_df = pd.read_csv("BooksDataset.csv")
# books_df.fillna("", inplace=True)
books_df.dropna(inplace=True)

# Combine features into a single column for content-based filtering
books_df['combined_features'] = books_df.apply(
    lambda row: f"'{row['Title']}' is written by {row['Authors']}. "
                f"It falls under the {row['Category']} genre. "
                f"Description: {row['Description']}", axis=1)

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'])

# Function to recommend books using content-based filtering
def recommend_books(input_query, top_n=5):
    query_vector = vectorizer.transform([input_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = books_df.iloc[similar_indices][['Title', 'Authors', 'Category', 'Description']]
    return recommendations.to_dict(orient='records')

# Define the main route
@app.route('/', methods=['GET', 'POST'])
def index():
    collab_recommendations = []
    content_recommendations = []
    book_title = ''

    if request.method == 'POST':
        # Handle collaborative filtering for the Mystery genre
        if request.form.get('action') == 'collaborative':
            mystery_books = books_df[books_df['Category'].str.contains('Mystery', case=False)].head(10)
            collab_recommendations = mystery_books.to_dict(orient='records')

        # Handle content-based filtering
        elif request.form.get('action') == 'content-based':
            book_title = request.form.get('book_title', '')
            top_n = int(request.form.get('top_n', 5))
            content_recommendations = recommend_books(book_title, top_n)

    # Render the template with recommendations
    return render_template(
        'index4.html',
        collab_recommendations=collab_recommendations,
        content_recommendations=content_recommendations,
        book_title=book_title
    )

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
