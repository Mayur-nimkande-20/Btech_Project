from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import chardet

app = Flask(__name__)

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

# Load datasets
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

# Collaborative: Static "Mystery" genre book list
mystery_books = books_df[books_df['Genre'] == 'Mystery'][['Book-Title', 'Book-Author', 'Publisher']].to_dict(orient='records')

# Content-Based Recommendation Function
def content_based_recommendations(book_title, top_n=5):
    books_df['combined_features'] = books_df['Book-Title'] + " " + books_df['Book-Author'] + " " + books_df['Publisher']
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books_df['combined_features'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(books_df.index, index=books_df['Book-Title']).drop_duplicates()

    if book_title not in indices:
        return []

    idx = indices[book_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    book_indices = [i[0] for i in sim_scores]
    return books_df.iloc[book_indices][['Book-Title', 'Book-Author', 'Publisher']].to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    collab_recommendations = []
    content_recommendations = []
    book_title = ''

    if request.method == 'POST':
        if request.form.get('action') == 'collaborative':
            collab_recommendations = mystery_books

        elif request.form.get('action') == 'content-based':
            book_title = request.form.get('book_title', '')
            top_n = int(request.form.get('top_n', 5))
            content_recommendations = content_based_recommendations(book_title, top_n)

    return render_template(
        'index3.html',
        collab_recommendations=collab_recommendations,
        content_recommendations=content_recommendations,
        book_title=book_title
    )

if __name__ == '__main__':
    app.run(debug=True)
