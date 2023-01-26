from flask import Flask, render_template, request
import pickle
import sklearn
import pandas

app = Flask(__name__)

# load model, movie pivot table
loaded_model = pickle.load(open('model.pkl', 'rb'))
loaded_movies = pickle.load(open('movie_pivot.pkl', 'rb'))

# all movies
movie_titles = [loaded_movies.index[i] for i in range(0, len(loaded_movies))]


@app.route('/', methods=['GET'])
def start():
    return render_template('home.html', movie_titles=movie_titles)


@app.route('/', methods=['POST'])
def predict_movies():
    selected_movie = request.form['my_select']
    seleted_index = movie_titles.index(selected_movie)

    # predict
    distances, indices = loaded_model.kneighbors(loaded_movies.iloc[seleted_index, :].values.reshape(1, -1), n_neighbors=6)
    recommendations = [loaded_movies.index[indices.flatten()[i]] for i in range(1, len(distances.flatten()))]

    return render_template('result.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(port=3000, debug=True)