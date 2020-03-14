from flask import Flask, request, g, json
import sqlite3
from flask_cors import CORS
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.compose import ColumnTransformer
app = Flask(__name__)
CORS(app)

DATABASE = 'db.sqlite3'


def get_db():
  db = getattr(g, '_database', None)
  if db is None:
    db = g._database = sqlite3.connect(DATABASE)
  return db


@app.teardown_appcontext
def close_connection(exception):
  db = getattr(g, '_database', None)
  if db is not None:
    db.close()


def query_db(query, args=(), one=False):
  cur = get_db().execute(query, args)
  rv = cur.fetchall()
  cur.close()
  return (rv[0] if rv else None) if one else rv


def train():
  query = 'SELECT * FROM sneakers'
  all_sneakers = query_db(query, (), False)
  # Make all sneakers a list of lists
  all_sneakers = [list(row) for row in all_sneakers]
  for sneaker in all_sneakers:
    X, y = zip(*ast.literal_eval(sneaker[7]))
    sneaker[7] = list(X)
    sneaker.append(list(y))

  print(all_sneakers[:2])

  df = pd.DataFrame(data=all_sneakers)
  X = df.iloc[:, [3, 5, 7]]
  y = df.iloc[:, 8]
  # X = np.array(X)
  print(X.iloc[0])
  X.columns = ['brand', 'color', 'history']
  encoder = OneHotEncoder()
  brands_colors = encoder.fit_transform(X.iloc[:, [0, 1]])
  X.iloc[:, [0,1]].assign(brands_colors)
  print(X)
  # Turn Brand and color features into numerical categories
  # encoder = OneHotEncoder()
  # columnTransformer = ColumnTransformer([('encoder', encoder, [0, 1])], remainder='passthrough')
  # transform = columnTransformer.fit_transform(X)
  # X = np.array(transform)
  # print(X[0])
  # print(X[0][1])
  # print(type(X[0][1]))
  # model = LinearRegression()  # Create an instance of the estimator
  # model.fit(X, y)  # Fit the model on the training data ”

  #
  #
  # sneaker1 = all_sneakers[0]
  # print(type(sneaker1[7]))
  # sneaker1_chart = ast.literal_eval(sneaker1[7])
  # print(sneaker1_chart)
  # print(type(sneaker1_chart[0]))
  # X, y = zip(*sneaker1_chart)
  # X = list(X)
  # y = list(y)
  # X = np.array(X).reshape(-1, 1)
  # print(X)
  # print(y)
  # model = LinearRegression()  # Create an instance of the estimator
  # model.fit(X, y)  # Fit the model on the training data ”
  # #Unseen timestamp
  # test_time = np.array([[1579302500000], [1580302600000]])
  # predicted_price = model.predict(test_time)
  # print(predicted_price)


@app.route('/')
def hello_world():
  train()
  return 'Hello World!'


@app.route('/graph')
def get_graph():
  sku = request.args.get('sku')
  print(sku)
  query = 'SELECT chart FROM sneakers WHERE sku=\'' + sku + '\''
  print(query)
  results = query_db(query, (), True)
  print(results)
  return json.dumps({'chart': results}), 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
  app.run()
