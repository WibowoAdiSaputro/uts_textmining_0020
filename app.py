from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	filename = "reviewrestoransteak_tanpa_preprocessing (1).csv"
	df = pd.read_csv(filename, encoding = 'latin-1')

	df.drop(columns = ['name_restoran_steak','name'], inplace = True)
	df.columns = ['Rating', 'Review']
	import string
	import re
	def clean_Review(Review):
		return re.sub('[^a-zA-Z]', ' ', Review).lower()

	df['cleaned_Review'] = df['Review'].apply(lambda x: clean_Review(str(x)))
	df['label'] = df['Rating'].map({1.0:0, 2.0:0, 3.0:0, 4.0:1, 5.0:1})
	def count_punct(Review):
		count = sum([1 for char in Review if char in string.punctuation])
		return round(count/(len(Review) - Review.count(" ")), 3)*100
  
	df['Review_len'] = df['Review'].apply(lambda x: len(str(x)) - str(x).count(" "))
	df['punct'] = df['Review'].apply(lambda x: count_punct(str(x)))
	def tokenize_Review(Review):
		tokenized_Review = Review.split()
		return tokenized_Review
	df['tokens'] = df['cleaned_Review'].apply(lambda x: tokenize_Review(x))

	import nltk
	nltk.download('stopwords')
	nltk.download('wordnet')
	nltk.download('omw-1.4')
	from nltk.corpus import stopwords
	all_stopwords = stopwords.words('english')
	all_stopwords.remove('not')

	def lemmatize_Review(token_list):
		return " ".join([lemmatizer.lemmatize(token) for token in token_list if not token in set(all_stopwords)])
	lemmatizer = nltk.stem.WordNetLemmatizer()
	df['lemmatized_Review'] = df['tokens'].apply(lambda x: lemmatize_Review(x))

	X = df[['lemmatized_Review', 'Review_len', 'punct']]
	y = df['label']
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf = TfidfVectorizer(max_df = 0.5, min_df = 2) # ignore terms that occur in more than 50% documents and the ones that occur in less than 2
	tfidf_train = tfidf.fit_transform(X_train['lemmatized_Review'])
	tfidf_test = tfidf.transform(X_test['lemmatized_Review'])

	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer()
	X_cv = cv.fit_transform(df['lemmatized_Review']) # Fit the Data
	y_cv = df['label']

	from sklearn.model_selection import train_test_split
	X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_cv, y_cv, test_size=0.3, random_state=42)

	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()

	clf.fit(X_train_cv, y_train_cv)
	clf.score(X_test_cv, y_test_cv)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)