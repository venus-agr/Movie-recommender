from sklearn.feature_extraction.text import CountVectorizer # it is a class
from sklearn.metrics.pairwise import cosine_similarity  # method not class so need for intialization

text = ["London Paris London","Paris Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
# print (count_matrix.toarray()) 
# to array converts the points into proper vector then sparse matrix

similarity_scores = cosine_similarity(count_matrix.toarray()) # using count_matrix.toarray() or count_matrix will give same result

print (similarity_scores)

