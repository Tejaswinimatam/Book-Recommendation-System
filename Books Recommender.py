# %%
"""
# Books Recommender system using clustering
Collaborative filtering
- Dataset :- https://www.kaggle.com/ra4u12/bookrecommendation
"""

# %%
# Importing necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
books = pd.read_csv(r"C:\Users\ashut\Downloads\BRS\data\Books.csv")

# %%
books.head()

# %%
books.iloc[237]['Image-URL-L']

# %%
# !curl "http://images.amazon.com/images/P/0195153448.01.THUMBZZZ.jpg" --out.png
# !curl http://images.amazon.com/images/P/0060973129.01.THUMBZZZ.jpg --output some.jpg

# %%
books.shape

# %%
books.columns

# %%
"""
#### Conclution:
Here Image URL columns is important for the poster. So, we will keep it
"""

# %%
books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher','Image-URL-L']]

# %%
books.head()

# %%
# Lets remane some wierd columns name
books.rename(columns={"Book-Title":'title',
                      'Book-Author':'author',
                     "Year-Of-Publication":'year',
                     "Publisher":"publisher",
                     "Image-URL-L":"image_url"},inplace=True)

# %%
books.head()

# %%
# Now load the second dataframe

users = pd.read_csv(r"C:\Users\ashut\Downloads\BRS\data\Users.csv")

# %%
users.head()

# %%
users.shape

# %%
# Lets remane some wierd columns name
users.rename(columns={"User-ID":'user_id',
                      'Location':'location',
                     "Age":'age'},inplace=True)

# %%
users.head(2)

# %%
# Now load the third dataframe

ratings = pd.read_csv(r"C:\Users\ashut\Downloads\BRS\data\Ratings.csv")

# %%
ratings.head()

# %%
ratings.shape

# %%
# Lets remane some wierd columns name
ratings.rename(columns={"User-ID":'user_id',
                      'Book-Rating':'rating'},inplace=True)

# %%
ratings.head(2)

# %%
"""
### Conclution:
Now we have 3 dataframes
- books
- users
- ratings
"""

# %%
print(books.shape, users.shape, ratings.shape, sep='\n')



# %%
ratings['user_id'].value_counts()

# %%
ratings['user_id'].value_counts().shape

# %%
ratings['user_id'].unique().shape

# %%
# Lets store users who had at least rated more than 200 books
x = ratings['user_id'].value_counts() > 200

# %%
x[x].shape

# %%
y= x[x].index

# %%
y

# %%
ratings = ratings[ratings['user_id'].isin(y)]

# %%
ratings.head()

# %%
ratings.shape

# %%
# Now join ratings with books

ratings_with_books = ratings.merge(books, on='ISBN')

# %%
ratings_with_books.head()

# %%
ratings_with_books.shape

# %%
number_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()

# %%
number_rating.head()

# %%
number_rating.rename(columns={'rating':'num_of_rating'},inplace=True)

# %%
number_rating.head()

# %%
final_rating = ratings_with_books.merge(number_rating, on='title')

# %%
final_rating.head()

# %%
final_rating.shape

# %%
# Lets take those books which got at least 50 rating of user

final_rating = final_rating[final_rating['num_of_rating'] >= 50]

# %%
final_rating.head()

# %%
final_rating.shape

# %%
# lets drop the duplicates
final_rating.drop_duplicates(['user_id','title'],inplace=True)

# %%
final_rating.shape

# %%
# Lets create a pivot table
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values= 'rating')

# %%
book_pivot

# %%
book_pivot.shape

# %%
book_pivot.fillna(0, inplace=True)

# %%
book_pivot

# %%
"""
# Training Model
"""

# %%
from scipy.sparse import csr_matrix

# %%
book_sparse = csr_matrix(book_pivot)

# %%
type(book_sparse)

# %%
# Now import our clustering algoritm which is Nearest Neighbors this is an unsupervised ml algo
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm= 'brute')

# %%
model.fit(book_sparse)

# %%
distance, suggestion = model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1), n_neighbors=6 )

# %%
distance

# %%
suggestion

# %%
book_pivot.iloc[241,:]

# %%
for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])

# %%
book_pivot.index[3]

# %%
#keeping books name
book_names = book_pivot.index

# %%
book_names[2]

# %%
np.where(book_pivot.index == '4 Blondes')[0][0]

# %%
"""
# find url
"""

# %%
# final_rating['title'].value_counts()
ids = np.where(final_rating['title'] == "Harry Potter and the Chamber of Secrets (Book 2)")[0][0]

# %%
final_rating.iloc[ids]['image_url']

# %%
book_name = []
for book_id in suggestion:
    book_name.append(book_pivot.index[book_id])
    
    

# %%
book_name[0]

# %%
ids_index = []
for name in book_name[0]: 
    ids = np.where(final_rating['title'] == name)[0][0]
    ids_index.append(ids)

# %%
for idx in ids_index:
    url = final_rating.iloc[idx]['image_url']
    print(url)

# %%
import pickle
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(book_names,open('book_names.pkl','wb'))
pickle.dump(final_rating,open('final_rating.pkl','wb'))
pickle.dump(book_pivot,open('book_pivot.pkl','wb'))

# %%
"""
# Testing model
"""

# %%
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                if j == book_name:
                    print(f"You searched '{book_name}'\n")
                    print("The suggestion books are: \n")
                else:
                    print(j)

# %%
book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
recommend_book(book_name)

# %%
pip install ipynb-py-convert


# %%
