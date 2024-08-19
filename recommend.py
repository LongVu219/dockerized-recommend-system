from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, KFold
import pandas as pd
import random as rd

num_user = 5000
num_item = 10665

new_user_id = 5000
new_user_item = 10
new_user = []

def generate_score():
    return rd.randint(0,10) / 2

def get_link_score(item_id):
    ans = item_link_df[item_link_df['ID'] == item_id]
    arr = ans['Link'].values
    return (arr[0], generate_score())

for i in range(0,new_user_item):
    rating = generate_score()
    item = rd.randint(0,500)
    new_user.append((new_user_id, item, rating))

new_user_df = pd.DataFrame(new_user, columns = ['userId', 'itemId', 'rating'])
#print(new_user_df)
#exit

home_path = '/app'

#path
path = home_path + '/data/user_per_item.csv'
reader = Reader()
ratings = pd.read_csv(path)
#ratings = ratings.drop('Unnamed: 0', axis = 0)
#print(ratings)
ratings = pd.concat([ratings, new_user_df])
ratings.reset_index(drop=True)
#print(ratings)

data = Dataset.load_from_df(ratings[['userId', 'itemId', 'rating']], reader)

# Define a cross-validation iterator
kf = KFold(n_splits = 20)
svd = SVD()
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=kf, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset)

predicted_rating = []

#path
path = home_path + '/data/id_link.csv'
item_link_df = pd.read_csv(path)

for i in range (0,num_item):
    item = i
    ratings = svd.predict(new_user_id, item)
    item_link, ratings = get_link_score(item)
    predicted_rating.append((ratings, item_link))

#print(predicted_rating)

sorted_result = sorted(predicted_rating, key=lambda x: x[0], reverse = True)

num5 = 0
for i in sorted_result:
    if i[0] == 5:
        num5 += 1

best = sorted_result[:num5]
rd.shuffle(best)

for tup in best[:20]:
    print(tup[1])

'''

docker build -t recomm .
docker run --name recommender_sys recomm

'''