dataset_path = '/home/nyongja/kie/goodbooks-10k-recommender/dataset/'
model_path = '/home/nyongja/kie/goodbooks-10k-recommender//models/model/'
model_name = 'xgboost-model-0.xgb'
prep_path = '/home/nyongja/kie/goodbooks-10k-recommender/prep_pickle/'
train_dataset = 'binary_ratings_train.csv'
test_dataset = 'binary_ratings_test.csv'

used_features = ['book_id', 'books_count', 'authors', 'original_publication_year', 'language_code', 'user_id', 'rating']

dont_used_features = ['id', 'work_id', 'best_book_id', 'authors', 'isbn', 'isbn13', 'original_title', 'title',
                      'work_ratings_count', 'language_code', 'image_url', 'small_image_url', 
                      'ratings_1','ratings_2', 'ratings_3', 'ratings_4', 'ratings_5', 'work_text_reviews_count', 'average_rating',
                      'ratings_count']