import os
import sys
sys.path.append('../../')

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import joblib
import pickle

import core.config as conf

class XGBoost:
    def __init__(self):
        self.dataset_path = conf.dataset_path
        self.model_path = conf.model_path
        self.model_name = conf.model_name
        self.language_lst = ['en-CA', 'en-GB', 'en-US', 'eng', 'fre', 'spa']
        
        self.TARGET = 'binary_rating'
        self.xgb_parms = { 
                'max_depth':8, 
                'learning_rate':0.025, 
                'subsample':0.85,
                'colsample_bytree':0.35, 
                'eval_metric':'logloss',
                'objective':'binary:logistic',
                #'tree_method':'gpu_hist',
                #'predictor': 'gpu_predictor',
                'seed': 1,
            }
    
    def language_encoding(self, df, train) :
        # language processing (One Hot Encoder)
        if train :
            one_hot = OneHotEncoder()
            encoded = one_hot.fit_transform(df[['language_code']])
            with open(conf.prep_path + "one_hot_encoder", "wb") as f: 
                pickle.dump(one_hot, f)
            with open(conf.prep_path + "language_encoded", "wb") as f: 
                pickle.dump(encoded, f)    
        else :
            with open(conf.prep_path + "one_hot_encoder", 'rb') as f :
                one_hot = pickle.load(f)
            encoded = one_hot.transform(df[['language_code']])
        df[one_hot.categories_[0]] = encoded.toarray()
        df = df.drop(conf.dont_used_features, axis = 1)
        
        return df
    
    def preprocessing(self, df, train) :
        book_df = pd.read_csv(self.dataset_path+'books.csv')
        df = pd.merge(book_df, df, how='right', on='book_id')
        
        # user average rating
        user_info = df['binary_rating'].groupby(df['user_id']).mean().to_frame()
        user_info = user_info.reset_index()
        user_info.columns = ["user_id", "user_mean_rating"]
        df = pd.merge(df, user_info, how='left', on = 'user_id')
        read_book_lst = df.groupby('user_id')['book_id'].apply(list)
        user_info = pd.merge(user_info, read_book_lst, on='user_id')
        user_info.to_csv(self.dataset_path +'user_info.csv', index = False)
        del user_info
        
        '''
        # book average rating
        book_rating = df[df['binary_rating'] == 1].groupby(df['book_id']).count()['binary_rating'].to_frame()
        book_rating['total_no_rating'] = df[df['binary_rating'] == 0].groupby(df['book_id']).count()['binary_rating'].to_frame()
        book_rating['aveage_rating'] = df['binary_rating'].groupby(df['book_id']).mean().to_frame()
        book_rating.columns = ["total_rating", "total_no_rating", "total_average_rating"]
        book_rating.to_csv(self.dataset_path +'book_ratings_info.csv')
        df = pd.merge(df, book_rating, how = 'left', on = 'book_id')
        del book_df, book_rating
        
        # authors info
        authors_info = df['binary_rating'].groupby(df['authors']).mean().to_frame()
        authors_info = authors_info.reset_index()
        authors_info.columns = ["authors", "authors_mean_rating"]
        authors_info.to_csv(self.dataset_path +'authors_info.csv', index = False)
        df = pd.merge(df, authors_info, how = 'left', on = 'authors')
        del authors_info
        '''
        
        df = df.dropna()
        # language processing (One Hot Encoder)
        df = self.language_encoding(df, train)

        return df
    
    def train(self):
        df = pd.read_csv(self.dataset_path + conf.train_dataset)
        df = self.preprocessing(df, True)
        
        col_x = df.columns
        col_x = col_x.drop(self.TARGET)
        
        dtrain = xgb.DMatrix(data=df[col_x], label = df[self.TARGET])
        
        model = xgb.train(self.xgb_parms,
                          dtrain=dtrain,
                          num_boost_round=500,
                         )
        
        joblib.dump(model, self.model_path+self.model_name)
        del dtrain, model, df

    def predict(self):
        df = pd.read_csv(self.dataset_path + conf.test_dataset)
        df = self.preprocessing(df, False)
        
        col_x = df.columns
        col_x = col_x.drop(self.TARGET)
        y_true = df[self.TARGET]
        
        dvalid = xgb.DMatrix(data=df[col_x])
        
        model = joblib.load(self.model_path+self.model_name)
        
        y_pred = model.predict(dvalid)
        y_true = df[self.TARGET]
        
        rmse = mean_squared_error(y_true, y_pred)**0.5
        acc_score = accuracy_score(list(map(lambda x: 1 if x >= 0.5 else 0, y_pred)), y_true)
        
        print("*"*30)
        print("RMSE : ", rmse)
        print("Accuray Score : ", acc_score)
        print("*"*30)
        
        del dvalid, model, df
    
    def popular_recommend(self) :
        ratings_df = pd.read_csv(self.dataset_path+'binary_ratings.csv')
        book_df = pd.read_csv(conf.dataset_path + 'books.csv')
        book_ratings = ratings_df['binary_rating'].groupby(ratings_df['book_id']).sum().to_frame()
        book_ratings = pd.merge(book_ratings, book_df[['book_id', 'title']], on = 'book_id')
        rec_pop_top10 = book_ratings.sort_values(by=['binary_rating'], ascending=False, axis=0)['title'].values[:10]
        return rec_pop_top10
    
    def recommend(self, user_id) :
        book_df = pd.read_csv(self.dataset_path+'books.csv')
        book_df = book_df.drop(book_df[~book_df['language_code'].isin(self.language_lst)].index)
        book_df = book_df.dropna()
        
        rec_df = book_df
        
        rec_df['user_id'] = int(user_id)
        user_info = pd.read_csv(self.dataset_path + 'user_info.csv')
        #authors_info = pd.read_csv(self.dataset_path + 'authors_info.csv')
        
        if user_id not in user_info['user_id'].values : # top popular 10 books
            rec_top10 = self.popular_recommend()
            print(f"User {user_id} has no info..")
            
        else :
            rec_df['user_mean_rating'] = user_info[user_info['user_id'] == int(user_id)]['user_mean_rating'].values[0]

            user_read_books = user_info[user_info['user_id'] == int(user_id)]['book_id']
            rec_df = rec_df.drop(rec_df[rec_df['book_id'].isin(user_read_books)].index)
            #book_ratings = pd.read_csv(self.dataset_path + 'book_ratings_info.csv')
            #rec_df = pd.merge(rec_df, book_ratings, on = 'book_id')
            #rec_df = pd.merge(rec_df, authors_info, on = 'authors')
            rec_df = self.language_encoding(rec_df, False)
            
            rec_df = rec_df.dropna()
        
            # recommend(predict)
            dpred = xgb.DMatrix(data = rec_df)
            model = joblib.load(conf.model_path+conf.model_name)
            y_pred = model.predict(dpred)

            recommend_lst = pd.DataFrame(rec_df['book_id'])
            recommend_lst = pd.merge(recommend_lst, book_df[['book_id', 'title']], on= 'book_id')
            recommend_lst['pred'] = y_pred

            rec_top10 = recommend_lst.sort_values(by=['pred'], ascending=False, axis=0)['title'].values[:10]

        print("*"*30)
        for idx, book in enumerate(rec_top10) :
            print(f"rank {idx+1} : ", book)
        print("*"*30)
        
        del book_df
        return