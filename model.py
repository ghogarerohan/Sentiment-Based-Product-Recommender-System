# ML model for deployment
# importing require libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



class Recommendation_engine:
    
    def __init__(self):
        self.data = pd.read_pickle(open('pickle/processed_data.pkl','rb'))
        self.user_final_rating = pd.read_pickle(open('pickle/user_final_rating.pkl','rb'))
        self.model = pd.read_pickle(open('pickle/logreg_model.pkl','rb'))
        self.raw_data = pd.read_csv("data/sample30.csv")
        self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)
        
        
    def Get_top_5_products(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        features = pickle.load(open('pickle/tfidf_vocab.pkl','rb'))
        vectorizer = TfidfVectorizer(vocabulary = features)
        temp=self.data[self.data.id.isin(items)]
        X = vectorizer.fit_transform(temp['reviews_combine'])
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Positive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id','name', 'brand','categories', 'manufacturer']].drop_duplicates().to_html(index=False)
    
        
        

