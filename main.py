#importing all the required libraries
from __future__ import division
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import math
import pandas as pd


#scraping data from the website using url
URL = "https://www.imdb.com/title/tt0944947/reviews/?ref_=tt_ql_urv"
html = request.urlopen(URL).read()
raw = BeautifulSoup(html,  features="lxml").get_text()



#extracting the unstructured scraped data into a file
raw_file = open("raw_data.txt","w+",encoding="utf-8")
raw_file.write(raw)
print("scraped raw data is extracted to the raw_data.txt file")


#reading data from the file and extracting only the user reviews from raw data
r=open("raw_data.txt","r",encoding="utf-8")
new_raw=r.read()
rawlist = new_raw.split("\n")
res = [i for i in rawlist if i != '']
List=[]
for i in range(len(res)):
    if res[i] == '                            Was this review helpful?  Sign in to vote.':
        List.append(res[i - 3])





#using Multinomial Naive Bayes algorithm for data matching
c=CountVectorizer()
c.fit(List)
c_tr=c.transform(List)

T_f=TfidfTransformer()
T_f.fit(c_tr)






#using nltk to find the least significant words(last 10% significant words) to remove it later
stoplist = stopwords.words("english")
s=list(T_f.idf_)
s.sort()
i=math.ceil(0.1*len(s))
val=s[i]
dict_items=c.vocabulary_.items()
count=0
for i in range(len(T_f.idf_)):
    if T_f.idf_[i]<val:
        stoplist.append([key for key,value in dict_items if value==i])
        count+=1





#importing the emotion-word list from external file
file = "words.xlsx"
df = pd.read_excel(file, index_col=None, na_values=['NA'])

anger=df[(df['Anger']==1)]
Anger=anger['English (en)']
Anger_list=Anger.tolist()


anticipation=df[(df['Anticipation']==1)]
Anticipation=anticipation['English (en)']
Anticipation_list=Anticipation.tolist()


disgust=df[(df['Disgust']==1)]
Disgust=disgust['English (en)']
Disgust_list=Disgust.tolist()


fear=df[(df['Fear']==1)]
Fear=fear['English (en)']
Fear_list=Fear.tolist()


joy=df[(df['Joy']==1)]
Joy=joy['English (en)']
Joy_list=Joy.tolist()


sadness=df[(df['Sadness']==1)]
Sadness=sadness['English (en)']
Sadness_list=Sadness.tolist()


surprise=df[(df['Surprise']==1)]
Surprise=surprise['English (en)']
Surprise_list=Surprise.tolist()


trust=df[(df['Trust']==1)]
Trust=trust['English (en)']
Trust_list=Trust.tolist()
print("emotion-word list imported from external file")


#emotion detection of each review using nltk
text=""
for i in range(len(List)):
    reviewList = List[i].split(" ")
    reviewList = [w for w in reviewList if w not in stoplist]
    Anger_reviewList = [w for w in reviewList if w in Anger_list]
    Anger_count=len(Anger_reviewList)
    Anticipation_reviewList = [w for w in reviewList if w in Anticipation_list]
    Anticipation_count = len(Anticipation_reviewList)
    Disgust_reviewList = [w for w in reviewList if w in Disgust_list]
    Disgust_count = len(Disgust_reviewList)
    Fear_reviewList = [w for w in reviewList if w in Fear_list]
    Fear_count = len(Fear_reviewList)
    Joy_reviewList = [w for w in reviewList if w in Joy_list]
    Joy_count = len(Joy_reviewList)
    Sadness_reviewList = [w for w in reviewList if w in Sadness_list]
    Sadness_count = len(Sadness_reviewList)
    Surprise_reviewList = [w for w in reviewList if w in Surprise_list]
    Surprise_count = len(Surprise_reviewList)
    Trust_reviewList = [w for w in reviewList if w in Trust_list]
    Trust_count = len(Trust_reviewList)
    List[i] = " ".join(reviewList)
    m={}
    m['Anger']=Anger_count
    m['Anticipation'] = Anticipation_count
    m['Disgust'] = Disgust_count
    m['Fear'] = Fear_count
    m['Joy'] = Joy_count
    m['Sadness'] = Sadness_count
    m['Surprise'] = Surprise_count
    m['Trust'] = Trust_count
    keymax = max(m, key=m.get)
    text+="Emotion : "+str(keymax)+"\n"+str(m)+"\n"+"Review :"+ List[i]+"\n\n\n\n"
print("emotion detection is carried out")
f = open("emotion.txt","w+",encoding="utf-8")
f.write(text)
print("data obtained from emotion detection is extracted to the emotion.txt file")
















