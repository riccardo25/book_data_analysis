import winsound as ws
import numpy as np 
import pandas as pd
import warnings
from sklearn.cluster import KMeans

class Predictions:


    # self.alldata          dataset of all books
    # self.userbook         dataset of ratings user-book
    # self.userresume       dataset of users(by )


    def __init__(self):
        self.alldata = pd.read_csv("..\\dataset\\finals\\complissivolibricongenere.csv").drop(axis=1, columns=["Unnamed: 0"])
        self.userbook = pd.read_csv("..\\dataset\\ratings.csv")
        self.userresume = pd.read_csv("..\\dataset\\finals\\userresume.csv").drop(axis=1, columns=["Unnamed: 0"])


    #returns the most rappresentative books
    def getselectionimages(self, n = 40):
        #generating data first selection
        #singular = alldata.drop_duplicates(subset=["authors"], keep="first")
        mostknown = self.alldata.sort_values(["work_ratings_count"],ascending= False )[self.alldata.work_ratings_count>100000]
        authors = mostknown["authors"].value_counts().index.tolist()
        firstauth = []
        for au in authors:
            firstauth.append(mostknown[mostknown.authors == au].sample(frac=1)[0:3])
        singular = pd.concat(firstauth)
        selbooks = singular.sort_values(["work_ratings_count"],ascending= False )[0:40]
            
        return self.formatbooksout(selbooks)

    def suggestBooks(self, selectedbooks):
        selectedbooksid = []
        for book in selectedbooks:
            selectedbooksid.append(self.alldata[self.alldata.book_id == book]["id"].values[0])
        positives = self.userbook[self.userbook.rating >=4]
        intersect = set([])
        for i in range(len(selectedbooksid)):
            if(i == 0):
                intersect = set(positives[positives.book_id == selectedbooksid[0]]["user_id"])
            else:
                intersect = intersect.intersection( set(positives[positives.book_id == selectedbooksid[i]]["user_id"]) )
        #these users has the same background 
        intersect = list(intersect)
        similusers = pd.DataFrame(data={"user_id":intersect})
        ratingpoint= []
        for userid in intersect:
            ratingpoint.append(self.userresume[self.userresume.user_id == userid]["ratingpoint"].values[0])
        similusers = similusers.assign(ratingpoints = ratingpoint)
        #take only 3 people
        bestusers = similusers.sort_values(["ratingpoints"],ascending= True )["user_id"].values[0:3]
        #now i have most similar person to my user, so I need to select books from him
        bestintersect = set([])
        for user in bestusers:
            if(user == bestusers[0]):
                bestintersect = set(positives[positives.user_id == user]["book_id"].values)
            else:
                bestintersect = bestintersect.intersection( set(positives[positives.user_id == user]["book_id"].values) )
        bestintersect = list(bestintersect)
        #remove the selected books
        for b in selectedbooksid:
            bestintersect.remove(b)
        returndata = []
        for bo in bestintersect:
            returndata.append(self.alldata[self.alldata.id == bo])
        allbooksout = pd.concat(returndata)
        
        return self.formatbooksout(allbooksout)
    

    #formats the output of books
    def formatbooksout(self, datasetin):
        returnid = datasetin["book_id"].values.tolist()
        returntitle = datasetin[ "title"].values.tolist()
        returntitle = [w.replace(',', '') for w in returntitle]
        returnprice = datasetin[ "price"].values.tolist()
        returnauthor = datasetin[ "authors"].values.tolist()
        returnauthor = [w.replace(',', '') for w in returnauthor]
        returnstr = ""
        for i in range(len(returnid)):
            if(i !=0):
                returnstr += ","
            returnstr += str(returnid[i])+"/"+returntitle[i]+"/"+str(returnprice[i])+"/"+returnauthor[i]
        return returnstr