import winsound as ws
import numpy as np 
import pandas as pd
import warnings
from sklearn.cluster import KMeans
import sklearn.utils

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
        #select the 3 bestusers
        self.elaborateselected( selectedbooks )
        #now i have most similar person to my user, so I need to select books from him
        bestintersect = set([])
        for user in self.bestusers:
            if(user == self.bestusers[0]):
                bestintersect = set(self.positives[self.positives.user_id == user]["book_id"].values)
            else:
                bestintersect = bestintersect.intersection( set(self.positives[self.positives.user_id == user]["book_id"].values) )
        bestintersect = list(bestintersect)
        #remove the selected books
        for b in self.selectedbooksid:
            bestintersect.remove(b)
        returndata = []
        for bo in bestintersect:
            returndata.append(self.alldata[self.alldata.id == bo])
        allbooksout = pd.concat(returndata)
        return self.formatbooksout(allbooksout)

    def suggestauthorbooks(self, selectedbooks):

        self.elaborateselected( selectedbooks)
        #now i have most similar person to my user, so I need to select books from him
        #get all books readed by selected users
        allbooksreaded = []
        for user in self.bestusers:
            allbooksreaded.append( self.positives[self.positives.user_id == user]) 
        allbooksreaded = pd.concat(allbooksreaded)
        for i in self.selectedbooksid:
            allbooksreaded = allbooksreaded[allbooksreaded.book_id != i]
        bookdesc = []
        for book in allbooksreaded["book_id"]:
            bookdesc.append( self.alldata[self.alldata.book_id == book]) 
        bookdesc = pd.concat(bookdesc)
        auth = bookdesc["authors"].value_counts()
        auth = auth[auth >1].index.values.tolist()
        selectedauthors = []
        for i in self.selectedbooksid:
            selectedauthors.append(self.alldata[self.alldata.id ==i]["authors"].values[0])
        auth = auth + selectedauthors
        auth = pd.Series(auth).drop_duplicates(keep="first").values.tolist()

        allbooksreaded = allbooksreaded.drop_duplicates(subset=["book_id"], keep="first")
        aulist = []
        for bid in allbooksreaded["book_id"].values.tolist():
            author = self.alldata[self.alldata.id == bid]["authors"]
            if(not author.empty):
                aulist.append(author.values[0])
            else:
                aulist.append("NaN")
        allbooksreaded = allbooksreaded.assign(authors = aulist)
        elabau = []
        for au in auth:
            elabau.append(sklearn.utils.shuffle(allbooksreaded[allbooksreaded.authors == au])[0:3])
        elabau = pd.concat(elabau)["book_id"].values.tolist()
        outbook = []
        for e in elabau:
            outbook.append(self.alldata[self.alldata.id == e])
        outbook = pd.concat(outbook)
        return self.formatbooksout(outbook)
        
    
    def suggestgenderbooks(self, selectedbooks):
        self.elaborateselected( selectedbooks)
        #now i have most similar person to my user, so I need to select books from him
        bookdesc = []
        for book in self.selectedbooksid:
            bookdesc.append( self.alldata[self.alldata.id == book]) 
        bookdesc = pd.concat(bookdesc)

        bookdesc = bookdesc.drop_duplicates(subset=["gendernames"], keep="first")
        gender = bookdesc["gendernames"].values.tolist()
        allbooksreaded = []
        for user in self.bestusers:
            allbooksreaded.append( self.positives[self.positives.user_id == user]) 
        allbooksreaded = pd.concat(allbooksreaded)
        for i in self.selectedbooksid:
            allbooksreaded = allbooksreaded[allbooksreaded.book_id != i]
        bookdesc = []
        for book in allbooksreaded["book_id"]:
            bookdesc.append( self.alldata[self.alldata.book_id == book]) 
        bookdesc = pd.concat(bookdesc)

        outbook =[]
        for g in gender:
            outbook.append(bookdesc[bookdesc.gendernames == g])
        outbook = pd.concat(outbook)
        return self.formatbooksout(outbook)


    def elaborateselected(self, selectedbooks, n_users =3):
        self.selectedbooksid = []
        for book in selectedbooks:
            self.selectedbooksid.append(self.alldata[self.alldata.book_id == book]["id"].values[0])
        self.positives = self.userbook[self.userbook.rating >=4]
        intersect = set([])
        for i in range(len(self.selectedbooksid)):
            if(i == 0):
                intersect = set(self.positives[self.positives.book_id == self.selectedbooksid[0]]["user_id"])
            else:
                intersect = intersect.intersection( set(self.positives[self.positives.book_id == self.selectedbooksid[i]]["user_id"]) )
        #these users has the same background 
        intersect = list(intersect)
        similusers = pd.DataFrame(data={"user_id":intersect})
        ratingpoint= []
        for userid in intersect:
            ratingpoint.append(self.userresume[self.userresume.user_id == userid]["ratingpoint"].values[0])
        similusers = similusers.assign(ratingpoints = ratingpoint)
        #take only 3 people
        self.bestusers = similusers.sort_values(["ratingpoints"],ascending= True )["user_id"].values[0:n_users]



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