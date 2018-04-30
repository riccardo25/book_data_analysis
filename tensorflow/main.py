import numpy as np
import pandas as pd
import tensorflow as tf
from Model import ModelCNN

books = pd.read_csv("..\\dataset\\finals\\complissivolibricongenere.csv").drop(axis=1, columns=["Unnamed: 0"])

print("Loading traing data.... ")
train_data = books[:9000][["original_publication_year", "color", "genderclass"]].as_matrix()
train_labels = books[:9000][[ "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]].as_matrix()

print("Training data:" + str(len(train_data)))
print("done")
print("Loading evaluation data... ")

evaluate_data = books[9000:][["original_publication_year", "color", "genderclass"]].as_matrix()
evaluate_labels = books[9000:][[ "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5"]].as_matrix()

print("done")
print("Creating model...")

cnn = ModelCNN()

print("Let's train!")

cnn.train(1000, train_data, train_labels, evaluate_data, evaluate_labels)

print("Train end")

result = cnn.evaluate([[2009, 0, 14]], False)
print(result)
