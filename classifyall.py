import pandas as pd
import numpy as np
import joblib

#Read in input data
X = pd.read_csv("traindata.txt", names=[str(i) for i in range(71)])
y = pd.read_csv("trainlabels.txt", names=["labels"])


#Split into continuous and discrete features
standardise_columns = [str(i) for i in range(64)]
one_hot_columns = [str(i) for i in range(64, 71)]

#Load scaler for continuous
scaler = joblib.load("good_models/rfc_rbf_55/rfc_rbf_55_scaler.pkl")
#Load centres for kmeans clustering
kmeans = joblib.load("good_models/rfc_rbf_55/rfc_rbf_55_kmeans.pkl")

centers = kmeans.cluster_centers_
gamma = 0.3

#Extract and scale continuous
x_cont = X[standardise_columns]
X_scaled = scaler.transform(x_cont)

#Add radial basis functions
transformed = np.exp(-gamma * np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)**2)
x_standardised = np.concatenate((X_scaled, transformed), axis = 1)
x_1 = pd.DataFrame(x_standardised, columns=[str(i) for i in range(96)])

#One hot encode all discrete features
x_rem = X[one_hot_columns]
x_rem = pd.get_dummies(x_rem, columns=one_hot_columns)
x_rem.reset_index(drop=True, inplace=True)

#Join dataset
X = pd.concat([x_1, x_rem ], axis=1).values

#Load random forest classifier
rfc = joblib.load("good_models/rfc_rbf_55.pkl")

#Write to text file and predict
out = open("testlabels.txt", "w")
preds = rfc.predict(X)

for pred in preds:
    out.write(str(pred) + "\n")
    
out.close()

print(rfc.score(X,y))







