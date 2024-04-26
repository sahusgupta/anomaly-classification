from sklearn import linear_model, preprocessing
import pandas as pd

# load the training dataset and initialize the object encoder
network = pd.read_csv("data/training.csv", header=0)
encoder = preprocessing.LabelEncoder()

# iterate through columns in network and transform non-integer values to doubles
for col in network.columns:
    if network[col].dtype == object:
        network[col] = encoder.fit_transform(network[col])
    else:
        continue
    
# columns of interest
cols = ["StartTime", "LastTime","SrcAddr","DstAddr","Mean","Sport","Dport","SrcPkts","DstPkts","TotPkts","DstBytes","SrcBytes","TotBytes","SrcLoad","DstLoad","Load","SrcRate","DstRate","Rate","SrcLoss","DstLoss","Loss","pLoss","SrcJitter","DstJitter","SIntPkt","DIntPkt","Proto","Dur","TcpRtt","IdleTime","Sum","Min","Max","sDSb","sTtl","dTtl","sIpId","dIpId","SAppBytes","DAppBytes","TotAppByte","SynAck","RunTime","sTos","SrcJitAct","DstJitAct","Traffic","Target"]
# initialize the dataframe
df = pd.DataFrame(network)
# load all data in the dataset
x = df.loc[:, :]
y = df.Traffic

# train the model, then use the trained model to predict in testing_model.py
model = linear_model.LogisticRegression(max_iter = 10**7)
model.fit(x, y)