import pandas as pd
from sklearn import metrics, preprocessing
from training import model

# load the dataset and  initialize object encoder
network = pd.read_csv("data/testing.csv", header=0)
encoder  = preprocessing.LabelEncoder()
# iterate through columns in network and transform non-integer values to doubles
for col in network.columns:
    if network[col].dtype == object:
        network[col] = encoder.fit_transform(network[col])
    else:
        continue
    
# columns of interest
cols = ["StartTime", "LastTime", "SrcAddr","DstAddr","Mean","Sport","Dport","SrcPkts","DstPkts","TotPkts","DstBytes","SrcBytes","TotBytes","SrcLoad","DstLoad","Load","SrcRate","DstRate","Rate","SrcLoss","DstLoss","Loss","pLoss","SrcJitter","DstJitter","SIntPkt","DIntPkt","Proto","Dur","TcpRtt","IdleTime","Sum","Min","Max","sDSb","sTtl","dTtl","sIpId","dIpId","SAppBytes","DAppBytes","TotAppByte","SynAck","RunTime","sTos","SrcJitAct","DstJitAct","Traffic","Target"]
# initialize the dataframe and load all columns (except whether the traffic is normal or anomaly) and rows
df = pd.DataFrame(network)
x = df.loc[:, cols]

# want to predict whether a piece of traffic is anomaly or normal, encoded as 1 or 2
y = df.Traffic
# test the model using the loaded information
classifications = model.predict(x)
# compute accuracy metrics
accuracy = metrics.accuracy_score(y, classifications)
print(accuracy)

