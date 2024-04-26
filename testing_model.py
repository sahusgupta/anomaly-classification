from sklearn import *
import pandas as pd
from sklearn import metrics
from training import model

network = pd.read_csv("data/testing.csv", header=0)

cols = ["SrcAddr","DstAddr","Mean","Sport","Dport","SrcPkts","DstPkts","TotPkts","DstBytes","SrcBytes","TotBytes","SrcLoad","DstLoad","Load","SrcRate","DstRate","Rate","SrcLoss","DstLoss","Loss","pLoss","SrcJitter","DstJitter","SIntPkt","DIntPkt","Proto","Dur","TcpRtt","IdleTime","Sum","Min","Max","sDSb","sTtl","dTtl","sIpId","dIpId","SAppBytes","DAppBytes","TotAppByte","SynAck","RunTime","sTos","SrcJitAct","DstJitAct","Target"]
df = pd.DataFrame(network)
# load all data in the dataset
x = df.loc[:, cols]
y = df.Traffic

classifications = model.predict(x)
accuracy = metrics.accuracy_score(y, classifications)
print(accuracy)