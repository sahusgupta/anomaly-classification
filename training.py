from sklearn import linear_model
import pandas as pd

network = pd.read_csv("data/training.csv", header=0)

cols = ["SrcAddr","DstAddr","Mean","Sport","Dport","SrcPkts","DstPkts","TotPkts","DstBytes","SrcBytes","TotBytes","SrcLoad","DstLoad","Load","SrcRate","DstRate","Rate","SrcLoss","DstLoss","Loss","pLoss","SrcJitter","DstJitter","SIntPkt","DIntPkt","Proto","Dur","TcpRtt","IdleTime","Sum","Min","Max","sDSb","sTtl","dTtl","sIpId","dIpId","SAppBytes","DAppBytes","TotAppByte","SynAck","RunTime","sTos","SrcJitAct","DstJitAct","Traffic","Target"]

df = pd.DataFrame(network)
# load all data in the dataset
x = df.loc[:, :]
y = df.Traffic

model = linear_model.LogisticRegression()
model.fit(x, y)