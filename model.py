import sklearn as sk
import pandas as pd

network = pd.read_csv("data/wustl_iiot_2021.csv", header=0)

df = pd.DataFrame(network)
