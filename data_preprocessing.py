import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

file_path = './data/'

df_queries = pd.read_csv(f"{file_path}queries.csv")
df_docs = pd.read_csv(f"{file_path}docs.csv")
df_query_doc = pd.read_csv(f"{file_path}query_doc.csv")

merge1 = pd.merge(df_queries,df_query_doc, on="id_query")
df = pd.merge(merge1, df_docs, on="loinc_num")

print(df.head(1))

