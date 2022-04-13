import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

file_path = './query1_dataset.csv'

df = pd.read_csv(file_path, sep=';')

# discourage
df.discourage = df.discourage.map({'no':False, 'yes':True})

# system
mlb = MultiLabelBinarizer()
df.system = df.system.str.split('/')
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('system')),
                columns=[f"system_{i}" for i in mlb.classes_],
                index=df.index))

# property
df = df.join(pd.get_dummies(df.pop('property'), prefix="property"))

# scale
df = df.join(pd.get_dummies(df.pop('scale'), prefix="scale"))

# class
df = df.join(pd.get_dummies(df.pop('class'), prefix="class"))

print(df.head(5))

# pairwise transformation

