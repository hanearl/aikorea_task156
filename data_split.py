import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train_ori.tsv', sep='\t')

X = df['comment']
y = df['tag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52, stratify=y, shuffle=True)

print(len(X_train))
print(len(X_test))

train_set = pd.DataFrame({'comment': X_train, 'tag': y_train})
test_set = pd.DataFrame({'comment': X_test, 'tag' : y_test})

train_set.to_csv('data/train.tsv', sep='\t', index=False)
test_set.to_csv('data/val.tsv', sep='\t', index=False)

