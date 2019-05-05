import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


columnNames = ['y','cap-shape',
                      'cap-surface',
                      'cap-color',
                      'bruises',
                      'odor',
                      'gill-attachment',
                      'gill-spacing',
                      'gill-size',
                      'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
'stalk-color-above-ring','stalk-color-below-ring', 'veil-type','veil-color', 'ring-number', 'ring-type',
                      'spore-print-color', 'population', 'habitat']

yX_train = pd.DataFrame(pd.read_csv('train/train.tsv', encoding="utf-8", delimiter='\t', header=None, names=columnNames))

y_temp= yX_train['y'].values
#print (y_temp)
y_train = np.where(y_temp == 'p', 0, 1)
#print (y_train)
X_train = yX_train.drop('y', axis=1)
print (X_train)
X_train_one_hot = pd.get_dummies( X_train)
print (X_train_one_hot)

#Zbiór testowy:
columnNames2 = ['cap-shape',
                      'cap-surface',
                      'cap-color',
                      'bruises',
                      'odor',
                      'gill-attachment',
                      'gill-spacing',
                      'gill-size',
                      'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
'stalk-color-above-ring','stalk-color-below-ring', 'veil-type','veil-color', 'ring-number', 'ring-type',
                      'spore-print-color', 'population', 'habitat']
X_test = pd.DataFrame(pd.read_csv('test-A/in.tsv', encoding="utf-8", delimiter='\t', header=None, names=columnNames2))

X_test_one_hot = pd.get_dummies(X_test)

def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0


def fix_columns( d, columns ):

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print ("extra columns in X_test:", extra_cols)

    d = d[ columns ]
    return d


fixed_X_test_one_hot = fix_columns(X_test_one_hot.copy(), X_train_one_hot.columns )

#trenowanie:
my_classifier = GaussianNB()
my_classifier.fit(X_train_one_hot, y_train)

y_out_predicted = my_classifier.predict(fixed_X_test_one_hot)
print (y_out_predicted)
y_out = np.where(y_out_predicted == 1, 'e', 'p')


with open('test-A/out.tsv', 'w') as output_file:
    for out in y_out:
        print('%s' % out, file=output_file)

