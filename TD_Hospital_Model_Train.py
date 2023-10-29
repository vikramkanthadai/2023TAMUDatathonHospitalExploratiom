import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# Added imports necessary for column choosing
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, RFECV, VarianceThreshold, f_regression

from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score


def data_preprocessing(df):

    #col_to_keep = ['timeknown', 'death', 'age', 'blood', 'psych1', 'race',
    #               'breathing', 'cancer', 'disability', 'glucose',
    #               'confidence', 'primary']

    col_to_keep = ['death', 'timeknown', 'age', 'cancer', 'disability', 'glucose', 'confidence']

    # removed columns: 'sex', 'psych3', 'reflex', 'bloodchem1', 'bloodchem2', 'psych2', 'extraprimary', 'dnr','comorbidity', 'diabetes',
    df = df[col_to_keep]

    df_cp = df.copy()

    # df['race'] = df_cp['race'].replace({'asian': 1, 'black': 2, 'hispanic': 3, 'white': 4, 'other': 0})
    # f['dnr'] = df_cp['dnr'].replace({'no dnr': 0, 'dnr before sadm': 1, 'dnr after sadm': 2})
    df['cancer'] = df_cp['cancer'].replace({'no': 0, 'metastatic': 1, 'yes': 2})
    # df['primary'] = df_cp['primary'].replace(
    #    {'ARF/MOSF w/Sepsis': 0, 'MOSF w/Malig': 1, 'CHF': 2, 'COPD': 3, 'Cirrhosis': 4, 'Colon Cancer': 5,
    #     'Lung Cancer': 6, 'Coma': 7})
    # df['extraprimary'] = df_cp['extraprimary'].replace({'ARF/MOSF': 0, 'COPD/CHF/Cirrhosis': 1, 'Cancer': 2, 'Coma': 3})
    df['disability'] = df_cp['disability'].replace(
        {'no(M2 and SIP pres)': 0, '<2 mo. follow-up': 1, 'adl>=4 (>=5 if sur)': 2, 'SIP>=30': 3, 'Coma or Intub': 4})

    df['confidence'] = df_cp['confidence'].fillna(df_cp['confidence'].mean())

    # Renoved due to unknown value of '1'
    # df['sex'] = df_cp['sex'].replace({'female' : 0, 'male' : 1, 'Male' : 1, 'M' : 1, '1' : 1})

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

'''def choose_columns(X, y):
    # CREATING TEMP DATA SET
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Choosing 10 best columns of data
    sfs = SequentialFeatureSelector(DecisionTreeRegressor(), direction='backward', scoring="neg_root_mean_squared_error",
                                    n_features_to_select='auto', tol=None, cv=5, n_jobs=-1)
    sfs.fit(X_train, y_train)

    chosen_cols = X_train.columns[sfs.support_]
    print(chosen_cols)
    return chosen_cols'''


def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(64, activation='relu'),  # Another hidden layer with 64 neurons and ReLU activation
        tf.keras.layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification

    ])
    # ADDITION: ADDED LAYER OF 256 AT THE BEGINNING

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.h5')
    
    print(f'Test accuracy: {test_accuracy}')

    # Optionally, you can plot training history to visualize model performance
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":

    data_path = './TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)

    # Code to look at the column data
    # print(df.columns)
    # df2 = df.pivot_table(index = ['dnr'], aggfunc ='size')
    # print(df2)

    cleaned_data = data_preprocessing(df)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)

    # Used with Feature selection
    #cols = choose_columns(X, y)
    #X = X[cols]

    train_model(X, y)
    