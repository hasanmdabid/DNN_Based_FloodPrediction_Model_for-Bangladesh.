import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df = pd.read_csv('flood.csv', na_values=['#NAME?'])
newdf = df.drop("Sl", axis='columns')
newdf['Flood?'].fillna(value=0, inplace=True)
x = newdf.drop(['Flood?'], axis=1)
y = newdf['Flood?']

print('Before the Dummy Categorical:', x.shape)

plt.figure(figsize=(8, 4))
y.value_counts().plot(kind='bar', rot=0, color=['C0', 'C1'])
plt.savefig('Binary Insatnces', dpi=300)

sns.set_style('whitegrid')
sns.heatmap(x.isnull(), cmap='viridis')
plt.savefig('Checking for missing data.png', dpi=300)
for col_name in x.columns:
    if x[col_name].dtypes == 'object':
        unique_cat = len(x[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
todummy_list = ['Station_Names']

#------------- Creating the dummy list of the categorical features -----------------------------------------------------
# Decide which categorical variables you want to use in model

def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, axis=1)
        df = pd.concat([df, dummies], axis=1)
    return df
x = dummy_df(x, todummy_list)
print('After The Dummy Categorical:', x.shape)

#-------------Balancing the dataset using the SMOTENC techniquue_-------------------------------------------------------

oversample = SMOTENC(random_state=42, categorical_features=[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
  41, 42, 43, 44, 45, 46, 47, 48], sampling_strategy='auto')
x_samp, y_samp = oversample.fit_resample(x, y)

print('shape of x_SMOTE:', x_samp.shape)
print('shape of y_SMOTE:', y_samp.shape)

plt.figure(figsize=(8,4))
y_samp.value_counts().plot(kind='bar', rot=0, color=['C0', 'C1'])
plt.savefig('Binary Insatnces after Data Augmentation', dpi=300)

from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames

    # Remove interaction terms with all 0 values
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)

    return df

x_samp = add_interactions(x_samp) # Creating Interactions Between each Column
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_samp, y_samp, train_size=0.70, random_state=1)

#-----------------Dimentionality Reduction Using PCA --------------------------------------------------------

# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
from sklearn.feature_selection import chi2
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(score_func=chi2, k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [x_samp.columns[i] for i in indices_selected]

X_train_selected_unscaled = X_train[colnames_selected]
X_test_selected_unscaled = X_test[colnames_selected]
print('Name of the Selcted Columns:', colnames_selected)


#-----------------Data Scaling and Normalization-----------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_selected_unscaled)
X_train_selected = scaler.transform(X_train_selected_unscaled)
X_test_selected = scaler.transform(X_test_selected_unscaled)

#----------------------------------Model Design-------------------------------------------------------------------------
# Design Multiple Model
model = input('Enter the Model Name:')
if model == 'KNN':
#------------------KNN Classifier -------------------------------------------------------------------------

    def KNN(X_train, y_train, X_test, y_test):
        model = KNeighborsClassifier(n_neighbors=21)
        model.fit(X_train, y_train)
        y_hat = [x[1] for x in model.predict_proba(X_test)]
        auc = roc_auc_score(y_test, y_hat)

        fpr_knn, tpr_knn, _ = metrics.roc_curve(y_test, y_hat)
        plt.plot(fpr_knn, tpr_knn, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        return auc, predictions, fpr_knn, tpr_knn
    # Find performance of model using preprocessed data
    auc_processed, predictions, fpr, kpr = KNN(X_train_selected, y_train, X_test_selected, y_test)
    print('\n')
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print('Accuracy of the KNN Model (Using Elbow Method):', auc_processed)

#----------------------------------------------LR----------------------------------------------------------------------
elif model == 'LR':
    from sklearn.linear_model import LogisticRegression
    def find_lr_model_perf(X_train, y_train, X_test, y_test):
        model = LogisticRegression(solver='lbfgs', max_iter=500)
        model.fit(X_train, y_train)
        y_hat = [x[1] for x in model.predict_proba(X_test)]
        fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test, y_hat)
        auc = roc_auc_score(y_test, y_hat)
        plt.plot(fpr_lr, tpr_lr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        return auc, predictions, fpr_lr, tpr_lr

    # Find performance of model using preprocessed data
    auc_processed, predictions, fpr_lr, tpr_lr = find_lr_model_perf(X_train_selected, y_train, X_test_selected, y_test)
    print('\n')
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print('Accuracy of the Model:', auc_processed)

#______________________________________________RF___________________________________________________________________
elif model == 'RF':
    from sklearn.ensemble import RandomForestClassifier

    def find_rf_model_perf(X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(n_estimators=500)
        model.fit(X_train, y_train)
        y_hat = [x[1] for x in model.predict_proba(X_test)]
        auc = roc_auc_score(y_test, y_hat)

        fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_hat)
        plt.plot(fpr_rf, tpr_rf, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()

        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        return auc, predictions, fpr_rf, tpr_rf


    # Find performance of model using preprocessed data
    auc_processed, predictions, frp, tpr = find_rf_model_perf(X_train_selected, y_train, X_test_selected, y_test)
    print('\n')
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print('Accuracy of Random Forest Model:', auc_processed)

elif model == 'SVM':
    from sklearn.svm import SVC
    def find_svc_model_perf(X_train, y_train, X_test, y_test):
        model = SVC(kernel='rbf', probability=True)
        model.fit(X_train, y_train)
        y_hat = [x[1] for x in model.predict_proba(X_test)]
        auc = roc_auc_score(y_test, y_hat)

        fpr_svc, tpr_svc, _ = metrics.roc_curve(y_test, y_hat)
        plt.plot(fpr_svc, tpr_svc, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()

        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        return auc, predictions, fpr_svc, tpr_svc

    # Find performance of model using preprocessed data
    auc_processed, predictions, frp, tpr = find_svc_model_perf(X_train_selected, y_train, X_test_selected, y_test)
    print('\n')
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print('Accuracy of the Model:', auc_processed)

    from sklearn.model_selection import GridSearchCV

    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=10)
    grid.fit(X_train_selected,y_train)
    print(grid.best_params_)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test_selected)
    print(confusion_matrix(y_test,grid_predictions))
    print('\n')
    print(classification_report(y_test,grid_predictions))


##-------------------------------->Deep convolutional network<----------------------------------------------------------
# Generating the 3D data frame (Nr Frame, Nr channels, Nr of Outputs)
elif model == 'DEEP':
    from sklearn.preprocessing import MinMaxScaler
    from keras.callbacks import EarlyStopping

    scaler = MinMaxScaler()
    scaler.fit_transform(X_train_selected_unscaled)

    X_train_selected = scaler.transform(X_train_selected_unscaled)
    X_test_selected = scaler.transform(X_test_selected_unscaled)

    model = Sequential()

    model.add(Dense(40, activation='relu'))

    model.add(Dense(20, activation='relu'))

    model.add(Dense(10, activation='relu'))

    # Binary Classification Probelm so the last activation function will be Sigmoid ...
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Fitting the model

    model.fit(x=X_train_selected, y=y_train, epochs=200, validation_data=(X_test_selected, y_test))
    # Plotting the Over Fitting problem
    losses = pd.DataFrame(model.history.history)

    losses.plot()
    plt.savefig('Deep Neural Network over fitting problem.png', dpi=200)

    #Solutu=ion of Over Fitting Problem by Dropout and Early Stopping

    model = Sequential()

    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))

    # Binary Classification Probelm so the last activation function will be Sigmoid ...
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')


    help(EarlyStopping)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    model.fit(x=X_train_selected, y=y_train, epochs=200, validation_data=(X_test_selected, y_test),
              callbacks=[early_stop])


    model_loss = pd.DataFrame(model.history.history)
    model_loss.plot()
    plt.savefig('Deep Neural Network over fitting problem solved by Dropout and Callback functoin.png', dpi=200)

    predictions = (model.predict(X_test_selected) > 0.5).astype("int32")
    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))
    print('\n')

