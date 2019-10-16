#importing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV

#Reading files
train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
y_test= pd.read_csv("gender_submission.csv")

#'y' is dependent variable for the train data 
y= train.iloc[:,1].values

train=train.drop('Survived',axis=1,inplace=False)

#combining the train and test data
titanic=train.append(test,ignore_index=True)

index= len(train)


#Feature Scaling
titanic["Title"]=titanic.Name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
titanic.Title = titanic.Title.map(normalized_titles)



titanic["Age"].fillna(titanic["Age"].median(), inplace=True)

titanic["Embarked"].fillna(titanic["Embarked"].value_counts().index[0],inplace=True)
 
titanic.Cabin.fillna("U",inplace=True)
titanic.Cabin=titanic.Cabin.map(lambda x:x[0])

titanic['Fare'].fillna(titanic['Fare'].median(),inplace=True)

titanic['Family']= titanic['SibSp'] + titanic['Parch'] +1
titanic['Singleton'] = titanic['Family'].map(lambda s: 1 if s == 1 else 0)
titanic['SmallFamily'] = titanic['Family'].map(lambda s: 1 if 2 <= s <= 4 else 0)
titanic['LargeFamily'] = titanic['Family'].map(lambda s: 1 if 5 <= s else 0)

titanic.Sex=titanic.Sex.map({"male":0,"female":1})

#creating dummy variables
pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")
title_dummies = pd.get_dummies(titanic.Title, prefix="Title")
cabin_dummies = pd.get_dummies(titanic.Cabin, prefix="Cabin")
embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")
titanic_dummies = pd.concat([titanic, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)
titanic_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket','SibSp','Parch','PassengerId'], axis=1, inplace=True)


#divinding the test and train data
x=titanic_dummies[:index]
x_test=titanic_dummies[index:]



#modeling RandomForest
forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

forrest = RandomForestClassifier()

forest_cv = GridSearchCV(estimator=forrest,param_grid=forrest_params, cv=5) 
forest_cv.fit(x, y)

print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))


forrest_pred = forest_cv.predict(x_test)

holdout_ids =  y_test["PassengerId"]
holdout_predictions = forrest_pred

submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv('submission.csv',index=False)



