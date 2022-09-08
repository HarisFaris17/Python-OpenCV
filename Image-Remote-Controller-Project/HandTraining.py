import pickle
import numpy as np
trainingDirectory = './Image-Remote-Controller-Project/training'
dataTrainingDirectory = '/data.txt'

# if the script run and not imported
if __name__=='__main__':
    dataset = np.loadtxt(trainingDirectory+dataTrainingDirectory,dtype=float,delimiter=',')
    # features
    x = dataset[:,:-1]
    # since the labels should be int, hence convert them to int
    y = dataset[:,-1].astype(int)
    print(dataset)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=True)
    print(f'x_train : {x_train}')
    print(f'y_train : {y_train}')
    print(f'x_test : {x_test}')
    print(f'y_test : {y_test}')
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(10,criterion='entropy',random_state=0)
    classifier.fit(x_train,y_train)
    print([classifier.predict(x_test),y_test])
    print(classifier.score(x_test,y_test))
    # dump will save the obj to the file directory. The file will be created (if there is no such file before) and the file will be opened as write byte. This have to be wb, since the obj (classifier) is bytes not texts
    pickle.dump(classifier,open(f'{trainingDirectory}/model_hand.sav','wb'))
