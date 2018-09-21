#SALVANDO O ARQUIVO DE TREINADO
"""
import pickle
pickle.dump(classifier, open('SVM_Trained_Model.sav', 'wb'))
"""


#CARREGANDO O ARQUIVO TREINADO.
"""
from sklearn.externals import joblib
classifier = joblib.load('SVM_Trained_Model.sav')
"""
