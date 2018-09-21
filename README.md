# Python_Saving-and-Loading-a-Trained-SVM-Model

_O Exemplo tem base em um outro repositorio (Python_Image-Classifier-with-SVM) onde mostro um classificador de imagens usando SVM.

_Repositorio do Projeto "Python_Image-Classifier-with-SVM":

https://github.com/F-Souza/Python_Image-Classifier-with-SVM

_OBS: O Arquivo 'SVM_Trained_Model.sav' que está nesse repositorio é um modelo treinado para classificar bananas e maçãs.

      #SALVANDO O ARQUIVO DE TREINADO
      
      import pickle
      pickle.dump(classifier, open('SVM_Trained_Model.sav', 'wb'))
      

      #CARREGANDO O ARQUIVO TREINADO.
      
      from sklearn.externals import joblib
      classifier = joblib.load('SVM_Trained_Model.sav')

- EXPLICAÇÃO ABAIXO.

- Treinamento SVM.

      import numpy as np
      import cv2
      from sklearn.svm import SVC

      Y = []

      for h in range(1,25):
      
          if (h <= 12):    
          
              Y.append(0)
              
          else:
          
              Y.append(1)

      Y = np.array(Y)

      imagens = []
      
      for i in range(1,25):
      
          imagens.append(cv2.imread('data_'+str(i)+'.png'))
          
      X = np.array(imagens)
      
      n_samples = len(X)
      
      data_images = X.reshape((n_samples, -1))

      from sklearn.model_selection import train_test_split
      
      X_train, X_test, Y_train, Y_test = train_test_split(data_images,Y, test_size = 0.20, random_state = 0)

      from sklearn.preprocessing import StandardScaler
      
      sc_X = StandardScaler()
      
      X_train = sc_X.fit_transform(X_train)
      
      sc_X2 = StandardScaler()
      
      X_test = sc_X2.fit_transform(X_test)

      classifier = SVC(kernel = 'rbf', random_state = 0)
      
      classifier.fit(X_train,Y_train)
      
      Y_pred = classifier.predict(X_test)

      from sklearn.metrics import confusion_matrix
      
      cm = confusion_matrix(Y_test, Y_pred)

- Salvando o Model.

- Importando a Biblioteca 'pickle' necessário para efetuar o salvamento.

      import pickle
      
- Para salvar o SVM treinado, passamos a variavel responsavel pelo classificador do nosso código, portanto, passamos a variavel
"classifier", logo depois passamos o nome do arquivo que vai ser salvo 'SVM_Trained_Model' na extensão '.sav'.
      
      pickle.dump(classifier, open('SVM_Trained_Model.sav', 'wb'))

- O Arquivo será salvo na pasta do projeto e a partir dele você pode poderá classificar de forma mais rápida em outros projetos.
Basta passar o arquivo '.sav' para a pasta do projeto no qual você deseja usalo.

- Agora, como carregar e usar o 'SVM_Trained_Model.sav'.
- Carregando as imagens a serem testadas da mesma forma que do projeto "Python_Image-Classifier-with-SVM":

      imgs = []
      
      for j in range(0,6):
      
          imgs.append(cv2.imread('teste_'+str(j)+'.png'))
          
      T = np.array(imgs)
      
      t_samples = len(T)
      
      teste_imagens = T.reshape((t_samples, -1))

      sc_T = StandardScaler()
      
      teste_imagens = sc_T.fit_transform(teste_imagens)

- Agora, já que temos nosso arquivo '.sav' com o classificador, não precisamos mais efetuar todo o processo do SVM, basta apenas
carregar o arquivo '.sav' e usalo.

- Importando a Biblioteca joblib para carregarmos o arquivo '.sav'.

      from sklearn.externals import joblib
      
- Ao carregar o arquivo precisamos guarda-lo em uma variavel para usarmos como classificador depois.
      
      classifier = joblib.load('SVM_Trained_Model.sav')

- Classificando usando a variavel que acabamos de criar.

      predict = classifier.predict(teste_imagens)
