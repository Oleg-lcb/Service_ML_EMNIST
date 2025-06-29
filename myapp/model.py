import os
import joblib


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        # your code here
        """
                Загружаем модель из файла model.pkl.
                """
        try:
            with open(model_path, 'rb') as file:
                # Попытка загрузить модель с помощью joblib
                self.model = joblib.load(file)
                print("Модель успешно загружена.")
        except FileNotFoundError:
            raise FileNotFoundError("Файл model.pkl не найден в текущей директории.")

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
        # Make predictions
        print(x.shape)
        predictions = self.model.predict(x)

        # Return the first prediction as a character
        return str(predictions)
