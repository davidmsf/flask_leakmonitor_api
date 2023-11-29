import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class PredictSniffingpoint:
    def __init__(self):
        self.model = self.load_model()
        self.scaler = self.load_scaler()
        self.encoder = self.load_encoder()

    @staticmethod
    def load_model():
        new_model = tf.keras.models.load_model('./models/leak.model')
        return new_model

    @staticmethod
    def load_scaler():
        with open('./models/scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
            return loaded_scaler

    @staticmethod
    def load_encoder():
        with open('./models/encoder.pkl', 'rb') as file:
            loaded_encoder = pickle.load(file)
            return loaded_encoder

    def reasons_one_hot_encode(self, df_leak):
        one_hot_encoded = self.encoder.transform(df_leak[['reason']])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.encoder.get_feature_names_out(['reason']))
        # Drop original 'reason' column and concatenate the one-hot encoded DataFrame
        df_leak = df_leak.drop(columns=['reason'])
        df_leak = pd.concat([df_leak, one_hot_df], axis=1)
        return df_leak

    def predict(self, input_data):
        true_result = self.get_true_result(input_data)

        input_data_edit = {'temp': [input_data['temp']], 'tryk': [input_data['tryk']],
                           'alder_år': [input_data['years']],
                           'plast': [input_data['plast']], 'kobber': [input_data['kobber']],
                           'stål': [input_data['stål']],
                           'reason': self.get_reason_nr(input_data)}
        values = pd.DataFrame(input_data_edit)

        numeric_cols = ['temp', 'tryk', 'alder_år']
        values[numeric_cols] = self.scaler.transform(values[numeric_cols])
        values = self.reasons_one_hot_encode(values)
        single_instance = np.array(values)

        # 1 indicates that you want the new shape to have one row.
        # -1 is a placeholder that means "whatever is needed," so numpy will automatically compute
        # the correct number for that dimension based on the original data.
        # By reshaping to (1, -1), you're effectively turning the single sample into a "batch"
        # with one sample so that it has the correct shape to feed into the model.
        single_instance = single_instance.reshape(1, -1)
        prediction = self.model.predict([single_instance])
        print(f'SP: {np.argmax(prediction) + 1}')
        rue_result_dict = true_result.to_dict()
        return {"true_result": rue_result_dict, "SP": int(np.argmax(prediction) + 1)}

    def get_reason_nr(self, input_data):
        reason_dict = {'Fitting': 0, 'Valve': 1, 'Joint': 2, 'Seal': 3, 'Gasket': 4, 'Flange': 5, 'Weld': 6,
                       'Connector': 7, 'Coupling': 8, 'Corrosion': 9, 'Crack': 10}
        return reason_dict[input_data['reason']]

    def get_true_result(self, input_data):
        df_leak_true_result = pd.DataFrame({
            'temp': [input_data['temp']],
            'tryk': [input_data['tryk']],
            'alder_år': [input_data['years']],
            'plast': [input_data['plast']],
            'kobber': [input_data['kobber']],
            'stål': [input_data['stål']],
            'reason': self.get_reason_nr(input_data)
        })
        reasons = ['Fitting', 'Valve', 'Joint', 'Seal', 'Gasket', 'Flange', 'Weld', 'Connector', 'Coupling',
                   'Corrosion',
                   'Crack']
        df_leak_true_result['label_leak'] = df_leak_true_result.apply(self.generate_label, axis=1, args=[reasons])
        return df_leak_true_result['label_leak']

    def generate_label(self, row, reasons):
        temp = 0
        if row['temp'] < 20:
            temp = 1
        elif 20 <= row['temp'] < 25:
            temp = 2
        elif row['temp'] >= 25:
            temp = 3

        tryk = 0
        if row['tryk'] < 4:
            tryk = 1
        elif 4 <= row['tryk'] < 6:
            tryk = 2
        elif row['tryk'] >= 6:
            tryk = 3

        overall = temp + tryk

        # test_values = pd.DataFrame({'temp': [23], 'tryk': [4], 'alder_år': [3], 'plast': [0], 'kobber': [1], 'stål': [0], 'reason':[9]})

        if row['kobber'] == 1:
            if overall < 4:
                if row['alder_år'] >= 3:
                    if reasons[row['reason']] == 'Valve' or reasons[row['reason']] == 'Joint':
                        return [1, 0, 0]
                    elif reasons[row['reason']] != 'Seal' or reasons[row['reason']] != 'Gasket':
                        return [0, 1, 0]
                    else:
                        return [1, 0, 0]
                elif row['alder_år'] < 3:
                    if reasons[row['reason']] != 'Flange' or reasons[row['reason']] != 'Weld':
                        return [1, 0, 0]
                    elif reasons[row['reason']] != 'Crack' or reasons[row['reason']] != 'Gasket':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]

            elif 4 <= overall <= 5:
                if row['alder_år'] >= 4:
                    if reasons[row['reason']] != 'Coupling' or reasons[row['reason']] != 'Corrosion':
                        return [0, 1, 0]
                    elif reasons[row['reason']] != 'Connector' or reasons[row['reason']] != 'Flange':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]
                elif row['alder_år'] < 4:
                    if reasons[row['reason']] == 'Flange' or reasons[row['reason']] == 'Connector':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Joint' or reasons[row['reason']] == 'Valve':
                        return [0, 0, 1]
                    else:
                        return [1, 0, 0]

            elif overall > 5:
                if row['alder_år'] >= 3:
                    if reasons[row['reason']] == 'Seal' or reasons[row['reason']] == 'Gasket':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Flange' or reasons[row['reason']] == 'Weld':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]
                elif row['alder_år'] < 3:
                    if reasons[row['reason']] == 'Connector' or reasons[row['reason']] == 'Coupling':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Corrosion' or reasons[row['reason']] == 'Crack':
                        return [0, 0, 1]
                    else:
                        return [1, 0, 0]

        if row['stål'] == 1:
            if overall < 4:
                if row['alder_år'] >= 2:
                    if reasons[row['reason']] != 'Flange' or reasons[row['reason']] != 'Weld':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Crack' or reasons[row['reason']] == 'Gasket':
                        return [0, 0, 1]
                    else:
                        return [1, 0, 0]
                elif row['alder_år'] < 2:
                    if reasons[row['reason']] == 'Coupling' or reasons[row['reason']] == 'Corrosion':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Connector' or reasons[row['reason']] == 'Flange':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]

            elif 4 <= overall <= 5:
                if row['alder_år'] >= 4:
                    if reasons[row['reason']] == 'Flange' or reasons[row['reason']] == 'Connector':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Joint' or reasons[row['reason']] == 'Valve':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]
                elif row['alder_år'] < 4:
                    if reasons[row['reason']] == 'Seal' or reasons[row['reason']] == 'Gasket':
                        return [1, 0, 0]
                    elif reasons[row['reason']] != 'Flange' or reasons[row['reason']] != 'Weld':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]

            elif overall > 5:
                if row['alder_år'] >= 4:
                    if reasons[row['reason']] == 'Valve' or reasons[row['reason']] == 'Joint':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Seal' or reasons[row['reason']] == 'Gasket':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]
                elif row['alder_år'] < 4:
                    return [0, 0, 1]

        if row['plast'] == 1:
            if overall < 4:
                if reasons[row['reason']] == 'Flange' or reasons[row['reason']] == 'Weld':
                    return [1, 0, 0]
                elif reasons[row['reason']] == 'Crack' or reasons[row['reason']] == 'Gasket':
                    return [0, 0, 1]
                else:
                    return [0, 1, 0]

            elif 4 <= overall <= 5:
                if row['alder_år'] >= 5:
                    if reasons[row['reason']] == 'Coupling' or reasons[row['reason']] == 'Corrosion':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Connector' or reasons[row['reason']] == 'Flange':
                        return [1, 0, 0]
                    else:
                        return [0, 1, 0]
                elif row['alder_år'] < 5:
                    if reasons[row['reason']] == 'Flange' or reasons[row['reason']] == 'Connector':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Joint' or reasons[row['reason']] == 'Valve':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]

            elif overall > 5:
                if row['alder_år'] >= 2:
                    if reasons[row['reason']] != 'Connector' or reasons[row['reason']] != 'Coupling':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Corrosion' or reasons[row['reason']] == 'Crack':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]
                elif row['alder_år'] < 2:
                    if reasons[row['reason']] == 'Coupling' or reasons[row['reason']] == 'Corrosion':
                        return [1, 0, 0]
                    elif reasons[row['reason']] == 'Connector' or reasons[row['reason']] == 'Flange':
                        return [0, 0, 1]
                    else:
                        return [0, 1, 0]
