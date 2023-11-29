import os
from flask_cors import CORS
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid

from DBConnection import connection
from PredictImage import PredictImage
from PredictSniffingpoint import PredictSniffingpoint

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}}, supports_credentials=True)
sniffpoint_predictor = PredictSniffingpoint()
predict_model_img = PredictImage()


@app.route('/predict_sniffingpoint', methods=['POST'])
def predict_sniffingpoint():
    data = request.get_json(force=True)
    prediction = sniffpoint_predictor.predict(data)
    return jsonify(prediction)


@app.route('/predict_img', methods=['POST'])
def predict_img():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        if file.content_type not in ['image/png', 'image/jpeg']:
            return jsonify({"error": "File is not a PNG or JPG image"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join('./images', filename)
        file.save(save_path)
        prediction = predict_model_img.predict(save_path)
        return jsonify(prediction)


@app.route('/get_all', methods=['GET'])
def get_all():
    try:
        conn = connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM TestObjects')

        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        print(rows)
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/update_testobject', methods=['POST'])
def update_testobject():
    try:
        conn = connection()
        cursor = conn.cursor()
        data = request.get_json(force=True)
        # Insert into TestObjects
        cursor.execute('UPDATE TestObjects SET type = ?, serialNr = ?, imagePath = ? WHERE id = ?',
                       (data.get('type'), data.get('serialNr'), data.get('imagePath'), data.get('id')))
        # Insert into SniffingPoint
        sniffing_points = data.get('sniffingPoints')
        if sniffing_points:
            for point in sniffing_points:
                # Check if the SniffingPoint already exists
                cursor.execute('SELECT id FROM SniffingPoint WHERE id = ?', (point.get('id'),))
                exists = cursor.fetchone()

                if exists:
                    # Update existing SniffingPoint
                    cursor.execute('UPDATE SniffingPoint SET name = ?, x = ?, y = ? WHERE id = ?',
                                   (point.get('name'), point.get('x'), point.get('y'), point.get('id')))
                else:
                    new_id = str(uuid.uuid4())  # Generate a new GUID
                    # Insert new SniffingPoint
                    cursor.execute('INSERT INTO SniffingPoint (name, x, y, testObjectId) VALUES (?, ?, ?, ?)',
                                   (point.get('name'), point.get('x'), point.get('y'), new_id))

        conn.commit()  # Commit the transaction
        return jsonify({"message": "TestObject added successfully"}), 201

    except Exception as e:
        conn.rollback()  # Rollback in case of error
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()


@app.route('/create_testobject', methods=['POST'])
def create_testobject():
    try:
        conn = connection()
        cursor = conn.cursor()
        data = request.get_json(force=True)
        new_id = str(uuid.uuid4())  # Generate a new GUID
        # Insert into TestObjects
        cursor.execute('INSERT INTO TestObjects (id, type, serialNr, imagePath) VALUES (?, ?, ?, ?)',
                       (new_id, data.get('type'), data.get('serialNr'), data.get('imagePath')))
        # Insert into SniffingPoint
        sniffing_points = data.get('sniffingPoints')
        if sniffing_points:
            for point in sniffing_points:
                cursor.execute('INSERT INTO SniffingPoint (name, x, y, testObjectId) VALUES (?, ?, ?, ?)',
                               (point.get('name'), point.get('x'), point.get('y'), new_id))

        conn.commit()  # Commit the transaction
        return jsonify({"message": "TestObject added successfully"}), 201

    except Exception as e:
        conn.rollback()  # Rollback in case of error
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()


@app.route('/get_testobject/<id>', methods=['GET'])
def get_testobject(id):
    if not id:
        return jsonify({"error": "Missing id parameter"}), 400

    try:
        conn = connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM TestObjects WHERE Id = ?', (id,))

        columns = [column[0] for column in cursor.description]
        row = cursor.fetchone()
        data = dict(zip(columns, row))

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/get_testobject_with_results/<id>', methods=['GET'])
def get_testobject_with_results(id):
    if not id:
        return jsonify({"error": "Missing id parameter"}), 400

    try:
        conn = connection()
        cursor = conn.cursor()

        # Fetch TestObject
        cursor.execute('SELECT * FROM TestObjects WHERE Id = ?', (id,))
        test_object_columns = [column[0] for column in cursor.description]
        test_object_data = [dict(zip(test_object_columns, row)) for row in cursor.fetchall()]

        if not test_object_data:
            return jsonify({"error": "TestObject not found"}), 404

        # Fetch associated SniffingPoints
        cursor.execute('SELECT * FROM SniffingPoint WHERE TestObjectId = ?', (id,))
        sniffing_point_columns = [column[0] for column in cursor.description]
        sniffing_points_data = [dict(zip(sniffing_point_columns, row)) for row in cursor.fetchall()]

        # Combine data
        combined_data = {
            "TestObject": test_object_data[0],
            "SniffingPoints": sniffing_points_data
        }

        return jsonify(combined_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



#
# class PredictSniffingpoint:
#
#     def __init__(self):
#         self.model, self.scaler_mean, self.scaler_std, self.one_hot_encoder = Model.load(
#             "C:\\Users\\marla\\Documents\\Datamatiker\\4. Semester\\generated_data_first_nn\\model_with_single_word_one_hot_encoded.pkl",
#             "C:\\Users\\marla\\Documents\\Datamatiker\\4. Semester\\generated_data_first_nn\\scaler_with_single_word_one_hot_encoded.pkl",
#             "C:\\Users\\marla\\Documents\\Datamatiker\\4. Semester\\generated_data_first_nn\\onehot_with_single_word_one_hot_encoded.pkl")
#
#     def predict(self, input_data):
#         values = pd.DataFrame(
#             {'temp': [23], 'tryk': [4], 'alder_år': [3], 'plast': [0], 'kobber': [1], 'stål': [0], 'reason': [5]})
#
#         values[['temp', 'tryk', 'alder_år']] = self.scale_transform(values[['temp', 'tryk', 'alder_år']])
#
#         values = self.reasons_one_hot_encode(values)
#
#         single_instance = np.array(values)
#
#         single_instance = single_instance.reshape(1, -1)
#         print(values.shape)
#
#         prediction = self.model.predict(single_instance)
#         print(prediction)
#         print(
#             f"Predicted Label: {str(round(prediction[0][0] * 100)) + '%', str(round(prediction[0][1] * 100)) + '%', str(round(prediction[0][2] * 100)) + '%'}")
#
#     def reasons_one_hot_encode(self, df_leak):
#         one_hot_encoded = self.one_hot_encoder.transform(df_leak[['reason']])
#         one_hot_df = pd.DataFrame(one_hot_encoded, columns=self.one_hot_encoder.get_feature_names_out(['reason']))
#         # Drop original 'reason' column and concatenate the one-hot encoded DataFrame
#         df_leak = df_leak.drop(columns=['reason'])
#         df_leak = pd.concat([df_leak, one_hot_df], axis=1)
#         return df_leak
#
#     def scale_transform(self, data):
#         if self.scaler_mean is None or self.scaler_std is None:
#             raise ValueError("You should first fit the normalizer on some data before scaling the data")
#         return (data - self.scaler_mean) / self.scaler_std
#
