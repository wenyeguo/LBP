import csv
import pickle


class File:
    def __init__(self, name):
        self.filename = name
        self.data = None

    def get_data(self):
        self.read_file_from_file()
        return self.data

    def store_data(self, data):
        self.data = data
        self.write_data_to_file()

    def read_file_from_file(self):
        with open(self.filename, 'rb') as f:
            self.data = pickle.load(f)

    def write_data_to_file(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f)

    def store_data_into_csvFile(self, rowName, data):
        with (open(self.filename, 'w', newline='') as csv_file):
            writer = csv.writer(csv_file)
            writer.writerow(rowName)
            for key, value in data.items():
                writer.writerow([key] + value)

    def store_threshold_result_into_csv_file(self, data):
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ths+', 'ths-', 'accuracy', 'recursion', 'precision', 'F1'])
            for key, val in data.items():
                t1, t2 = key.split(':')
                accuracy, precision, recall, f1 = val[0], val[1], val[2], val[3]
                writer.writerow([t1, t2, accuracy, precision, recall, f1])
