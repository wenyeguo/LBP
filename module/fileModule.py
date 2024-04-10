import csv
import pickle


class File:
    def __init__(self, name):
        self.name = name
        self.data = None

    def get_data(self):
        self.read_file_from_file()
        return self.data

    def store_data(self, data):
        self.data = data
        self.write_data_to_file()

    def read_file_from_file(self):
        with open(self.name, 'rb') as f:
            self.data = pickle.load(f)

    def write_data_to_file(self):
        with open(self.name, 'wb+') as f:
            pickle.dump(self.data, f)

    def store_data_into_csvFile(self, rowName, data):
        with (open(self.name, 'w', newline='') as csv_file):
            writer = csv.writer(csv_file)
            writer.writerow(rowName)
            for key, value in data.items():
                writer.writerow([key] + value)
