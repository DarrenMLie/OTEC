from csv import reader
import pandas as pd
import numpy as np

fileName = 'practiceDataOneYears.csv'
data = [['', 'Probe', 'Latitude', 'Longitude', 'Month', 'Day', 'Year', 'Depth', 'Temperature']]
with open(fileName, 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    try:

        for row in csv_reader:
            try:
                if (int(float(row[5])) % 9) == 0:
                    data.append(row)
            except ValueError:
                continue
    except:
        print(f'Row #{csv_reader.line_num} {row}')
df = pd.DataFrame(data)
df.dropna(inplace=True)
df.to_csv('PracticeOneYear.csv', sep=',', index=False, header = False)
