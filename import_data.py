import pandas as pd
import csv
import itertools

def import_data(data_dir = 'data'):
    relevant_lines = [66, 67, 72, 78, 111]
    param_names = ['PNS index', 'SNS index', 'Mean RR', 'RMSSD', 'LF/HF']
    case_id = 3
    phase_name = 'cognitiveLoad-phases-5min'
    role_name = 'Anes'
    file_name = f"{data_dir}/Case{case_id:02d}/{phase_name}/3296_{case_id:02d}_{role_name}_hrv.csv"
    dataset = []
    with open(file_name, 'r') as f:
        r = csv.reader(f)
        for i in itertools.count(start=1):
            if i > relevant_lines[-1]:
                break
            elif i not in relevant_lines:
                next(r)
            else:
                print(i)
                try:
                    row = next(r)
                    row = [x.replace(' ', '') for x in row]
                    row = [x for x in row if x!=''][1:]
                    row = [float(x) if x!='NaN' else None for x in row]
                    dataset.append(row)
                except StopIteration as e:
                    print("End of file reached")
    dataset[-1] = dataset[-1][::2]
    return dataset

dataset = import_data()
print(dataset[-1])
