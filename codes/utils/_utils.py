import os

results = './data/results.txt'
results = open(results, 'r')
# remove blank lines
results = [line for line in results if line.strip()]
data = dict()
for i in range(0, len(results), 2):
    std = eval(results[i])
    mean = eval(results[i + 1])
    for model_name in std.keys():
        if model_name not in data:
            data[model_name] = dict()
        for key in std[model_name].keys():
            if key not in data[model_name]:
                data[model_name][key] = str
            # form in mean±std where mean and std are rounded to 3 decimal places, keep 0 if str is less than 3 decimal places
            _mean = str(round(mean[model_name][key], 3))
            _std = str(round(std[model_name][key], 3))
            if len(_mean.split('.')[1]) < 3:
                _mean += '0' * (3 - len(_mean.split('.')[1]))
            if len(_std.split('.')[1]) < 3:
                _std += '0' * (3 - len(_std.split('.')[1]))
            data[model_name][key] = _mean + '±' + _std

# for values with same key but different model_name, add suffix and prefix '__' to the highest value

for key in data['mlp'].keys():
    max_value = 0
    max_model_name = ''
    for model_name in data.keys():
        if key in data[model_name].keys():
            value = float(data[model_name][key].split('±')[0])
            if value > max_value:
                max_value = value
                max_model_name = model_name
    for model_name in data.keys():
        if key in data[model_name].keys():
            if model_name == max_model_name:
                data[model_name][key] = '__' + data[model_name][key] + '__'
            else:
                data[model_name][key] = data[model_name][key]

# print data in a table format with model_name as the first column in markdown format
model_names = list(data.keys())
keys = list(data['svm'].keys())
print('|model_name|', end='')
for key in keys:
    print(key + '|', end='')
print()
print('|---|', end='')
for key in keys:
    print('---|', end='')
print()
for model_name in model_names:
    print('|' + model_name + '|', end='')
    for key in keys:
        print(data[model_name][key] + '|', end='')
    print()




