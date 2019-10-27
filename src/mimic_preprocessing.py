import pickle
import copy
import numpy as np


def load_data():
    print('Loading files ...')
    vitals = pickle.load(open('vitals_records.p', 'rb'))
    adm_info = pickle.load(
        open('adm_type_los_mortality.p', 'rb'))
    print('Loading Done!')
    adm_id = [record[0] for record in adm_info]
    adm_id_needed = [record[0] for record in adm_info if record[2] >= 48]

    vitals_dict = {}
    for i in range(len(adm_id)):
        vitals_dict[adm_id[i]] = vitals[i]

    vitals = [vitals_dict[x] for x in adm_id_needed]
    label = [rec[3] for x in adm_id_needed for rec in adm_info if x == rec[0]]
    print(len(vitals), len(label))
    return vitals, label


def trim_los(data, length_of_stay):
    num_features = 12  # final features (excluding EtCO2)
    max_length = 2881  # maximum length of time stamp
    a = np.zeros((len(data), num_features, max_length))
    timestamps = []
    for i in range(len(data)):
        l = []
        for elem in data[i][7]:
            if elem[1] != None:
                # Fahrenheit->Celcius conversion
                tup = (elem[0], elem[1]*1.8 + 32)
                data[i][6].append(tup)

        for elem in data[i][10]:
            data[i][9].append(elem)
        for elem in data[i][11]:
            data[i][9].append(elem)

        # removing duplicates and EtCO2
        del data[i][5]
        del data[i][6]
        del data[i][8]
        del data[i][8]

        # taking union of all time stamps,
        # we don't actually need this for our model
        for j in range(num_features):
            for k in range(len(data[i][j])):
                l.append(data[i][j][k][0])

        # keeping only unique elements
        TS = []
        for j in l:
            if j not in TS:
                TS.append(j)
        TS.sort()

        # extracting first 48hr data
        T = copy.deepcopy(TS)
        TS = []
        for t in T:
            if (t - T[0]).total_seconds()/3600 <= length_of_stay:
                TS.append(t)
        T = []
        timestamps.append(TS)
        for j in range(num_features):
            c = 0
            for k in range(len(TS)):
                if c < len(data[i][j]) and TS[k] == data[i][j][c][0]:
                    if data[i][j][c][1] is None:
                        a[i, j, k] = -100  # missing data
                    elif (data[i][j][c][1] == 'Normal <3 secs' or
                          data[i][j][c][1] == 'Normal <3 Seconds' or
                          data[i][j][c][1] == 'Brisk'):
                        a[i, j, k] = 1
                    elif (data[i][j][c][1] == 'Abnormal >3 secs' or
                          data[i][j][c][1] == 'Abnormal >3 Seconds' or
                          data[i][j][c][1] == 'Delayed'):
                        a[i, j, k] = 2
                    elif (data[i][j][c][1] == 'Other/Remarks' or
                          data[i][j][c][1] == 'Comment'):
                        a[i, j, k] = -100  # missing data
                    else:
                        a[i, j, k] = data[i][j][c][1]

                    c += 1
                else:
                    a[i, j, k] = -100  # missing data

    return a, timestamps


def fix_input_format(x, T):
    """Return the input in the proper format
    x: observed values
    M: masking, 0 indicates missing values
    delta: time points of observation
    """
    timestamp = 200
    num_features = 12

    # trim time stamps higher than 200
    for i in range(len(T)):
        if len(T[i]) > timestamp:
            T[i] = T[i][:timestamp]

    x = x[:, :, :timestamp]
    M = np.zeros_like(x)
    delta = np.zeros_like(x)
    print(x.shape, len(T))

    for t in T:
        for i in range(1, len(t)):
            t[i] = (t[i] - t[0]).total_seconds()/3600.0
        if len(t) != 0:
            t[0] = 0

    # count outliers and negative values as missing values
    # M = 0 indicates missing value
    # M = 1 indicates observed value
    # now since we have mask variable, we don't need -100
    M[x > 500] = 0
    x[x > 500] = 0.0
    M[x < 0] = 0
    x[x < 0] = 0.0
    M[x > 0] = 1

    for i in range(num_features):
        for j in range(x.shape[0]):
            for k in range(len(T[j])):
                delta[j, i, k] = T[j][k]

    return x, M, delta

