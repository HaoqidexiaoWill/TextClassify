import pandas as pd

def sampling(read_file_path, write_file_path, neg_rate):
    with open(read_file_path, 'r') as read_file:
        with open(write_file_path, 'w') as write_file:
            count = 0
            line = 1
            while line:
                line = read_file.readline()
                try:
                    new_line = line.split()
                    if new_line[-1] == '1':
                        count = 0
                    else:
                        count += 1
                    if count <= neg_rate:
                        write_file.write(line)
                except:
                    print(line)




# sampling('dev.txt', 'dev_1_3.txt', 3)
sampling('train.txt', 'train_1_3.txt', 3)