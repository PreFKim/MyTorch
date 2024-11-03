
import csv
import numpy as np
import struct
import os

# https://archive.ics.uci.edu/static/public/53/iris.zip
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
# http://lib.stat.cmu.edu/datasets/boston
# https://yann.lecun.com/exdb/mnist/

def mnist(mnist_path='./dataset/mnist'):

    def read_mnist_image(file_path):
        with open(file_path, 'rb') as f:
            # 헤더 읽기
            magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051  # IDX magic number for images
            
            # 이미지 데이터 읽기
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, num_rows, num_cols)
            
        return images

    def read_mnist_label(file_path):
        with open(file_path, 'rb') as f:
            # 헤더 읽기
            magic, num_labels = struct.unpack('>II', f.read(8))
            assert magic == 2049  # IDX magic number for labels
            
            # 레이블 데이터 읽기
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return labels
    
    train_x = read_mnist_image(os.path.join(mnist_path, "train-images-idx3-ubyte"))
    train_y = read_mnist_label(os.path.join(mnist_path, "train-labels-idx1-ubyte"))
    test_x = read_mnist_image(os.path.join(mnist_path, "t10k-images-idx3-ubyte"))
    test_y = read_mnist_label(os.path.join(mnist_path, "t10k-labels-idx1-ubyte"))

    return train_x, train_y, test_x, test_y

def titanic(csv_path='./dataset/titanic.csv'):
    titanic_data = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 헤더 읽기
        for row in reader:
            titanic_data.append(row)

    target_index = header.index('Survived')


    x = [row[:target_index] + row[target_index + 1:] for row in titanic_data]  
    y = [row[target_index] for row in titanic_data] 


    x_header = header[:target_index] + header[target_index + 1:]
    # 빈 문자열을 NaN으로 변환 + float 타입 변환
    for i, col in enumerate(x_header):
        for j in range(len(x)):
            if x[j][i] == '':
                x[j][i] = -1

            try:
                x[j][i] = float(x[j][i])
            except ValueError:
                pass  

    # 필요 없는 열 제거 
    remove_cols = ['Name', 'Ticket', 'Cabin']
    remove_indices = [x_header.index(col) for col in remove_cols if col in x_header]
    x = [[value for k, value in enumerate(row) if k not in remove_indices] for row in x]
    x_header = [col for k, col in enumerate(x_header) if k not in remove_indices]

    # 범주형 데이터 인코딩
    for i, col in enumerate(x_header):
        if isinstance(x[0][i], str):
            unique_values = list(set(row[i] for row in x if row[i] is not np.nan))
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            for j in range(len(x)):
                x[j][i] = mapping[x[j][i]]

    for i in range(len(y)):
        y[i] = int(y[i])

    return np.array(x), np.array(y)

def boston(txt_path='./dataset/boston.txt'):
    data = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '' or line == "END":
            continue
        
        row = line.split()
        numeric_row = [float(value) for value in row]
        data.append(numeric_row)


    x = np.array([row[:-1] for row in data])
    y = np.array([row[-1] for row in data])

    return x, y