import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import argparse



def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--train_labels', type=str, default='labels.txt')
    parser.add_argument('--output', type=str, default='models')
    parser.add_argument('--scaler', type=str, default='scaler')
    
    return parser.parse_args()
    
def model_train(dataset_path, model_path, scaler_path):
    # breakpoint()
    parent_directory = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    
    labels = []
    files = []
    
    for p in parent_directory:
        label_name = os.path.basename(p)
        if label_name is not labels:
            labels.append(label_name)
        for f in os.listdir(p):
            files.append(os.path.join(p, f))

    X = []
    y = []


    num = 0
    for idx, file in enumerate(files):
        
        if labels[num] in file:
            lbl = num
            data = np.load(file)
            X.extend(data)  
            y.extend([lbl] * data.shape[0])
            
        else :
            num += 1



    # breakpoint()
    X = np.array(X)
    y = np.array(y)


    with open(label_file, 'w') as f:
        for label in labels:
            f.write(label + "\n")


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train SVM model
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)


    joblib.dump(clf, model_path)

    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")






if __name__ == '__main__':
    
    args = get_parser()
    
    
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    
    dataset_path = args.dataset
    label_file = args.train_labels
    
    # if os.path.exists(args.output):
    #     os.makedirs(args.output)
    # breakpoint()
    
    # if os.path.exists(args.scaler):
    #     os.makedirs(args.scaler)    
    
    
    model_path = os.path.join(args.output ,model_filename)  
    scaler_path = os.path.join(args.scaler ,scaler_filename)
    
    

    train = model_train(dataset_path, model_path, scaler_path)

