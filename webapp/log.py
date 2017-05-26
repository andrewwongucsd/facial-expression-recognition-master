from keras.models import  Sequential
from keras.models import model_from_json
import _pickle as pickle
import json
import time

starttime = time.asctime(time.localtime(time.time()))
starttime = starttime.replace(":","-");
def save_model(model, dirpath='./data/results/'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dirpath+starttime+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dirpath+starttime+"model.h5")
    print("Saved model to disk")
    #with open(dirpath + starttime+'-model.txt', 'w') as f:
        #f.write(json_string)

def save_history(history, dirpath='./data/results/'):
    with open(dirpath +starttime+ '-history.txt', 'w') as f:
        json.dump(history, f);
        #f.write(str(history) + '\n')

def save_config(config, dirpath='./data/results/'):
    with open(dirpath + 'config_log.txt', 'a') as f:
        f.write(starttime + '\n')
        f.write(str(config) + '\n')

def save_result(train_val_accuracy,model, notes, conv_arch=[(32,3),(64,3),(128,3)], dense = [64,2],dirpath='./data/results/'):
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    with open(dirpath + starttime +'_train_val.txt', 'w') as f:
            f.write(str(train_acc) + '\n')
            f.write(str(val_acc) + '\n')

    endtime = time.asctime(time.localtime(time.time()))
    endtime = endtime.replace(":","-");
    with open(dirpath + 'result_log.txt', 'a') as f:
        f.write(starttime + '--' + endtime + ' comment: ' + notes + '\n' )
        f.write(str(conv_arch) + ','+ str(dense) + '\n')
        f.write('Train acc: ' + str(train_acc[-1]) +
                'Val acc: ' + str(val_acc[-1]) +
                'Ratio: ' + str(val_acc[-1]/train_acc[-1]) + '\n')

def load_model(modelName,modelWeightName,dirpath='./data/results/'):
    json_file = open(dirpath+modelName, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(dirpath+modelWeightName);
    print("Loaded model from disk")
    return loaded_model;

def load_history(filename,dirpath='./data/results/'):
    with open(dirpath + filename, encoding = 'cp1252') as data_file: 
        #decoded_data = data_file.decode('cp1252')   
        data = json.load(data_file)
    return data;