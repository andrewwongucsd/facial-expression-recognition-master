from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import random
import sys

# fer2013 dataset:
# Training       28709
# PrivateTest     3589
# PublicTest      3589

# emotion labels from FER2013:
emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emo     = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']

def reconstruct(pix_str, size=(48,48)):
    #print("Pixel: ",len(pix_str.split(" ")));
    #pix_arr = np.array(map(int, pix_str.split(" ")))
    pix_arr = np.fromstring(pix_str, dtype=int, sep=" ")
    return pix_arr.reshape(size)

def emotion_count(y_train, classes, verbose=True):
    emo_classcount = {}
    print ('Disgust classified as Angry')
    y_train.loc[y_train == 1] = 0
    classes.remove('Disgust')
    for new_num, _class in enumerate(classes):
        y_train.loc[(y_train == emotion[_class])] = new_num
        class_count = sum(y_train == (new_num))
        if verbose:
            print ('{}: {} with {} samples'.format(new_num, _class, class_count))
        emo_classcount[_class] = (new_num, class_count)
    return y_train.values, emo_classcount

def load_data(sample_split=0.3, usage='Training',usage2 = '', to_cat=True, verbose=True,
              classes=['Angry','Happy'], filepath='./data/fer2013.csv'):
    df = pd.read_csv(filepath)
    # print df.tail()
    # print df.Usage.value_counts()
    #df = df[(df.Usage == 'Training' | (df.Usage == 'PublicTest')]
    df = df[(df.Usage == usage) | (df.Usage == usage2)]
    frames = []
    classes.append('Disgust')
    for _class in classes:
        class_df = df[df['emotion'] == emotion[_class]]
        frames.append(class_df)
    data = pd.concat(frames, axis=0)
    #print(list(data.index))
    #data = data.sample( int(len(data)*sample_split))
    #print(rows);
    #data = data.ix[random.sample(data.index, int(len(data)*sample_split))]
    #data = data.ix[rows]
    #data = rows
    #print("Pixel length ",len(data['pixels'].iloc[0].split(" ")))
    print ('{} set for {}: {}'.format(usage, classes, data.shape))
    data['pixels'] = data.pixels.apply(lambda x: reconstruct(x))
    x = np.array([mat for mat in data.pixels]) # (n_samples, img_width, img_height)
    X_train = x.reshape(-1, 1, x.shape[1], x.shape[2])
    y_train, new_dict = emotion_count(data.emotion, classes, verbose)
    print (new_dict)
    y_train = to_categorical(y_train)
    return X_train, y_train, new_dict

def save_data(X_train, y_train, fname='', folder='./data/'):
    np.save(folder + 'X' + fname, X_train)
    np.save(folder + 'y' + fname, y_train)

if __name__ == '__main__':
    # makes the numpy arrays ready to use:
    print ('Generating data...')
    emo = ['Angry', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                            to_cat=False,
                                           classes=emo,
                                           usage='Training',
                                           verbose=True)
    save_data(X_train, y_train, fname='_train')
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                            to_cat=False,
                                           classes=emo,
                                           usage='PrivateTest',
                                           verbose=True)
    save_data(X_train, y_train, fname='_test_private')
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                            to_cat=False,
                                           classes=emo,
                                           usage='PublicTest',
                                           verbose=True)
    X_train, y_train, emo_dict = load_data(sample_split=1.0,
                                            to_cat=False,
                                           classes=emo,
                                           usage='Training',
                                           usage2='PrivateTest',
                                           verbose=True)
    save_data(X_train, y_train, fname='_train_full')
    print ('Saving...')
    save_data(X_train, y_train, fname='_test_public')
    print (X_train.shape)
    print (y_train.shape)
    print ('Done!')