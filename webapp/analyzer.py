
class Engine:

    def __init__(self):
        load_CNN_model()
        get_accuracy_training_dataset()

    def load_CNN_model(self):
        # load json and create model
        json_file = open('data/results/model.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into model
        model.load_weights('data/weights/model.h5')

    def get_accuracy_training_dataset(self):
        # import private test:
        X_fname = 'data/X_testing.npy'
        y_fname = 'data/y_testing.npy'
        X = np.load(X_fname)
        y = np.load(y_fname) # data/y_training.npy
        print('Training set')
        y_labels = [np.argmax(lst) for lst in y]
        counts = np.bincount(y_labels)
        labels = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print(zip(labels, counts))
        # evaluate model on private test set
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        score = model.evaluate(X, y, verbose=0)
        print("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    
