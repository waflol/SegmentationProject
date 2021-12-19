import segmentation_models as sm
sm.set_framework('tf.keras')

def get_trained_model(backbone_name,model_name,classes,labels):
    preprocess_input = sm.get_preprocessing(backbone_name)
    # define network parameters
    CLASSES = classes
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    if model_name == 'FPN':
        model = sm.FPN(BACKBONE, classes=n_classes, activation=activation,input_shape=(320,320,3))
    elif model_name == 'Unet':
        model = sm.Unet(BACKBONE, classes=n_classes, activation=activation,input_shape=(320,320,3))
    else:
        model = sm.Linknet(BACKBONE, classes=n_classes, activation=activation,input_shape=(320,320,3))

    return preprocess_input,model


