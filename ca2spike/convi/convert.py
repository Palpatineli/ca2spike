from keras import backend as K
from keras.models import Model
from keras.utils.conv_utils import convert_kernel

def convert_model(model: Model, from_: str) -> Model:
    """Performs conversion between theano and tensorflow conv net weights."""
    to_ = K.backend()
    if from_ == to_:
        return model
    elif from_ == "tensorflow" and to_ == "theano":
        return _tf2th(model)
    elif from_ == "theano" and to_ == "tensorflow":
        return _th2tf(model)
    else:
        raise NotImplemented("Conversion from {} to {} is not implemented!".format(from_, to_))

def _tf2th(model: Model) -> Model:
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            K.set_value(layer.W, converted_w)
    return model

def _th2tf(model: Model) -> Model:
    import tensorflow as tf
    ops = list()
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)
    K.get_session().run(ops)
