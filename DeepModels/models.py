from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Conv2DTranspose, Flatten, Dense
)
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid


# -------------------------------------------------------------------------------
# ---------------------------- classifiction models -----------------------------
# -------------------------------------------------------------------------------


def getModel(preTrained, learning_rate=6e-5, freezeRange=0, trainbatchnorm=False, dropout=0.4):
    """
    returns a compiled, pretrained model with input_shape=(256, 256, 3)
    and binary classification output
    """
    model = preTrained(
      include_top=False,
      weights='imagenet',
      input_tensor=None,
      input_shape=(256, 256, 3),
      pooling='max', 
    ) 

    if not trainbatchnorm:
        for layer in model.layers:
            if isinstance(layer, BatchNormalization) or isinstance(layer, Dropout):
                layer.trainable = False

    for layer in model.layers[:freezeRange]:
                layer.trainable = False

    model = Sequential([model,
                        Flatten(name='top_flatten'),
                        Dense(500, activation='relu', name='dense_500'),
                        Dropout(dropout),
                        Dense(256, activation='relu', name='dense_256'),
                        Dropout(dropout),
                        Dense(1, activation=sigmoid, name='output_layer')
    ])

    model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'binary_crossentropy', metrics = ['Accuracy', 'Recall', 'Precision'])

    return model


# -------------------------------------------------------------------------------
# ---------------------------- segmentation model -------------------------------
# -------------------------------------------------------------------------------

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
  
    conv = BatchNormalization()(conv, training=False)
    if dropout_prob > 0:     
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv
    skip_connection = conv    
    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),
                 strides=(2,2),
                 padding='same')(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)
    conv = Conv2D(n_filters, 
                 3,  
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3, 
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv


def UNetCompiled(input_size=(256, 256, 3), n_filters=32, n_classes=1):
   """
   Combine both encoder and decoder blocks according to the U-Net research paper 
   Return the model as output 
   """
   inputs = Input(input_size)

   cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
   cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
   cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
   cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
   cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 

   ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
   ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
   ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
   ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

   conv9 = Conv2D(n_filters,
               3,
               activation='relu',
               padding='same',
               kernel_initializer='he_normal')(ublock9) 

   conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
   
   model = Model(inputs=inputs, outputs=conv10)

   model.compile(optimizer = Adam(learning_rate = 1e-3), loss = 'binary_crossentropy', metrics = ['Accuracy'])

   return model
