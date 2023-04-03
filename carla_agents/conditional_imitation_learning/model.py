import tensorflow as tf

def create_model(side_images = False, weights_path = None, model_name = "mobilenetv3", verbose = 0):
    input_shape = ()
    if side_images:
        input_shape = (128 * 3,256,3)
    else:
        input_shape = (128,256,3)

    model_stages = []

    if model_name == "vgg16":
        model_stages = [0]
        inputs = tf.keras.Input(shape=input_shape, name = "image")
        input2 = tf.keras.Input(shape=(3), name = "command")
        base_model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape, weights = "imagenet", include_top=False)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x) #BS = 64
        #x = tf.keras.layers.Flatten()(x) #BS = 32
        x = tf.keras.layers.Concatenate()([x, input2])
        #x = tf.keras.layers.Dense(1000, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
        x = tf.keras.layers.Dense(1000, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
        #x = tf.keras.layers.Dropout(0.1)(x)
        output = tf.keras.layers.Dense(1)(x)
    
    #base_model.trainable = False
    #print(base_model.summary())
    #base_model = tf.keras.Model({"image":base_model.input}, base_model.get_layer(index = -2).output)
    if model_name == "mobilenetv3":
        model_stages = [14,23,32,42,49,66,83,98,113,128,143,165,188,211,234,257,264,270]
        inputs = tf.keras.Input(shape=input_shape, name = "image")
        input2 = tf.keras.Input(shape=(1,1,3), name = "command")
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, weights = "imagenet", include_top=True)
        main_model = tf.keras.Model(base_model.input, base_model.get_layer("global_average_pooling2d").output)
        last_stage_model = tf.keras.Model(base_model.get_layer("Conv_2").input, base_model.get_layer("dropout").output)

        x = main_model(inputs, training=False)
        x = last_stage_model(x, training=False)
        
        x = tf.keras.layers.Concatenate()([x, input2])
        x = tf.keras.layers.Conv2D(1000,1, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
        conv = tf.keras.layers.Conv2D(1,1)(x)
        output = tf.keras.layers.Flatten()(conv)
        
    if model_name == "efficientnetv2":
        model_stages = [0]
        inputs = tf.keras.Input(shape=input_shape, name = "image")
        input2 = tf.keras.Input(shape=(3), name = "command")
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_shape=input_shape, weights = "imagenet", include_top=False, pooling="avg")
        x = base_model(inputs, training=False)
        x = tf.keras.layers.Concatenate()([x, input2])
        x = tf.keras.layers.Dense(1000, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
        output = tf.keras.layers.Dense(1)(x)
        
    model = tf.keras.Model({"image":inputs, "command": input2}, output)
    #last_stage_model.trainable = True
    if verbose > 0:
        print(base_model.summary())
        print(model.summary())
    #model = tf.keras.Model((inputs, input2), outputs)
    
    if weights_path != None:
        model.load_weights(weights_path)

    return model, base_model, model_stages