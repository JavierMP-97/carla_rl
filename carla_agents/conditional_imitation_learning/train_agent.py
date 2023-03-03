
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import glob
import cv2
import numpy as np
import random

@tf.function
def load_img(img_path):
    encoded_image = tf.io.read_file(img_path)
    return encoded_image
@tf.function
def decode_img(encoded_img):
    image = tf.io.decode_png(encoded_img)
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image
@tf.function
def decode_info(info_path):
    txt = tf.io.read_file(info_path)
    txt = tf.io.decode_csv(txt, [float(), int()], select_cols = [3, 6])
    label = txt[0]
    command = tf.one_hot(txt[1], 3)
    command = tf.reshape(command, [1,1,3])
    return label, command


def load_data(side_images = False):
    @tf.function
    def _load_data(info_path, img_path):
        label, command = decode_info(info_path)
        encoded_image = load_img(img_path = img_path)
        return (encoded_image, command, label)
    @tf.function
    def _load_full_data(info_path, left_img_path, middle_img_path, right_img_path):
        label, command = decode_info(info_path)
        encoded_left_image = load_img(img_path = left_img_path)
        encoded_middle_image = load_img(img_path = middle_img_path)
        encoded_right_image = load_img(img_path = right_img_path)
        return (encoded_left_image, encoded_middle_image, encoded_right_image, command, label)
    
    if side_images:
        return _load_full_data
    else:
        return _load_data

def decode_data(side_images = False):
    @tf.function
    def _decode_data(encoded_image, command, label):
        image = decode_img(encoded_image)
        return {"image":image, "command": command}, label
        #return (image, command), label
    @tf.function
    def _decode_full_data(encoded_left_image, encoded_middle_image, encoded_right_image, command, label):
        left_image = decode_img(encoded_left_image)
        middle_image = decode_img(encoded_middle_image)
        right_image = decode_img(encoded_right_image)
        image = tf.concat([left_image, middle_image, right_image], 0)
        return {"image":image, "command": command}, label
    
    if side_images:
        return _decode_full_data
    else:
        return _decode_data

def create_tf_dataset(n_instances = 400000, instances_per_map = 200000, n_maps = 2, split = 0.75, batch_size = 128, preload_images = False, side_images = False):

    left_path = glob.glob("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\left_rgb_*.png")
    middle_path = glob.glob("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\middle_rgb_*.png")
    right_path = glob.glob("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\right_rgb_*.png")
    info_path = glob.glob("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\info_*.txt")
    train_dict = {"left":[], "middle":[], "right": [], "info": []}
    test_dict = {"left":[], "middle":[], "right": [], "info": []}

    if n_instances > instances_per_map * n_maps:
        n_instances = instances_per_map * n_maps

    train_split_per_map = int(split * (n_instances / n_maps))
    val_split_per_map = int((1 - split) * (n_instances / n_maps))

    random_train_indexes = list(range(0, int(instances_per_map * split)))
    random_val_indexes = list(range(int(instances_per_map * split), instances_per_map))
    random.shuffle(random_train_indexes)
    random.shuffle(random_val_indexes)  
    for m in range(n_maps):
        for train_instance_idx in range(train_split_per_map):
            random_train_instance_idx = (m * instances_per_map) + random_train_indexes[train_instance_idx]
            train_dict["left"].append(left_path[ random_train_instance_idx])
            train_dict["middle"].append(middle_path[random_train_instance_idx])
            train_dict["right"].append(right_path[random_train_instance_idx])
            train_dict["info"].append(info_path[random_train_instance_idx])
        for val_instance in range(val_split_per_map):
            random_val_instance_idx = (m * instances_per_map) + random_val_indexes[val_instance]
            test_dict["left"].append(left_path[random_val_instance_idx])
            test_dict["middle"].append(middle_path[random_val_instance_idx])
            test_dict["right"].append(right_path[random_val_instance_idx])
            test_dict["info"].append(info_path[random_val_instance_idx])
    dataset_list = []
    for dataset_idx, path_dict in enumerate([train_dict, test_dict]):
        
        input_tuple = ()
        if side_images:
            input_tuple = (path_dict["info"], path_dict["left"], path_dict["middle"], path_dict["right"])
        else:
            input_tuple = (path_dict["info"], path_dict["middle"])

        with tf.device('CPU'):
            loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            if preload_images:
                loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                decoded_dataset = loaded_dataset.map(decode_data(side_images = side_images), num_parallel_calls=12, deterministic=False)
                decoded_dataset = decoded_dataset.cache()
                if dataset_idx == 0:
                    decoded_dataset = decoded_dataset.shuffle(int(train_split_per_map * n_maps), reshuffle_each_iteration=True)
            else:
                loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).cache()
                if dataset_idx == 0:
                    loaded_dataset = loaded_dataset.shuffle(int(train_split_per_map * n_maps), reshuffle_each_iteration=True)
                decoded_dataset = loaded_dataset.map(decode_data(side_images = side_images), num_parallel_calls=12, deterministic=False)
            decoded_dataset = decoded_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        dataset_list.append(decoded_dataset)

    return dataset_list

def create_model(side_images = False, weights_path = None):
    input_shape = ()
    if side_images:
        input_shape = (128 * 3,256,3)
    else:
        input_shape = (128,256,3)

    #base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, weights = "imagenet", include_top=False)
    #base_model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape, weights = "imagenet", include_top=False)
    inputs = tf.keras.Input(shape=input_shape, name = "image")
    input2 = tf.keras.Input(shape=(1,1,3), name = "command")
    #x = base_model(inputs, training=False)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x) #BS = 64
    #x = tf.keras.layers.Flatten()(x) #BS = 32
    #x = tf.keras.layers.Concatenate()([x, input2])
    #x = tf.keras.layers.Dense(1000, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
    #x = tf.keras.layers.Dense(1000, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
    #x = tf.keras.layers.Dropout(0.1)(x)
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=input_shape, weights = "imagenet", include_top=True) #
    #base_model.trainable = False
    #print(base_model.summary())
    #base_model = tf.keras.Model({"image":base_model.input}, base_model.get_layer(index = -2).output)
    main_model = tf.keras.Model(base_model.input, base_model.get_layer("global_average_pooling2d").output)
    last_stage_model = tf.keras.Model(base_model.get_layer("Conv_2").input, base_model.get_layer("dropout").output)

    print(base_model.summary())
    
    #x = base_model(inputs, training=False)
    x = main_model(inputs, training=False)
    x = last_stage_model(x, training=False)
    
    x = tf.keras.layers.Concatenate()([x, input2])
    x = tf.keras.layers.Conv2D(1000,1, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
    conv = tf.keras.layers.Conv2D(1,1)(x)
    output = tf.keras.layers.Flatten()(conv)

    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(1000, activation = "gelu", kernel_initializer = tf.keras.initializers.HeNormal())(x)
    #output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model({"image":inputs, "command": input2}, output)
    #last_stage_model.trainable = True
    print(model.summary())
    #model = tf.keras.Model((inputs, input2), outputs)
    

    if weights_path != None:
        model.load_weights(weights_path)

    return model, base_model

def warmup_network(target_lr = 1e-4, num_epochs = 3, base_lr = 1e-6):
    def _warmup_network(epoch, lr):
        lr_ratio = target_lr / base_lr
        lr_step_mult = lr_ratio ** (1/(num_epochs))
        if epoch < num_epochs:
            return base_lr * (lr_step_mult ** epoch)
        else:
            return target_lr
    return _warmup_network

SIDE_IMAGES = True

EXPERIMENT_NAME = "MN3-extra1000conv-noweights"
if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
    current_path = os.path.realpath(os.path.dirname(__file__))
    if os.path.isdir(current_path + "/logs/" + EXPERIMENT_NAME):
        print("Change the name of the experiment")
        quit()
    else:
        os.mkdir(current_path + "/logs/" + EXPERIMENT_NAME)

dataset_list = create_tf_dataset(n_instances = 30000, side_images = SIDE_IMAGES, batch_size = 64, preload_images = True)

model, base_model = create_model(side_images = SIDE_IMAGES)

warmup_lr = 1e-6
lr = 1e-4
fine_tune_lr = 1e-5
warmup_epochs = 3
epochs = 10
fine_tune_epochs = 10

cb = []
#cb.append(tf.keras.callbacks.ModelCheckpoint("model_weights4.hdf5", monitor = 'val_loss', save_best_only = True, save_weights_only = True))
cb.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5))
if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
    cb.append(tf.keras.callbacks.TensorBoard(log_dir = current_path + "/logs/" + EXPERIMENT_NAME))
cb.append(tf.keras.callbacks.LearningRateScheduler(warmup_network(target_lr = 1e-4, num_epochs = warmup_epochs, base_lr=warmup_lr)))

model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

model.fit(dataset_list[0], epochs=warmup_epochs + epochs, validation_data = dataset_list[1], callbacks = cb)

#model.load_weights("model_weights.hdf5")

base_model.trainable = True
cb = cb[:-1]
#cb.append(tf.keras.callbacks.ModelCheckpoint("model_weights5.hdf5", monitor = 'val_loss', save_best_only = True, save_weights_only = True))

#model.compile(optimizer=tf.keras.optimizers.Adam(fine_tune_lr),
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

model.fit(dataset_list[0], epochs=warmup_epochs + epochs + fine_tune_epochs, initial_epoch = warmup_epochs + epochs, validation_data = dataset_list[1], callbacks = cb)
