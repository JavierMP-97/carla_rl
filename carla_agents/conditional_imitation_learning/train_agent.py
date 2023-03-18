import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import glob
import random
from shutil import copyfile
from model import create_model

@tf.function
def load_img(img_path):
    encoded_image = tf.io.read_file(img_path)
    return encoded_image

def decode_img(vgg16 = False):
    if vgg16:
        preprocess_fun = tf.keras.applications.vgg16.preprocess_input
    else:
        preprocess_fun = tf.keras.applications.mobilenet_v3.preprocess_input
    @tf.function
    def _decode_img(encoded_img):
        image = tf.io.decode_png(encoded_img)
        image = tf.cast(image, tf.float32)
        image = preprocess_fun(image)
        return image
    return _decode_img
def decode_info(mobilenetv3 = True):
    @tf.function
    def _decode_info(info_path):
        txt = tf.io.read_file(info_path)
        txt = tf.io.decode_csv(txt, [float(), int()], select_cols = [3, 6])
        label = txt[0]
        command = tf.one_hot(txt[1], 3)
        if mobilenetv3:
            command = tf.reshape(command, [1,1,3])
        return label, command
    return _decode_info


def load_data(side_images = False, mobilenetv3 = True):
    @tf.function
    def _load_data(info_path, img_path):
        label, command = decode_info(mobilenetv3)(info_path)
        encoded_image = load_img(img_path = img_path)
        return (encoded_image, command, label)
    @tf.function
    def _load_full_data(info_path, left_img_path, middle_img_path, right_img_path):
        label, command = decode_info(mobilenetv3)(info_path)
        encoded_left_image = load_img(img_path = left_img_path)
        encoded_middle_image = load_img(img_path = middle_img_path)
        encoded_right_image = load_img(img_path = right_img_path)
        return (encoded_left_image, encoded_middle_image, encoded_right_image, command, label)
    
    if side_images:
        return _load_full_data
    else:
        return _load_data

def decode_data(side_images = False, vgg16 = False):
    @tf.function
    def _decode_data(encoded_image, command, label):
        image = decode_img(vgg16)(encoded_image)
        return {"image":image, "command": command}, label
        #return (image, command), label
    @tf.function
    def _decode_full_data(encoded_left_image, encoded_middle_image, encoded_right_image, command, label):
        left_image = decode_img(vgg16)(encoded_left_image)
        middle_image = decode_img(vgg16)(encoded_middle_image)
        right_image = decode_img(vgg16)(encoded_right_image)
        image = tf.concat([left_image, middle_image, right_image], 0)
        return {"image":image, "command": command}, label
    
    if side_images:
        return _decode_full_data
    else:
        return _decode_data

def create_tf_dataset(n_instances = 400000, instances_per_map = 200000, n_maps = 2, split = 0.75, batch_size = 128, preload_images = False, side_images = False, model_name = "mobilenetv3"):

    left_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/left_rgb_*.png")
    middle_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/middle_rgb_*.png")
    right_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/right_rgb_*.png")
    info_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/info_*.txt")
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
            if side_images:
                train_dict["left"].append(left_path[ random_train_instance_idx])
                train_dict["right"].append(right_path[random_train_instance_idx])
            train_dict["middle"].append(middle_path[random_train_instance_idx])
            train_dict["info"].append(info_path[random_train_instance_idx])
        for val_instance_idx in range(val_split_per_map):
            random_val_instance_idx = (m * instances_per_map) + random_val_indexes[val_instance_idx]
            if side_images:
                test_dict["left"].append(left_path[random_val_instance_idx])
                test_dict["right"].append(right_path[random_val_instance_idx])
            test_dict["middle"].append(middle_path[random_val_instance_idx])    
            test_dict["info"].append(info_path[random_val_instance_idx])
    dataset_list = []
    for dataset_idx, path_dict in enumerate([train_dict, test_dict]):
        
        input_tuple = ()
        if side_images:
            input_tuple = (path_dict["info"], path_dict["left"], path_dict["middle"], path_dict["right"])
        else:
            input_tuple = (path_dict["info"], path_dict["middle"])

        with tf.device('CPU'):
            loaded_dataset = None #tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            decoded_dataset = None
            if preload_images:
                loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images, mobilenetv3 = (model_name == "mobilenetv3")), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                decoded_dataset = loaded_dataset.map(decode_data(side_images = side_images, vgg16 = (model_name == "vgg16")), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                decoded_dataset = decoded_dataset.cache()
                if dataset_idx == 0:
                    decoded_dataset = decoded_dataset.shuffle(int(train_split_per_map * n_maps), reshuffle_each_iteration=True)
            else:
                loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images, mobilenetv3 = (model_name == "mobilenetv3")), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).cache()
                if dataset_idx == 0:
                    loaded_dataset = loaded_dataset.shuffle(int(train_split_per_map * n_maps), reshuffle_each_iteration=True)
                decoded_dataset = loaded_dataset.map(decode_data(side_images = side_images, vgg16 = (model_name == "vgg16")), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            decoded_dataset = decoded_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        dataset_list.append(decoded_dataset)

    return dataset_list

def warmup_network(target_lr = 1e-4, num_epochs = 3, base_lr = 1e-6, base_epoch = 0):
    def _warmup_network(epoch, lr):
        if num_epochs > 0:
            new_epoch = epoch - base_epoch
            lr_ratio = target_lr / base_lr
            lr_step_mult = lr_ratio ** (1/(num_epochs))
            if new_epoch < num_epochs:
                return base_lr * (lr_step_mult ** new_epoch)
            else:
                return target_lr
        else:
            return target_lr
    return _warmup_network

SIDE_IMAGES = True
MODEL_NAME = "mobilenetv3"

n_instances = 30000
warmup_lr = 1e-6
first_train_lr = 1e-4
fine_tune_lr = 1e-4
warmup_epochs = 3
epochs = 40
freeze_level = 9
batch_size = 56
unfreeze_steps = 1
fully_unfreeze = True

EXPERIMENT_NAME = "MN3-lr-4-clipvalue.2x1" #"MN3-extra1000conv-noweights"
if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
    current_path = os.path.realpath(os.path.dirname(__file__))
    if os.path.isdir(current_path + "/logs/" + EXPERIMENT_NAME):
        print("Change the name of the experiment")
        quit()
    else:
        os.mkdir(current_path + "/logs/" + EXPERIMENT_NAME)
        copyfile(current_path + "/train_agent.py", current_path + "/logs/" + EXPERIMENT_NAME + "/train_agent.py")
        copyfile(current_path + "/model.py", current_path + "/logs/" + EXPERIMENT_NAME + "/model.py")


dataset_list = create_tf_dataset(n_instances = n_instances, side_images = SIDE_IMAGES, batch_size = batch_size, preload_images = (n_instances<=30000), model_name = MODEL_NAME)

model, base_model, model_stages = create_model(side_images = SIDE_IMAGES, model_name = MODEL_NAME)
if model_stages[0] != 0:
    model_stages = [0] + model_stages

cb = []
#cb.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5))
if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
    cb.append(tf.keras.callbacks.ModelCheckpoint(current_path + "/logs/" + EXPERIMENT_NAME + "/best_weights.hdf5", monitor = 'val_loss', save_best_only = True, save_weights_only = True))
    cb.append(tf.keras.callbacks.TensorBoard(log_dir = current_path + "/logs/" + EXPERIMENT_NAME))


#model.load_weights("model_weights.hdf5")

base_model.trainable = True

curr_epoch = 0
last_loss = 0
for unfreeze_step in range(unfreeze_steps + 1):
    base_model.trainable = True
    curr_freeze_level = freeze_level
    if fully_unfreeze:
        curr_freeze_level -= int(freeze_level * (unfreeze_step / unfreeze_steps))
    else:
        curr_freeze_level -= unfreeze_step
    if freeze_level > 0 and (not fully_unfreeze or unfreeze_step < unfreeze_steps):
        for l in base_model.layers[:model_stages[curr_freeze_level]]:
            l.trainable = False

    # optimizer=tf.keras.optimizers.experimental.SGD(learning_rate = lr, momentum = 0.9)
    # optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate = lr, momentum = 0.9, use_ema=True)
    # optimizer=tf.keras.optimizers.Adam(lr)

    curr_lr = fine_tune_lr
    clip_norm = last_loss * 1
    if unfreeze_step == 0:
        curr_lr = first_train_lr  
        clip_norm = 0.2  

    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate = curr_lr, momentum = 0.9, use_ema=True, clipvalue=clip_norm),#, use_ema = True),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])

    cb.append(tf.keras.callbacks.LearningRateScheduler(warmup_network(target_lr = curr_lr, num_epochs = warmup_epochs, base_lr=warmup_lr, base_epoch=curr_epoch)))

    h = model.fit(dataset_list[0], epochs = curr_epoch + warmup_epochs + epochs, initial_epoch = curr_epoch, validation_data = dataset_list[1], callbacks = cb)
    last_loss = h.history['loss'][-1]
    
    curr_epoch += warmup_epochs + epochs
    cb = cb[:-1]
    if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
        model.save_weights(current_path + "/logs/" + EXPERIMENT_NAME + "/weights_step_{}.hdf5".format(unfreeze_step))