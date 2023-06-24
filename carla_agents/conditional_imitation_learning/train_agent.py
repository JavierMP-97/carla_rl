import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'    

import tensorflow as tf
'''
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
'''
import glob
import random
from shutil import copyfile
from model import create_model
import gc

@tf.function
def load_img(img_path):
    encoded_image = tf.io.read_file(img_path)
    return encoded_image

def decode_img(model_name = "mobilenetv3"):
    if model_name == "vgg16":
        preprocess_fun = tf.keras.applications.vgg16.preprocess_input
    elif model_name == "mobilenetv3":
        preprocess_fun = tf.keras.applications.mobilenet_v3.preprocess_input
    elif model_name == "efficientnetv2":
        preprocess_fun = tf.keras.applications.efficientnet_v2.preprocess_input
    @tf.function
    def _decode_img(encoded_img):
        image = tf.io.decode_png(encoded_img)
        image = tf.cast(image, tf.float32)
        image = preprocess_fun(image)
        return image
    return _decode_img
def decode_info(conv_output = True):
    @tf.function
    def _decode_info(info_path):
        txt = tf.io.read_file(info_path)
        txt = tf.io.decode_csv(txt, [float(), int()], select_cols = [3, 6])
        label = txt[0]
        command = tf.one_hot(txt[1], 3)
        if conv_output:
            command = tf.reshape(command, [1,1,3])
        return label, command
    return _decode_info


def load_data(side_images = False, conv_output = True):
    @tf.function
    def _load_data(info_path, img_path):
        label, command = decode_info(conv_output)(info_path)
        encoded_image = load_img(img_path = img_path)
        return (encoded_image, command, label)
    @tf.function
    def _load_full_data(info_path, left_img_path, middle_img_path, right_img_path):
        label, command = decode_info(conv_output)(info_path)
        encoded_left_image = load_img(img_path = left_img_path)
        encoded_middle_image = load_img(img_path = middle_img_path)
        encoded_right_image = load_img(img_path = right_img_path)
        return (encoded_left_image, encoded_middle_image, encoded_right_image, command, label)
    
    if side_images:
        return _load_full_data
    else:
        return _load_data

def decode_data(side_images = False, model_name = "mobilenetv3"):
    @tf.function
    def _decode_data(encoded_image, command, label):
        image = decode_img(model_name)(encoded_image)
        return {"image":image, "command": command}, label
        #return (image, command), label
    @tf.function
    def _decode_full_data(encoded_left_image, encoded_middle_image, encoded_right_image, command, label):
        left_image = decode_img(model_name)(encoded_left_image)
        middle_image = decode_img(model_name)(encoded_middle_image)
        right_image = decode_img(model_name)(encoded_right_image)
        image = tf.concat([left_image, middle_image, right_image], 0)
        return {"image":image, "command": command}, label
    
    if side_images:
        return _decode_full_data
    else:
        return _decode_data

def create_tf_dataset(n_instances = 400000, instances_per_map = 200000, n_maps = 2, split = 0.75, batch_size = 128, preload_images = False, side_images = False, model_name = "mobilenetv3"):

    current_instances_per_map = int(n_instances / n_maps)

    if current_instances_per_map > instances_per_map:
        current_instances_per_map = instances_per_map
    

    if side_images:
        left_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/left_rgb_*.png")
        right_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/right_rgb_*.png")
        left_path.sort()
        right_path.sort()
    middle_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/middle_rgb_*.png")
    info_path = glob.glob("D:/PC-Javier/Desktop/Carla14/carla_rl/log/agente/info_*.txt")
    middle_path.sort()
    info_path.sort()  
    
    middle_path_per_map = [middle_path[i:i+instances_per_map] for i in range(0, len(middle_path), instances_per_map)]
    info_path_per_map = [info_path[i:i+instances_per_map] for i in range(0, len(info_path), instances_per_map)]
    if side_images:
        left_path_per_map = [left_path[i:i+instances_per_map] for i in range(0, len(left_path), instances_per_map)]
        right_path_per_map = [right_path[i:i+instances_per_map] for i in range(0, len(right_path), instances_per_map)]
          
    for map_idx, map_info_path in enumerate(info_path_per_map):
        len_map_info_path = len(map_info_path)
        for i in range(len_map_info_path):
            curr_idx = len_map_info_path - i -1
            episode_step = int(map_info_path[curr_idx].split("_")[-1].split(".")[0])
            if i == 0:
                middle_path_per_map[map_idx].pop()
                if side_images:
                    left_path_per_map[map_idx].pop()
                    right_path_per_map[map_idx].pop()
            if episode_step == 0:
                info_path_per_map[map_idx].pop(curr_idx)
                if curr_idx != 0:
                    middle_path_per_map[map_idx].pop(curr_idx-1)
                    if side_images:
                        left_path_per_map[map_idx].pop(curr_idx-1)
                        right_path_per_map[map_idx].pop(curr_idx-1)     
    
    train_dict = {"left":[], "middle":[], "right": [], "info": []}
    test_dict = {"left":[], "middle":[], "right": [], "info": []}

    for m in range(n_maps):
        random_indexes = list(range(0, len(middle_path_per_map[m])))
        random.shuffle(random_indexes)
        random_indexes = random_indexes[:current_instances_per_map]
        random_train_indexes = random_indexes[:int(len(random_indexes) * split)]
        random_val_indexes = random_indexes[int(len(random_indexes) * split):]
        for train_idx in random_train_indexes:
            if side_images:
                train_dict["left"].append(left_path_per_map[m][train_idx])
                train_dict["right"].append(right_path_per_map[m][train_idx])
            train_dict["middle"].append(middle_path_per_map[m][train_idx])
            train_dict["info"].append(info_path_per_map[m][train_idx])
        for val_idx in random_val_indexes:
            if side_images:
                test_dict["left"].append(left_path_per_map[m][val_idx])
                test_dict["right"].append(right_path_per_map[m][val_idx])
            test_dict["middle"].append(middle_path_per_map[m][val_idx])  
            test_dict["info"].append(info_path_per_map[m][val_idx])
    
    for idx, _ in enumerate(train_dict["info"]):
        middle_idx = int(train_dict["middle"][idx].split("_")[-1].split(".")[0])
        left_idx = int(train_dict["left"][idx].split("_")[-1].split(".")[0])
        right_idx = int(train_dict["right"][idx].split("_")[-1].split(".")[0])
        info_idx = int(train_dict["info"][idx].split("_")[-1].split(".")[0])
        if ((info_idx-1) != middle_idx) or ((info_idx-1) != left_idx) or ((info_idx-1) != right_idx):
            print("Error")
            print(info_idx)
            print(middle_idx)
            print(left_idx)
            print(right_idx)
            
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
                loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images, conv_output = (model_name == "mobilenetv3")), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                decoded_dataset = loaded_dataset.map(decode_data(side_images = side_images, model_name = model_name), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                decoded_dataset = decoded_dataset.cache()
                if dataset_idx == 0:
                    decoded_dataset = decoded_dataset.shuffle(int(current_instances_per_map * n_maps), reshuffle_each_iteration=True)
            else:
                loaded_dataset = tf.data.Dataset.from_tensor_slices(input_tuple).map(load_data(side_images = side_images, conv_output = (model_name == "mobilenetv3")), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).cache()
                if dataset_idx == 0:
                    loaded_dataset = loaded_dataset.shuffle(int(current_instances_per_map * n_maps), reshuffle_each_iteration=True)
                decoded_dataset = loaded_dataset.map(decode_data(side_images = side_images, model_name = model_name), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            decoded_dataset = decoded_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        dataset_list.append(decoded_dataset)

    return dataset_list

def custom_learning_rate_schedule(base_lr = 1e-4, warmup_epochs = 0, warmup_lr = 1e-6, initial_epoch = 0, training_epochs = 20, target_lr = 1e-4, tf_schedule = None):
    def _custom_learning_rate_schedule(epoch, lr):
        new_epoch = epoch - initial_epoch
        if new_epoch < warmup_epochs:
            lr_ratio = base_lr / warmup_lr
            lr_step_mult = lr_ratio ** (1/(warmup_epochs))
            return warmup_lr * (lr_step_mult ** new_epoch)
        else:
            new_epoch = new_epoch - warmup_epochs
            if tf_schedule != None:
                lr_ratio =  target_lr / base_lr
                lr_step_mult = lr_ratio ** (1/(training_epochs))
                return base_lr * (lr_step_mult ** new_epoch)
            else:
                return tf_schedule(new_epoch)
        
    return _custom_learning_rate_schedule

class SaveBestWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, filepath, base_model, monitor = "val_loss", verbose = 0):
        super(SaveBestWeightsCallback, self).__init__()
        self.filepath = filepath
        self.best_val_loss = None
        self.monitor = monitor
        self.verbose = verbose
        self.base_model = base_model
        
    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs[self.monitor]
        if self.best_val_loss == None or current_val_loss < self.best_val_loss:
            # Save the current weights as the best weights so far
            self.best_val_loss = current_val_loss

            layer_trainable_states = []
            # Save the current trainable state of every layer
            for layer in self.base_model.layers:
                layer_trainable_states.append(layer.trainable)

            # Set the trainable state of every layer to True
            for layer in self.base_model.layers:
                layer.trainable = True

            # Save the model weights
            self.model.save_weights(self.filepath)

            # Return the trainable state of every layer to the previous value
            for i, layer in enumerate(self.base_model.layers):
                layer.trainable = layer_trainable_states[i]

            if self.verbose > 0:
                print('Saved best weights at epoch', epoch)
            
SIDE_IMAGES = True
MODEL_NAME = "efficientnetv2"

n_instances = 30000
warmup_lr = 1e-6
base_lr = 3e-5
target_lr = 3e-6
warmup_epochs = 3
epochs = 100
freeze_level = 0
batch_size = 16
unfreeze_steps = 0
fully_unfreeze = False

weights_path = None #"D:/PC-Javier/Desktop/Carla14/carla_rl/carla_agents/conditional_imitation_learning/logs/EN2_lr3e-5-6_freeze4/weights_step_0.hdf5"
starting_step = 1
if weights_path == None:
    starting_step = 0
if starting_step > unfreeze_steps:
    starting_step = unfreeze_steps

EXPERIMENT_NAME = "EN2_lr3e-5-6-cosine" 
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

model, base_model, model_stages = create_model(side_images = SIDE_IMAGES, model_name = MODEL_NAME, weights_path = weights_path, verbose=0)
if model_stages[0] != 0:
    model_stages = [0] + model_stages

cb = []
#cb.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5))
cb.append(tf.keras.callbacks.LambdaCallback(on_epoch_end = lambda epoch, logs: gc.collect()))
if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
    cb.append(SaveBestWeightsCallback(current_path + "/logs/" + EXPERIMENT_NAME + "/best_weights.hdf5", base_model=base_model, monitor = 'val_loss', verbose=1))
    cb.append(tf.keras.callbacks.TensorBoard(log_dir = current_path + "/logs/" + EXPERIMENT_NAME))


#model.load_weights("model_weights.hdf5")

base_model.trainable = True
curr_lr = 0
clip_norm = 0
last_loss = 0
if unfreeze_steps < 0:
    print("unfreeze_steps must be at least 0 (unfreeze_steps set to 0)")
    unfreeze_steps = 0
if freeze_level <= 0:
    print("if every layer is trainable you don't have to take steps to unfreeze them (unfreeze_steps set to 0)")
    unfreeze_steps = 0
if unfreeze_steps > freeze_level:
    print("the maximum number of steps to unfreeze the layers is the number of layers frozen (unfreeze_steps set to freeze_level)")
    unfreeze_steps = freeze_level
if fully_unfreeze and freeze_level > 0 and unfreeze_steps <= 0:
    print("if the network must be fully unfrozen by the end of the training, it should take at least one step to unfreeze it (unfreeze_steps set to 1)")
    unfreeze_steps = 1
for unfreeze_step in range(starting_step, unfreeze_steps + 1):
    curr_epoch = unfreeze_step * (warmup_epochs + epochs)
    base_model.trainable = True
    curr_freeze_level = freeze_level
    if fully_unfreeze:
        if unfreeze_steps != 0:
            curr_freeze_level -= int(freeze_level * (unfreeze_step / unfreeze_steps))
        else:
            curr_freeze_level = 0
    else:
        curr_freeze_level -= unfreeze_step
        
    if freeze_level > 0 and (not fully_unfreeze or unfreeze_step < unfreeze_steps):
        for l in base_model.layers[:model_stages[curr_freeze_level]]:
            l.trainable = False

    if unfreeze_step == starting_step:  
        clip_norm = 0.2
        #clip_norm = 0.0034
        #clip_norm = 0.0001
    else:
        clip_norm = last_loss * 1

    print(curr_freeze_level)
    
    # optimizer=tf.keras.optimizers.experimental.Lion(learning_rate = base_lr, clipvalue=clip_norm)
    model.compile(tf.keras.optimizers.experimental.RMSprop(learning_rate = base_lr, momentum = 0.9, use_ema=True, clipvalue=clip_norm),#, use_ema = True),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    cb.append(tf.keras.callbacks.LearningRateScheduler(custom_learning_rate_schedule(base_lr = base_lr, warmup_epochs = warmup_epochs, warmup_lr = warmup_lr, initial_epoch = curr_epoch, training_epochs = epochs, target_lr = target_lr, tf_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = base_lr, decay_steps = epochs, alpha = target_lr/base_lr))))

    h = model.fit(dataset_list[0], epochs = curr_epoch + warmup_epochs + epochs, initial_epoch = curr_epoch, validation_data = dataset_list[1], callbacks = cb)
    last_loss = h.history['loss'][-1]

    cb = cb[:-1]
    if (EXPERIMENT_NAME != None and EXPERIMENT_NAME != ""):
        saved_layer_states = []
        for l in base_model.layers:
            saved_layer_states.append(l.trainable)
            l.trainable = True
        model.save_weights(current_path + "/logs/" + EXPERIMENT_NAME + "/weights_step_{}.hdf5".format(unfreeze_step))
        for l, layer_state in zip(base_model.layers, saved_layer_states):
            l.trainable = layer_state
    
    tf.keras.backend.clear_session()
    gc.collect()