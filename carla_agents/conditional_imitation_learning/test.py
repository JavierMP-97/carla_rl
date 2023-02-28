import tensorflow as tf
import cv2
import numpy as np
import time

l = []
l2 = []
def parse(im_path, txt_path):
  image = tf.io.read_file(im_path)
  image = tf.io.decode_png(image)
  image = tf.cast(image, tf.float32)

  txt = tf.io.read_file(txt_path)
  txt = tf.io.decode_csv(txt, [float(), int()], select_cols = [3, 6])
  lab = txt[0]
  command = tf.one_hot(txt[1], 3)

  return (image, command), lab

for i in range(400000):
    s = "D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\middle_rgb_" + str(i).zfill(7) + ".png"
    s2 = "D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\info_" + str(i).zfill(7) + ".txt"
    l.append(s)
    l2.append(s2)

with tf.device('CPU'):
    dataset = tf.data.Dataset.from_tensor_slices((l, l2))
    dataset = dataset.map(parse)
t = time.time()
for i, element in enumerate(dataset):
    print(element[0][1].numpy())
    if i % 10000 == 0:
        print(i)
        print(time.time()-t)
        t = time.time()
        #print(element)
'''
t = time.time()
l = []
for i in range(400000):
    if i % 10000 == 0:
        print(i)
        print(time.time()-t)
        t = time.time()
    with open("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\middle_rgb_" + str(i).zfill(7) + ".png","rb") as f:
        im = f.read()
        f.close()
    im = np.fromstring(im, np.uint8)
    l.append(im)
    #f = open("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\middle_rgb_0000000.png","rb")
    #l.append(np.fromfile("D:\\PC-Javier\\Desktop\\Carla14\\carla_rl\\log\\agente\\middle_rgb_" + str(i).zfill(7) + ".png", dtype=np.uint8)) #np.frombuffer(f.read(), dtype=np.uint8)
    #img = cv2.imdecode(byte, cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #with tf.device('CPU'):
    #    img = tf.convert_to_tensor(img)
'''