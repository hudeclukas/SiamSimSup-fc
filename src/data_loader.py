import numpy as np
import os
import random
import matplotlib.pyplot as plt
from markdown.extensions.sane_lists import SaneUListProcessor
from skimage import transform

class ObjectSuperpixels:
    def __init__(self):
        self.superpixels = []

class SUPSIM:
    visualize = False
    def __init__(self, path, batch_size=128, max_steps=5000, image_size=None):
        self.batch_size = batch_size
        self.train = SUPSIM.train(path, batch_size, max_steps, image_size)
        self.test = SUPSIM.test(path, batch_size, max_steps, image_size)
        self.max_steps = max_steps

    def set_path(self, path):
        if os.path.exists(os.path.abspath(path)):
            self.train.abs_path = os.path.abspath(path) + "\\train"
            self.test.abs_path = os.path.abspath(path) + "\\test"

    def load_data(self):
        self.read_data_to_array(self.train.abs_path, train=True)
        self.read_data_to_array(self.test.abs_path, test=True)
        return True

    def read_data_to_array(self, abspath=None, train=False, test=False):
        if not os.path.exists(abspath):
            return
        if not (train or test):
            return
        files = os.listdir(abspath)
        for file in files:
            with open(os.path.join(abspath, file), "rb") as bf:
                objs = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                for o in range(objs):
                    obj_sups = ObjectSuperpixels()
                    sups = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                    for s in range(sups):
                        rows = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        cols = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        data = np.empty(shape=[rows, cols, 3], dtype=np.ubyte)
                        bf.readinto(data.data)
                        data = (data / 255).astype(np.float32)
                        # plt.imshow(data)
                        obj_sups.superpixels.append(data)
                    if train:
                        self.train.append(obj_sups)
                    elif test:
                        self.test.append(obj_sups)
                bf.close()

    @staticmethod
    def next_batch(data, batch_size=None, image_size=None, visualize=False):
        if batch_size == None:
            return []
        neg_size = batch_size
        pos_size = int(batch_size / 2)
        neg_classes = random.sample(data, neg_size)
        pos_classes = random.sample(data, pos_size)
        neg_pairs_count = int(neg_size / 2)
        neg_s = [random.choice(c.superpixels) for c in neg_classes]
        # neg_s = np.array(neg_s).reshape([neg_pairs_count, 2])
        neg_s_1 = neg_s[0:neg_pairs_count]
        neg_s_2 = neg_s[neg_pairs_count:neg_size]
        neg_l = np.zeros(neg_pairs_count, dtype=np.float32)
        pos_s = []
        for i in range(pos_size):
            pos_s.append(random.sample(pos_classes[i].superpixels, 2))
        pos_s = np.array(pos_s).transpose()
        pos_s_1 = pos_s[0]
        pos_s_2 = pos_s[1]
        pos_l = np.ones(pos_size, dtype=np.float32)
        batch_s_t_1 = np.concatenate((neg_s_1, pos_s_1))
        batch_s_t_2 = np.concatenate((neg_s_2, pos_s_2))
        batch_l_t = np.concatenate((neg_l, pos_l))
        if not image_size == None:
            batch_s_t_1 = np.asarray([transform.resize(image, image_size, mode="reflect") for image in batch_s_t_1])
            batch_s_t_2 = np.asarray([transform.resize(image, image_size, mode="reflect") for image in batch_s_t_2])
            # zeros[0:twos.shape[0],0:twos.shape[1]] = twos
        # indices = np.arange(batch_size, dtype=np.int32)
        # random.shuffle(indices)
        # batch_s = np.zeros([batch_size])
        # batch_l = np.zeros([batch_size])
        # for i in range(batch_size):
        #     batch_s[i] = batch_s_t[indices[i]]
        #     batch_l[i] = batch_l_t[indices[i]]
        if visualize:
            SUPSIM.vizualize_batch(batch_s_t_1, batch_s_t_2)
        return batch_s_t_1, batch_s_t_2, batch_l_t

    @staticmethod
    def vizualize_batch(batch_1, batch_2):
        viz = [np.concatenate((sup1, sup2), 1) for sup1, sup2 in zip(batch_1, batch_2)]
        viz = np.vstack(viz)
        viz = viz.transpose([1, 0, 2])
        plt.imshow(viz)

    class train:
        def __init__(self,path,batch_size=128,max_steps=5000,size=None):
            self.abs_path = os.path.abspath(path) + "\\train"
            self.batch_size = batch_size
            self.max_steps = max_steps
            self.data = []
            self.image_size=size

        def append(self, o):
            self.data.append(o)

        def __iter__(self):
            self.current_step = 0
            return self

        def __next__(self):
            return self.next()

        def next(self):
            if self.current_step < self.max_steps:
                self.current_step += 1
                # print("another iteration")
                return SUPSIM.next_batch(
                    data=self.data,
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    visualize=SUPSIM.visualize
                ), self.current_step
            else:
                raise StopIteration

    class test:
        def __init__(self,path,batch_size=128,max_steps=5000,size=None):
            self.abs_path = os.path.abspath(path) + "\\test"
            self.batch_size = batch_size
            self.max_steps = max_steps
            self.data = []
            self.image_size=size

        def append(self, o):
            self.data.append(o)

        def __iter__(self):
            self.current_step = 0
            return self

        def __next__(self):
            return self.next()

        def next(self):
            if self.current_step < self.max_steps:
                self.current_step += 1
                return SUPSIM.next_batch(
                    data=self.data,
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    visualize=SUPSIM.visualize
                ), self.current_step
            else:
                raise StopIteration
