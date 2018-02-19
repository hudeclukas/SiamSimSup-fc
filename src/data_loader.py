from _tracemalloc import start

import numpy as np
import os
import random
import matplotlib.pyplot as plt

from array import array

from skimage import transform
import sklearn.preprocessing as prep


def resize_batch_images(batch: list, image_size: tuple) -> np.ndarray:
    return np.asarray([transform.resize(image, image_size, mode="reflect") for image in batch])


class ImageObjects:
    def __init__(self):
        self.objects = []
        self.name = ''

    def write_similarities_to_file(self, path: str, labels_order: list, ith: int):
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        with open(path, mode='ab+') as bf:
            if ith == 0:
                bf.write(len(labels_order).to_bytes(4, 'little', signed=False))
                lb = array('i', labels_order)
                lb.tofile(bf)
            if ith >= 0:
                bf.write(len(self.objects[ith].labels).to_bytes(4, 'little', signed=False))
                for j in range(len(self.objects[ith].labels)):
                    sim = array('d', self.objects[ith].similarity[j])
                    sim.tofile(bf)

            bf.close()
        print('File saved {:s}'.format(path))


class ObjectSuperpixels:
    def __init__(self):
        self.superpixels = []
        self.labels = []
        self.similarity = []


class SUPSIM:
    visualize = False

    def __init__(self, path, batch_size=128, image_size=None, use_grayscale=False):
        self.batch_size = batch_size
        self.train = SUPSIM.train(path, batch_size, image_size, use_grayscale)
        self.test = SUPSIM.test(path, batch_size, image_size, use_grayscale)

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
        print("Loading from \""+abspath +"\"")
        files = os.listdir(abspath)
        for file in files:
            img_objs = ImageObjects()
            img_objs.name = file
            try:
                with open(os.path.join(abspath, file), "rb") as bf:
                    channels = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                    objs = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                    for o in range(objs):
                        obj_sups = ObjectSuperpixels()
                        sups = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                        for s in range(sups):
                            if file.endswith('.supl'):
                                label = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                                obj_sups.labels.append(label)
                            rows = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                            cols = int.from_bytes(bf.read(4), byteorder="little", signed=False)
                            data = np.empty(shape=[rows, cols, channels], dtype=np.ubyte)
                            bf.readinto(data.data)
                            # data = (data / 255).astype(np.float32)
                            data = prep.scale(data.reshape((rows*cols*channels))).reshape((rows, cols, channels))
                            # plt.imshow(data)
                            obj_sups.superpixels.append(data)
                        if train:
                            self.train.append(obj_sups)
                        elif test:
                            self.test.append(obj_sups)
                        img_objs.objects.append(obj_sups)
                    bf.close()
                if train:
                    self.train.add_object(img_objs)
                elif test:
                    self.test.add_object(img_objs)
            except IOError:
                print("File {:s} does not exist".format(os.path.join(abspath, file)))
        print(str(len(self.train.data)) + " train objects loaded")
        print(str(len(self.test.data)) + " test objects loaded")
        if train:
            self.train.data = np.array(self.train.data)
        if test:
            self.test.data = np.array(self.test.data)

    @staticmethod
    def next_batch(data, batch_size=None, image_size=None, visualize=False, use_grayscale=False):
        if batch_size == None:
            return []
        if batch_size > len(data):
            batch_size = (len(data) >> 1) << 1
        neg_size = (batch_size >> 1) << 1
        pos_size = int(batch_size) >> 1
        neg_classes = np.random.choice(data, neg_size, False)
        pos_classes = np.random.choice(data, pos_size, False)
        neg_pairs_count = int(neg_size) >> 1
        neg_s = [random.choice(c.superpixels) for c in neg_classes]
        # neg_s = np.array(neg_s).reshape([neg_pairs_count, 2])
        neg_s_1 = neg_s[0:neg_pairs_count]
        neg_s_2 = neg_s[neg_pairs_count:neg_size]
        neg_l = np.zeros(neg_pairs_count, dtype=np.float32)
        pos_s = []
        for i in range(pos_size):
            pos_s.append(random.sample(pos_classes[i].superpixels, 2))
        pos_s = np.array(pos_s)
        axes = np.arange(len(pos_s.shape))
        a, b, *c = axes
        axes = np.concatenate((np.array((b, a), dtype=int), c)).astype(int)
        pos_s = np.array(pos_s).transpose(axes)
        pos_s_1 = pos_s[0]
        pos_s_2 = pos_s[1]
        pos_l = np.ones(pos_size, dtype=np.float32)
        batch_s_t_1 = np.concatenate((neg_s_1, pos_s_1))
        batch_s_t_2 = np.concatenate((neg_s_2, pos_s_2))
        batch_l_t = np.concatenate((neg_l, pos_l))

        # batch_s_t_1 = np.array([cv2.normalize(i, np.array([]), alpha=1, norm_type=cv2.NORM_L2) for i in batch_s_t_1])
        # batch_s_t_2 = np.array([cv2.normalize(i, np.array([]), alpha=1, norm_type=cv2.NORM_L2) for i in batch_s_t_2])

        # if not image_size == None and not batch_s_t_1[0].shape == image_size:
        #     batch_s_t_1 = resize_batch_images(batch_s_t_1, image_size)
        #     batch_s_t_2 = resize_batch_images(batch_s_t_2, image_size)

        # if use_grayscale:
        #     neg_size_h = int(neg_pairs_count / 2)
        #     pos_size_h = int(pos_size / 2)
        #     batch_s_t_1[neg_size_h:neg_pairs_count] = [color.gray2rgb(color.rgb2gray(i)) for i in batch_s_t_1[neg_size_h:neg_pairs_count]]
        #     batch_s_t_2[neg_size_h:neg_pairs_count] = [color.gray2rgb(color.rgb2gray(i)) for i in batch_s_t_2[neg_size_h:neg_pairs_count]]
        #     batch_s_t_1[neg_pairs_count+pos_size_h:batch_size] = [color.gray2rgb(color.rgb2gray(i)) for i in batch_s_t_1[neg_pairs_count+pos_size_h:batch_size]]
        #     batch_s_t_2[neg_pairs_count+pos_size_h:batch_size] = [color.gray2rgb(color.rgb2gray(i)) for i in batch_s_t_2[neg_pairs_count+pos_size_h:batch_size]]

        return batch_s_t_1, batch_s_t_2, batch_l_t

    @staticmethod
    def vizualize_batch(batch_1, batch_2):
        viz = [np.concatenate((sup1, sup2), 1) for sup1, sup2 in zip(batch_1, batch_2)]
        viz = np.vstack(viz)
        viz = viz.transpose([1, 0, 2])
        plt.imshow(viz)

    class train:
        class teacher:
            def __init__(self):
                # pairs
                self.fp = []
                # singles
                self.fn = []

        def __init__(self, path, batch_size=128, start_step=0, max_steps=5000, size=None, grayscale=False):
            self.abs_path = os.path.abspath(path) + "\\train"
            self.batch_size = batch_size
            self.start_step = start_step
            self.max_steps = max_steps
            self.images = []
            self.data = []
            self.teacher = self.teacher()
            self.image_size = size
            self.use_grayscale = grayscale

        def append(self, o):
            self.data.append(o)

        def add_object(self, o):
            self.images.append(o)

        def __iter__(self,min, max):
            self.current_step = min
            self.min = min
            self.max = max
            return self

        def __next__(self):
            return self.next()

        def next(self):
            if self.current_step < self.max:
                self.current_step += 1
                # print("another iteration")
                return SUPSIM.next_batch(
                    data=self.data,
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    visualize=SUPSIM.visualize,
                    use_grayscale=self.use_grayscale
                ), self.current_step
            else:
                raise StopIteration

        def next_validation_batch(self, size:int=128):
            neg_size = size
            pos_size = size >> 1
            neg_classes_idx = np.random.choice(np.arange(len(self.data)-1), neg_size, False)
            pos_classes_idx = np.random.choice(np.arange(len(self.data)-1), pos_size, False)
            pos_classes = self.data[pos_classes_idx]
            neg_classes = self.data[neg_classes_idx]
            neg_pairs_count = int(neg_size) >> 1
            neg_s = [random.choice(c.superpixels) for c in neg_classes]
            # neg_s = np.array(neg_s).reshape([neg_pairs_count, 2])
            neg_s_1 = neg_s[0:neg_pairs_count]
            neg_s_2 = neg_s[neg_pairs_count:neg_size]
            neg_l = np.zeros(neg_pairs_count, dtype=np.float32)
            pos_s = []
            for i in range(pos_size):
                pos_s.append(random.sample(pos_classes[i].superpixels, 2))
            pos_s = np.array(pos_s)
            axes = np.arange(len(pos_s.shape))
            a, b, *c = axes
            axes = np.concatenate((np.array((b, a), dtype=int), c)).astype(int)
            pos_s = np.array(pos_s).transpose(axes)
            pos_s_1 = pos_s[0]
            pos_s_2 = pos_s[1]
            pos_l = np.ones(pos_size, dtype=np.float32)
            batch_s_t_1 = np.concatenate((neg_s_1, pos_s_1))
            batch_s_t_2 = np.concatenate((neg_s_2, pos_s_2))
            batch_l_t = np.concatenate((neg_l, pos_l))

            neg_s_idx = neg_classes_idx.reshape((neg_pairs_count,2),order='F')
            pos_s_idx = np.concatenate((pos_classes_idx.reshape((pos_size,1)), pos_classes_idx.reshape((pos_size,1))),axis=1)
            idx = np.concatenate((neg_s_idx, pos_s_idx))
            return batch_s_t_1, batch_s_t_2, batch_l_t, idx

    class test:
        def __init__(self, path, batch_size=128, size=None, grayscale=False):
            self.abs_path = os.path.abspath(path) + "\\test"
            self.batch_size = batch_size
            self.data = []
            self.images = []
            self.image_size = size
            self.use_grayscale = grayscale

        def append(self, o):
            self.data.append(o)

        def add_object(self, o):
            self.images.append(o)

        def __iter__(self, min, max):
            self.current_step = self.min
            self.min = min
            self.max = max
            return self

        def __next__(self):
            return self.next()

        def next(self):
            if self.current_step < self.max:
                self.current_step += 1
                return SUPSIM.next_batch(
                    data=self.data,
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    visualize=SUPSIM.visualize,
                    use_grayscale=self.use_grayscale
                ), self.current_step
            else:
                raise StopIteration

    def next_teacher_batch(self, batch_size, image_size):
        batch_size_teacher = int(0.2 * batch_size)
        fn_size = batch_size_teacher >> 1
        fn_size = fn_size if fn_size < len(self.train.teacher.fn) else len(self.train.teacher.fn)
        fp_size = batch_size_teacher - fn_size
        fp_size = fp_size if fp_size < len(self.train.teacher.fp) else len(self.train.teacher.fp)
        batch_size_classic = batch_size - (fp_size + fn_size)
        batch_1, batch_2, labels = self.next_batch(self.train.data, batch_size_classic, self.train.image_size)
        if len(self.train.teacher.fp) > 0:
            fp = np.array(self.train.teacher.fp)
            batch_fp_idx = np.random.choice(np.arange(fp.shape[0]), fp_size, replace=False)
            batch_fp_idx = fp[batch_fp_idx]
            batch_fp_idx = batch_fp_idx.transpose()
            batch_fp_classes_1 = np.array(self.train.data)[batch_fp_idx[0]]
            batch_fp_classes_2 = np.array(self.train.data)[batch_fp_idx[1]]
            batch_fp_1 = np.array([random.choice(sp.superpixels) for sp in batch_fp_classes_1])
            batch_fp_2 = np.array([random.choice(sp.superpixels) for sp in batch_fp_classes_2])
            batch_1 = np.concatenate((batch_1, batch_fp_1))
            batch_2 = np.concatenate((batch_2, batch_fp_2))
            labels = np.concatenate((labels, np.zeros(fp_size ,dtype=np.float32)))
        if len(self.train.teacher.fn) > 0:
            fn = self.train.teacher.fn
            batch_fn_idx = random.sample(fn, fn_size)
            batch_fn_classes = np.array(self.train.data)[batch_fn_idx]
            batch_fn = np.array([random.sample(sp.superpixels, 2) for sp in batch_fn_classes])
            axes = np.arange(len(batch_fn.shape))
            a, b, *c = axes
            axes = np.concatenate((np.array((b, a), dtype=int), c)).astype(int)
            batch_fn = batch_fn.transpose(axes)
            batch_1 = np.concatenate((batch_1, batch_fn[0]))
            batch_2 = np.concatenate((batch_2, batch_fn[1]))
            labels = np.concatenate((labels, np.ones(fn_size, dtype=np.float32)))

        return batch_1, batch_2, labels
