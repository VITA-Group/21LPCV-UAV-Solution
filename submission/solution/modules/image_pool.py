import numpy as np

class ImagePool(object):
    def __init__(self, pool_size, pool_h, pool_w, img_h, img_w):
        self.pool_size = pool_size
        self.pool_h, self.pool_w = pool_h, pool_w
        self.ksize = img_h // pool_h
        self.image_pool = np.zeros((pool_size, pool_h, pool_w, 3), dtype=np.uint8)
        self.idx_pool = np.zeros((pool_size,), dtype=np.int32)
        self.tail = 0
        self.head = 0

    def dsample_image(self, img, ksize):
        h, w = img.shape[:2]
        resized_img = np.lib.stride_tricks.as_strided(
            img,
            shape=(int(h / ksize), int(w / ksize), ksize, ksize, 3),
            strides=img.itemsize * np.array([ksize * w * 3, ksize * 3, w * 3, 1 * 3, 1]))
        return resized_img[:, :, 0, 0].copy()

    def read(self):
        frame_idx = self.idx_pool[self.head]
        img_resized = self.image_pool[self.head, :, :, :]
        self.head += 1
        return frame_idx, img_resized

    def write(self, img_orig, idx):
        self.image_pool[self.tail, :, :, :] = self.dsample_image(img_orig, self.ksize)
        self.idx_pool[self.tail] = idx
        self.tail += 1

    def get_pool_hw(self):
        return self.pool_h, self.pool_w

    def is_full(self):
        return self.tail == self.pool_size

    def is_empty(self):
        return self.head == self.tail

    def reset(self):
        self.head = 0
        self.tail = 0
