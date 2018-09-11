import cv2, os
import numpy as np

class DataLoader(object):

    def __init__(self, file_path, training_labels=80, test_labels=20, img_size=84, img_num_per_class=600):

        self.traing_labels = training_labels
        self.test_labels = test_labels
        self.img_size = img_size
        self.img_num_per_class = img_num_per_class
        self.data = self.load_images(file_path)


    def load_images(self, file_path):

        image_name = os.listdir(file_path)
        data = dict()
        training_data = []; test_data = []

        for i in range(self.traing_labels):

            images = []
            start = i * self.img_num_per_class
            end = start + self.img_num_per_class

            for j in range(start, end):

                img = cv2.imread(file_path + '/' + img_name[j], cv2.IMREAD_COLOR)
                img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                images.append(img / 255.0)

            training_data.append(images)

        for i in range(self.traing_labels, self.traing_labels + self.test_labels):

            images = []
            start = i * self.img_num_per_class
            end = start + self.img_num_per_class

            for j in range(start, end):
                img = cv2.imread(file_path + '/' + img_name[j], cv2.IMREAD_COLOR)
                img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                images.append(img / 255.0)

            test_data.append(images)

        training_data = np.asarray(training_data)
        test_data = np.asarray(test_data)

        data['train'] = training_data
        data['test'] = test_data

        return data

    def train_batch(self, AttFeat, batch_size=64):

        data = self.data['train']
        anchor = []; positive = []; negative = []

        for bz in range(batch_size):
            cls_idx = np.random.randint(self.traing_labels)
            anchor_idx = np.random.randint(self.img_num_per_class)
            anchor_img = data[cls_idx, anchor_idx]
            anchor_feat = AttFeat(anchor_img)

            '''
            We need to find hard positive, which is 
            defined as max{||f(anchor) - f(positive)||}
            '''
            max_dist = 0; pos_idx = -1
            for i in range(self.img_num_per_class):
                if i == anchor_idx: continue
                feat = AttFeat(data[cls_idx, i])
                dist = np.linalg.norm(anchor_feat - feat)
                if dist > max_dist:
                    max_dist = dist
                    pos_idx = i
            pos_img = data[cls_idx, pos_idx]

            '''
            We need to find hard negative, which is
            defined as min{||f(anchor) - f(negative)||}
            '''
            while(True):
                r = np.random.randint(self.traing_labels)
                if r != cls_idx:
                    cls_idx = r
                    break

            min_dist = 1000000; neg_idx = -1
            for i in range(self.img_num_per_class):
                feat = AttFeat(data[cls_idx, i])
                dist = np.linalg.norm(anchor_feat - feat)
                if dist < min_dist:
                    min_dist = dist
                    neg_idx = i
            neg_img = data[cls_idx, neg_idx]

            anchor.append(anchor_img)
            positive.append(pos_img)
            negative.append(neg_img)

        anchor = np.asarray(anchor)
        positive = np.asarray(positive)
        negative = np.asarray(negative)

        return anchor, positive, negative


    def test_batch(self, n_way, k_shot, test_size = 50):

        data = self.data['test']
        query_data = []; test_data = []; test_labels = []

        cls_idx = np.random.permutation(range(self.test_labels))[: n_way]

        for i in cls_idx:
            images = data[i]
            img_idx = np.random.permutation(range(self.img_num_per_class))[: k_shot + test_size]
            query_img = images[: k_shot]
            test_img = images[k_shot:]

            query_data.append(query_img)
            test_data.append(test_img)
            test_labels.append(cls_idx.idx(i))

        query_data = np.asarray(query_data).reshape([-1, self.img_size, self.img_size])
        test_data = np.asarray(test_data).reshape([-1, self.img_size, self.img_size])

        return query_data, test_data, test_labels







