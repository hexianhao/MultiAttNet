import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Module import AttNet, weight_init
from Dataset import DataLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='./images',
                        help='the file path of images')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--disp_step', type=int, default=200,
                        help='display step during training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--max_step', type=int, default=10000, help='max step during training')
    parser.add_argument('--test_step', type=int, default=50, help='test times')
    parser.add_argument('--img_size', type=int, default=84, help='the size of image fed into CNN')
    parser.add_argument('--n_way', type=int, default=5, help='n way during test')
    parser.add_argument('--k_shot', type=int, default=1, help='k shot during test')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')

    arg_opt = parser.parse_args()

    loader = DataLoader(arg_opt.image_file)
    AttFeat = AttNet()
    triplet_loss = nn.TripletMarginLoss()
    optimizer = optim.SGD(AttFeat.parameters(), lr=arg_opt.learning_rate)

    for s in range(1, arg_opt.max_step+1):

        anchor, positive, negative = loader.train_batch(AttFeat, arg_opt.batch_size)
        anchor = Variable(anchor)
        positive = Variable(positive)
        negative = Variable(negative)

        loss = triplet_loss(anchor, positive, negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if s % arg_opt.disp_step == 0:  ## 测试阶段

            correct = 0.; wrong = 0.
            for _ in range(arg_opt.test_step):

                query_data, test_data, test_labels = loader.test_batch(arg_opt.n_way, arg_opt.k_shot)
                query_data = Variable(query_data)
                test_data = Variable(test_data)

                query_feat = AttFeat(query_data)
                test_feat = AttFeat(test_data)

                for i in range(len(test_feat)):
                    min_dist = 10000000; cls = -1

                    for j in range(len(query_feat)):

                        dist = torch.dist(query_feat[j], test_feat[i])
                        if dist < min_dist:
                            min_dist = dist
                            cls = j
                    cls = cls / arg_opt.k_shot

                    if cls == test_labels[i]:
                        correct += 1
                    else: wrong += 1

            print("Epoch {}: test accuracy is {}".format(s, correct / (correct + wrong)))
















