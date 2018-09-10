import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='./images',
                        help='the file path of images')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--disp_step', type=int, default=200,
                        help='display step during training')

