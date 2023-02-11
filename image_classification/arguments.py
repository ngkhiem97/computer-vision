import argparse

"""
Here are the param for the training
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-size', type=float, default=.9, help='Split size for train and test')
    parser.add_argument('--aug-rescale', type=float, default=1./255, help='Data augmentation rescale')
    parser.add_argument('--aug-rotation', type=int, default=40, help='Data augmentation rotation range')
    parser.add_argument('--aug-width', type=float, default=0.2, help='Data augmentation width shift range')
    parser.add_argument('--aug-height', type=float, default=0.2, help='Data augmentation height shift range')
    parser.add_argument('--aug-shear', type=float, default=0.2, help='Data augmentation shear range')
    parser.add_argument('--aug-zoom', type=float, default=0.2, help='Data augmentation zoom range')
    parser.add_argument('--aug-horiz', type=bool, default=True, help='Data augmentation horizontal flip')
    parser.add_argument('--aug-vert', type=bool, default=False, help='Data augmentation vertical flip')
    parser.add_argument('--aug-fill', type=str, default='nearest', help='Data augmentation fill mode')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    args = parser.parse_args()

    return args