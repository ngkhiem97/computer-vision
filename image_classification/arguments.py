import argparse

"""
Here are the param for the training
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer')
    parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='Loss function')
    parser.add_argument('--metrics', type=str, default='accuracy', help='Metrics')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    return args