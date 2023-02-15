import argparse

"""
Here are the param for the training
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--steps-per-epoch', type=int, default=60000, help='Number of steps per epoch')
    parser.add_argument('--validation-steps', type=int, default=1, help='Number of validation steps')
    args = parser.parse_args()
    return args