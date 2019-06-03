import json
import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(
    description='ISIC 2019 Skin Lesion Analysis Towards Melanoma Detection',
    formatter_class=SmartFormatter)

parser.add_argument('--verbose', action='store_true', help='Verbose mode')
parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')

parser.add_argument('--model_name', type=str, default='efficientnet',
                    choices=['efficientnet'],  # ToDo: Add more?!
                    help='Model name for training')
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd', 'rmsprop'],
                    help='Optimizer for training')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_scheduler', action='store_true', help='Use a LR scheduler to control LR')
parser.add_argument('--data_augmentation', action='store_true', help='Apply data augmentations at train time')

parser.add_argument('--img_size', type=int, default=200, help='Initial image size (square img)')
parser.add_argument('--crop_size', type=int, default=200, help='Image Center crop size (square img)')

parser.add_argument('--output_dir', type=str, default='results/new_logs+train_info',
                    help='Where progress will be saved')
parser.add_argument('--path_extension', type=str, default='', help='Info to add to the save dir')

try:
    args = parser.parse_args()
except:
    print("Working with Jupyter notebook! (Default Arguments)")
    args = parser.parse_args("")

if args.output_dir == "results/new_logs+train_info":
    optional_info = ""
    if args.lr_scheduler: optional_info += "_LRScheduler"
    if args.data_augmentation: optional_info += "_DA"
    if args.path_extension != "": optional_info += "_" + args.path_extension
    args.output_dir = "results/new_logs_{}_{}".format(args.model_name, args.optimizer, optional_info)

    # Save arguments
    # https://stackoverflow.com/a/55114771
    with open(args.output_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
