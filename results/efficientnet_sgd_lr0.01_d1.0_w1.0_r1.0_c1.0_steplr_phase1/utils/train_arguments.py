import argparse
import json
import os


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
parser.add_argument('--epochs', type=int, default=200, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for training')

parser.add_argument('--model_name', type=str, default='efficientnet',
                    choices=['efficientnet'],  # ToDo: Add more?!
                    help='Model name for training')
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd', 'rmsprop'],
                    help='Optimizer for training')

parser.add_argument('--balanced_sampler', action='store_true', help='Use a balanced train dataloader')

parser.add_argument('--depth_coefficient', type=float, default=1.0, help='[Efficientnet] Depth Coefficient')
parser.add_argument('--width_coefficient', type=float, default=1.0, help='[Efficientnet] Width Coefficient')
parser.add_argument('--resolution_coefficient', type=float, default=1.0, help='[Efficientnet] Resolution Coefficient')
parser.add_argument('--compound_coefficient', type=float, default=1.0,
                    help='[Efficientnet] compound coefficient φ to uniformly scales network width, depth, and resolution')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--data_augmentation', action='store_true', help='Apply data augmentations at train time')

parser.add_argument('--img_size', type=int, default=256, help='Initial image size (square img)')
parser.add_argument('--crop_size', type=int, default=224, help='Final Image size - Crop (square img)')

parser.add_argument('--output_dir', type=str, default='results/new_logs+train_info',
                    help='Where progress will be saved')
parser.add_argument('--path_extension', type=str, default='', help='Info to add to the save dir')

try:
    args = parser.parse_args()
except:
    print("Working with Jupyter notebook! (Default Arguments)")
    args = parser.parse_args("")

optional_info = ""
if args.data_augmentation: optional_info += "_DA"
if args.balanced_sampler: optional_info += "_batchBalanced"
if args.path_extension != "": optional_info += "_" + args.path_extension

if args.output_dir == "results/new_logs+train_info":
    args.output_dir = "results/new_logs_{}_{}/".format(args.model_name, args.optimizer, optional_info)
else:
    args.output_dir = args.output_dir + optional_info + "/"

print(args.output_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Save arguments
# https://stackoverflow.com/a/55114771
with open(args.output_dir + '/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
