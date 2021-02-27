import random
random.seed(42)
import numpy as np
import os
import sys
import argparse
import provider
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Models import PointNet_Classic, Momenet
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import copy
from prettytable import PrettyTable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

#Parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or momenet_cls [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--test_name', default='test', help='Test Name [default: test]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--reg_alpha', type=float, default=0.001, help='Regularization factor alpha for Tnet loss [default: 0.001]')
parser.add_argument('--use_knn', type=int, default=0, help='Use KNN Componnet [default: False]')
parser.add_argument('--moment_order', type=int, default=2, help='Moment Max Order [default: 2]')
parser.add_argument('--use_lifting', type=int, default=0, help='Use Input Lifting [default: False]')
parser.add_argument('--model_path', default="", help='Use Input Lifting [default: Path to evaluated model]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL = FLAGS.model
LOG_DIR = FLAGS.log_dir
TEST_NAME = FLAGS.test_name
REG_ALPHA = FLAGS.reg_alpha
USE_KNN = FLAGS.use_knn
MOMENT_ORDER = FLAGS.moment_order
USE_LIFTING = FLAGS.use_lifting
MODEL_PATH = FLAGS.model_path
MAX_NUM_POINT = 2048
NUM_CLASSES = 40
if MODEL_PATH == "":
    raise Exception("Please enter correct model path")
else:
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use Device: " + DEVICE.__str__())

# ModelNet40 official train/test split
DATA_PATH = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')
TRAIN_FILES = provider.getDataFiles( \
os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def one_epoch_eval(model, use_Normals=False):
    print("Evaluate Accuracy on Test")

    valid_ds = provider.PointCloudDataSet(DATA_PATH, numOfPoints=NUM_POINT, valid=True, use_normals=use_Normals)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=BATCH_SIZE * 2)

    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    running_val_loss = 0.0
    total_observations = 0
    correct_predeicts = 0

    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data['data'].to(DEVICE).float(), data['label'].to(DEVICE)

            if MODEL == 'pointnet_cls':
                # Forward
                main_net_output, Tnet_3x3_output, Tnet_64x64_output = model.forward(inputs)
                # Loss
                running_val_loss += model.get_model_loss(main_net_output, labels, Tnet_3x3_output, Tnet_64x64_output)
            elif MODEL == 'momenet_cls' or MODEL == 'momenet_with_normals':
                # Forward
                main_net_output = model.forward(inputs)
                # Loss
                running_val_loss += model.get_model_loss(main_net_output, labels)
            else:
                raise Exception("No such Model: " + MODEL)

            # Predict
            predicts = torch.argmax(main_net_output, 1)

            # Total Accuracy
            total_observations += labels.size(0)
            correct_predeicts += (predicts == labels).sum().item()

            # Mean Accuracy per class
            for sample_idx in range(labels.size(0)):
                l = labels[sample_idx]
                total_seen_class[l] += 1
                total_correct_class[l] += (predicts[sample_idx] == l).item()

        val_mean_class_accuracy = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        val_accuracy = correct_predeicts / total_observations

        pt_params = PrettyTable()
        pt_params.title = "Model Params"
        pt_params.field_names = ["Param", "Value"]
        pt_params.add_row(["Model", MODEL])
        pt_params.add_row(["Use Normals", use_Normals])
        pt_params.add_row(["Knn (k)", USE_KNN if USE_KNN else "No Knn"])
        pt_params.add_row(["Moment Order", MOMENT_ORDER])
        pt_params.add_row(["Num Of Points", NUM_POINT])
        pt_params.add_row(["Use Lifting", USE_LIFTING])
        pt_params.add_row(["Model Path", MODEL_PATH])
        print(pt_params)

        pt_acc = PrettyTable()
        pt_acc.title = "Model Accuracy"
        pt_acc.field_names = ["Total Accuracy", "Class Mean Accuracy"]
        pt_acc.add_row(["{:.4}".format(val_accuracy), "{:.4}".format(val_mean_class_accuracy)])
        print(pt_acc)

        pt_cls_acc = PrettyTable()
        pt_cls_acc.title = "Class Accuracy"
        pt_cls_acc.field_names = [valid_ds.classes[cls_num] for cls_num in range(NUM_CLASSES)]
        acc_arr = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64)
        pt_cls_acc.add_row(["{:.4}".format(cls_acc) for cls_acc in acc_arr])
        print(pt_cls_acc)



def evaluate():
    use_normals = False
    k = 20 if USE_KNN else 0  # If k==0 no use of knn


    # Select Model
    if MODEL == 'pointnet_cls':
        model = PointNet_Classic(numOfClasses=NUM_CLASSES, reg_alpha=REG_ALPHA).to(DEVICE)
    elif MODEL == 'momenet_cls':
        model = Momenet(numOfClasses=NUM_CLASSES, use_normals=False, k=k, moment_order=MOMENT_ORDER,
                        use_lifting=USE_LIFTING).to(DEVICE)
    elif MODEL == 'momenet_with_normals':
        model = Momenet(numOfClasses=NUM_CLASSES, use_normals=True, k=k, moment_order=MOMENT_ORDER,
                        use_lifting=USE_LIFTING).to(DEVICE)
        use_normals = True
    else:
        raise Exception('No Such Model: ' + MODEL)

    #Load Model
    model.load_state_dict(torch.load(MODEL_PATH))

    #Validation Epoch
    one_epoch_eval(model, use_Normals=use_normals)


if __name__ == "__main__":
    evaluate()

