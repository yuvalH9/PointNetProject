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
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--reg_alpha', type=float, default=0.001, help='Regularization factor alpha for Tnet loss [default: 0.001]')
parser.add_argument('--use_knn', type=int, default=0, help='Use KNN Componnet [default: False]')
parser.add_argument('--moment_order', type=int, default=2, help='Moment Max Order [default: 2]')
parser.add_argument('--use_lifting', type=int, default=0, help='Use Input Lifting [default: False]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


MODEL = FLAGS.model
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
TEST_NAME = FLAGS.test_name
REG_ALPHA = FLAGS.reg_alpha
USE_KNN = FLAGS.use_knn
MOMENT_ORDER = FLAGS.moment_order
USE_LIFTING = FLAGS.use_lifting

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# if os.name == 'nt': #TODO: FIX COPY
#     os.system('copy %s %s' % (os.path.join('models', FLAGS.model+'.py'), LOG_DIR))  # bkp of model def
#     os.system('copy train.py %s' % (LOG_DIR))  # bkp of train procedure
# else:
#     os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
#     os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure

# LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
# LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Use Device: " + DEVICE.__str__())



# ModelNet40 official train/test split
DATA_PATH = os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048')
TRAIN_FILES = provider.getDataFiles( \
os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def one_epoch_train(epoch,model, optimizer, summary_writer,iter_verbose = 10, aug_params=None, use_Normals=False):

    print("-----Start Of Epoch {} -----".format(epoch))
    model.train()
    transform = None
    # Do augmentations
    if not use_Normals:
        if aug_params:
            noise_transform = transforms.RandomApply([provider.NoisedPointCloud(sigma=aug_params['sigma'],clip=aug_params['clip'])], p=aug_params['p_noise'])
            rotation_transform = transforms.RandomApply([provider.RotatePointCloud()], p=aug_params['p_rotation'])
            transform = transforms.Compose([noise_transform, rotation_transform, provider.ToTensor()])
    else:#Defualt Transform
        transform = transforms.Compose([provider.ToTensor()])

    train_ds = provider.PointCloudDataSet(DATA_PATH, numOfPoints=NUM_POINT,transform=transform, use_normals=use_Normals)
    train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)

    running_loss = 0.0
    total_observations = 0
    correct_predeicts = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['data'].to(DEVICE).float(), data['label'].to(DEVICE)
        optimizer.zero_grad()


        if MODEL == 'pointnet_cls':
            # Forward
            main_net_output, Tnet_3x3_output, Tnet_64x64_output = model.forward(inputs)
            #Backward
            loss = model.get_model_loss(main_net_output, labels, Tnet_3x3_output, Tnet_64x64_output)
        elif MODEL == 'momenet_cls' or MODEL == 'momenet_with_normals':
            # Forward
            main_net_output = model.forward(inputs)
            # Backward
            loss = model.get_model_loss(main_net_output, labels)
        else:
            raise Exception("No such Model: " +MODEL)
        loss.backward()
        optimizer.step()

        #Acumulate Prformence data
        running_loss += loss.item()
        total_observations += labels.size(0)
        predicts = torch.argmax(main_net_output,1)
        correct_predeicts += (predicts == labels).sum().item()

        #Logs
        if i % iter_verbose == (iter_verbose-1):  # print every iter_verbose mini-batches
            #Train Loss
            print('[Epoch: %d, Batch: %4d / %4d], Train Loss: %.4f, Train Accuracy: %.4f' %
                  (epoch, i + 1, len(train_loader), running_loss / 10, correct_predeicts/total_observations))
            summary_writer.add_scalar('Train Loss', running_loss / 10, epoch * len(train_loader) + i)
            summary_writer.add_scalar('Train Accuracy', correct_predeicts/total_observations, epoch * len(train_loader) + i)
            running_loss = 0.0
            total_observations = 0
            correct_predeicts = 0

    print("-----End Of Epoch {} -----".format(epoch))

def one_epoch_eval(epoch, model, summary_writer, best_acc=0, use_Normals=False):
    print("Evaluate Accuracy on Test After Epoch:{}".format(epoch))

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

        print(
            'Validation Accuracy: %.4f, Validation Mean Class Accuracy: %.4f' % (val_accuracy, val_mean_class_accuracy))
        summary_writer.add_scalar('Test Accuracy', val_accuracy, epoch)
        summary_writer.add_scalar('Test Mean Class Accuracy', val_mean_class_accuracy, epoch)
        summary_writer.add_scalar('Validation Mean Loss', running_val_loss / valid_loader.__len__(), epoch)

    # Save Best Model
    if best_acc < val_accuracy:
        best_acc = val_accuracy
        best_model = copy.deepcopy(model.state_dict())
        torch.save(best_model, os.path.join(summary_writer.log_dir, "Best_model.pth"))

    return best_acc


def train():

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    use_normals = False
    k = 20 if USE_KNN else 0  # If k==0 no use of knn
    #Select Model
    if MODEL == 'pointnet_cls':
        model = PointNet_Classic(numOfClasses=NUM_CLASSES, reg_alpha=REG_ALPHA).to(DEVICE)
    elif MODEL == 'momenet_cls':
        model = Momenet(numOfClasses=NUM_CLASSES, use_normals=False, k=k, moment_order=MOMENT_ORDER, use_lifting=USE_LIFTING).to(DEVICE)
    elif MODEL == 'momenet_with_normals':
        model = Momenet(numOfClasses=NUM_CLASSES, use_normals=True, k=k, moment_order=MOMENT_ORDER, use_lifting=USE_LIFTING).to(DEVICE)
        use_normals = True
    else:
        raise Exception('No Such Model: '+ MODEL)

    #Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)

    # Aug Params
    aug_params = {'sigma': 0.01, 'clip': 0.05, 'p_noise': 0.3, 'p_rotation': 0.5} #TODO: Add to parser

    #Define train log params
    log_path = os.path.join(LOG_DIR,TEST_NAME,date_time)
    summary_writer = SummaryWriter(log_path)
    LOG_FOUT = open(os.path.join(log_path, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(FLAGS) + '\n')
    best_acc = 0

    #summary_writer.add_graph(model)

    for epoch in range(MAX_EPOCH):
        one_epoch_train(epoch, model, optimizer, summary_writer, iter_verbose=10, aug_params=aug_params, use_Normals=use_normals)
        best_acc = one_epoch_eval(epoch, model, summary_writer, best_acc=best_acc, use_Normals=use_normals)

    summary_writer.close()
    LOG_FOUT.close()



if __name__ == "__main__":
    train()

