import os
import sys
import numpy as np
import h5py
import zipfile as ZipAPIs
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pyvista as pv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

TRAIN_FILES = 'train_files.txt'
TEST_FILES = 'test_files.txt'
CLASS_LIST = 'shape_names.txt'

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)

    if os.name == 'nt':
        os.system('wget --no-check-certificate %s' % (www))
        with ZipAPIs.ZipFile(zipfile, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

    else:
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

#Augmentations
class NoisedPointCloud(object):
    def __init__(self,sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
    def __call__(self, pointCloud):
        assert len(pointCloud.shape) == 2
        N = pointCloud.shape[0]
        jittered_data = np.clip(self.sigma * np.random.randn(N, 3), -1 * self.clip, self.clip)
        jittered_data += pointCloud
        return jittered_data
class RotatePointCloud(object):
    def __call__(self, pointCloud):
        assert len(pointCloud.shape) == 2
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])

        return np.dot(pointCloud, rotation_matrix)
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)

def showPointCloud(pointCloud_xyz):
    point_cloud = pv.PolyData(pointCloud_xyz)
    point_cloud.plot(render_points_as_spheres=True)
    pass


class PointCloudDataSet(Dataset):
    def __init__(self, data_base_path, numOfPoints, valid=False, transform= None, use_normals=False):
        self.data_base_path = data_base_path
        self.numOfPoints = numOfPoints
        self.use_normals =use_normals

        self.classes = {i:class_name for i,class_name in enumerate(getDataFiles(os.path.join(data_base_path, CLASS_LIST)))}

        if not valid:
            files_names_list = getDataFiles(os.path.join(data_base_path, TRAIN_FILES)) #'data/modelnet40_ply_hdf5_2048/train_files.txt' or 'data/modelnet40_ply_hdf5_2048/test_files.txt')
        else:
            files_names_list = getDataFiles(os.path.join(data_base_path, TEST_FILES))

        self.transforms = transform if not valid else None

        self.data_list = []
        self.labels_list = []

        #Load all data
        for fn in range(len(files_names_list)):
            current_data, current_label, current_normal = loadDataFile(files_names_list[fn])
            if self.use_normals:
               self.data_list.extend(
                   [np.concatenate((current_data[i, 0:self.numOfPoints, :], current_normal[i, 0:self.numOfPoints, :]),axis=1)
                    for i in range(current_data.shape[0])])
            else:
                self.data_list.extend([current_data[i, 0:self.numOfPoints, :] for i in range(current_data.shape[0])])
            self.labels_list.extend(np.squeeze(current_label))

    def __len__(self):
        return(len(self.data_list))
    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels_list[idx]

        #Transforms
        if self.transforms:
            data = self.transforms(data)

        return {'data': data, 'label': label}


#Moments
def getMoments(data, moments_order=1):
    """
    Computes moments for the input point cloud ([x,y,z]) for a given batch
    Args:
        data: input data as a batch [bs x N x 3] (in case using normals [bs x N x 6] )
        moments_order: maximal computed moments - 1,2,3

    Returns: batch tensor with moments:
            - first order: dim2: [x,y,z] (with normals:[x,y,z,Nx,Ny,Nz]) shape [bs x N x 3] ([bs x N x 6]);
            - second order: dim2: [x,y,z,x^2,y^2,z^2,xz,yz,zy]
                (with normals:[x,y,z,Nx,Ny,Nz,x^2,y^2,z^2,xz,yz,zy]) shape [bs x N x 9] ([bs x N x 12]);
            - third order dim2: [firstOrder,secondOrder, x^3,y^3,z^3,x^2z,y^2x,z^y,xyz]
                (with normals:[x,y,z,Nx,Ny,Nz,secondOrder, x^3,y^3,z^3,x^2z,y^2x,z^y,xyz]) shape [bs x N x 19] ([bs x N x 22]);

    """

    data_xyz = data[:, :, 0:3]
    if moments_order == 1:
        return data
    elif moments_order == 2: #Upto 2 order moment

        pure_second = data_xyz * data_xyz #x^2,y^2,z^2
        mixed_second = data_xyz * torch.roll(data_xyz, shifts=1, dims=2) #xz,yz,zy
        return torch.cat((data,pure_second,mixed_second), dim=2)

    elif moments_order == 3: #Upto 3 order moment
        xyz_shift_right = torch.roll(data_xyz, shifts=1, dims=2)
        xyz_shift_left = torch.roll(data_xyz, shifts=-1, dims=2)

        #Second Moments
        pure_second = data_xyz * data_xyz  # x^2,y^2,z^2
        mixed_second = data_xyz * xyz_shift_right  # xz,yz,z
        second_moments = torch.cat((pure_second,mixed_second), dim=2)

        pure_third = data_xyz * data_xyz * data_xyz  # x^3,y^3,z^3
        mixed_third1 = data_xyz * data_xyz * xyz_shift_right #x^2z,y^2x,z^y
        mixed_third2 = data_xyz * xyz_shift_right * xyz_shift_right #xz^2,x^y,y^z
        mixed_third3 = data_xyz * xyz_shift_right * xyz_shift_left # xyz, xyz,xyz
        third_moments = torch.cat((pure_third, mixed_third1,mixed_third2,torch.unsqueeze(mixed_third3[:,:,0],dim=2) ), dim=2)

        return torch.cat((data,second_moments,third_moments), dim=2)

    else:
        raise Exception('moments_order can be 1,2 or 3')

def knn(x, k):
    """
    Preforms KNN with given k to batch x
    Args:
        x: data [bs x N x 3]
        k: integer

    Returns:
        knn indexs for each point in a batch for each batch [bs x N x k]

    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # topK take k larges, for k smallest we take negative [bs x N x k]
    return idx

def getKnnDescriptor(x, k=20, idx=None):
    """
    Add to input data x knn distances for each point in each batch
    Args:
        x: point cloud data batch [bs x N x 3]
        k: integer
        idx: knn idxs result, if None finds them

    Returns:
        feature: [bs x N x k x 6] where [:,:,:,0:3] are xyz constant by dim=2 and [:,:,:,3:6] are distances to knn

    """
    #Change dims to Bs X 3 X N
    x = x.transpose(1,2)

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x.transpose(1,2), k=k)  # (batch_size, num_points, k)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((x, feature - x), dim=3).contiguous()

    return feature

def createUnifiedDescriptor(x,moment_order=2,use_Knn=True,k=20):
    """
    Creates Unified descriptor with the following options: with\without moments (upto 3 order), with\without normals, \with\without knn
    Args:
        x: [bs x N x 3/6] (6 if got normals); 3 dim: [x,y,z,(Nx,Ny,Nz)]
        moment_order: 1,2,3
        use_Knn: use Knn or not
        k: knn k value

    Returns:
        output sizes:
            - 1 order moments [bs x N x 3/6]; 3 dim: [x,y,z,(Nx,Ny,Nz)]
            - 1 order moments + knn [bs x N x k x 6/9]; 3 dim: [x,y,z,(Nx,Ny,Nz),Knn_dist_x,Knn_dist_y,Knn_dist_z] were first 3/6 enries are constant for each k
            - 2 order moments [bs x N x 9/12]; 3 dim: [x,y,z,(Nx,Ny,Nz),second_order_moments(6)]
            - 2 order moments + knn [bs x N x k x 12/15]; 3 dim: [x,y,z,(Nx,Ny,Nz),second_order_moments(6),Knn_dist_x,Knn_dist_y,Knn_dist_z] were first 9/12 enries are constant for each k
            - 3 order moments [bs x N x 19/22]; 3 dim: [x,y,z,(Nx,Ny,Nz),second_order_moments(6),third_order_moments(10)]
            - 3 order moments + knn [bs x N x k x 22/25]; 3 dim: [x,y,z,(Nx,Ny,Nz),second_order_moments(6),third_order_moments(10),Knn_dist_x,Knn_dist_y,Knn_dist_z] were first 19/22 enries are constant for each k

    """
    batch_size, num_points, num_dims = x.size()

    if num_dims == 6: #Check if data with normals
        use_Normals = True
    else:
        use_Normals = False

    if moment_order == 1:
        if use_Normals:
            if use_Knn: #No moments, with normal, with KNN
                knn_res = getKnnDescriptor(x[:,:,0:3], k=k, idx=None) #[bs x N x k x 6]
                x_expended = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
                return torch.cat((x_expended, knn_res[:,:,:,3:6]), dim=3).contiguous() # [bs x N x k x 9], dim3: [x,y,z,Nx,Ny,Nz,knn_dist]

            else: #No moments, with normals, no KNN
                return x # [bs x N x k x 6], dim3: [x,y,z,Nx,Ny,Nz]
        else:
            if use_Knn: #no moments, no normals, with KNN
                return getKnnDescriptor(x, k=k, idx=None)

            else: #No moments,no normals, no Knn
                return x # [bs x N x 3], dim3: [x,y,z]

    elif moment_order == 2 or moment_order == 3:  # Upto 2 / 3 order moment
        if use_Normals:
            if use_Knn: #upto second/third moments, with normal, with KNN
                knn_res = getKnnDescriptor(x[:,:,0:3], k=k, idx=None) #res [bs x N x k x 6]
                x_moments = getMoments(x,moments_order=moment_order)
                x_expended = x_moments.view(batch_size, num_points, 1, x_moments.size()[2]).repeat(1, 1, k, 1)
                return torch.cat((x_expended, knn_res[:,:,:,3:6]), dim=3).contiguous() # [bs x N x k x 15/25], dim3: [x,y,z,Nx,Ny,Nz,second_moments,(third_moments),knn_dist]

            else: #upto second/third moments, with normals, no KNN
                return getMoments(x,moments_order=moment_order) # [bs x N x k x 12/22], dim3: [x,y,z,Nx,Ny,Nz,second_moments,(third_moments)]
        else:
            if use_Knn: #upto second/third moments, no normals, with KNN
                knn_res = getKnnDescriptor(x, k=k, idx=None)  # res [bs x N x k x 6]
                x_moments = getMoments(x, moments_order=moment_order)
                x_expended = x_moments.view(batch_size, num_points, 1, x_moments.size()[2]).repeat(1, 1, k, 1)
                return torch.cat((x_expended, knn_res[:, :, :, 3:6]),
                                 dim=3).contiguous()  # [bs x N x k x 12/22], dim3: [x,y,z,second_moments(third_moments),knn_dist]

            else: #upto second/third moments,no normals, no Knn
                return getMoments(x, moments_order=moment_order)  # [bs x N x k x 9], dim3: [x,y,z,Nx,Ny,Nz]

    else:
        raise Exception('moments_order can be 1,2 or 3')

def get_patch_evd(input,k=20):

    bs, N, dim = input.size()
    #knn
    knn_res = getKnnDescriptor(input,k=k) #[bs x N x k x6] last dim: 0:2-x,y,z 3:5- knn dists
    xyz, knn_dist = torch.split(knn_res,3,3)
    ngbrs = knn_dist + xyz #[bs x N x k x 3]

    #Covarince matrix patch
    #Center by mean
    ngbrs_centered = ngbrs - torch.mean(ngbrs,dim=2,keepdim=True).repeat(1, 1, k, 1)
    M = ngbrs_centered.reshape((-1,k,3)) #[(bs*N) x k x 3]
    #Cov Tensor
    cov_tensor = torch.bmm(M.transpose(1,2),M) * (1/k) #[(bs*N) x 3 x 3]
    cov_tensor = cov_tensor.reshape((bs, N, 3,3)) #[bs x N x 3 x 3]

    #EVD - Done On CPU, very slow on GPU for small matrcies
    eigenvalues, _ = torch.symeig(cov_tensor.to('cpu'), eigenvectors=False)  #[bs x N x 3]
    eigenvalues = eigenvalues.to('cuda:0')

    curv = eigenvalues[:, :, 0].reshape((bs, N, 1)) / eigenvalues.sum(dim=2, keepdim=True) #[bs x N x 1]

    return curv



#Loaders
def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label,normal)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
