import torch
import torch.nn as nn
import torch.nn.functional as torch_func
import provider

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tnet(nn.Module):
    def __init__(self,input_size=3):
        super(Tnet, self).__init__()
        self.input_size = input_size
        self.conv1D_1 = nn.Conv1d(self.input_size, 64, 1)
        self.conv1D_2 = nn.Conv1d(64, 128, 1)
        self.conv1D_3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.input_size * self.input_size)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(64)

    def forward(self,input):

        #Get Batch Size
        batch_size = input.size(0)

        #Tnet Arch
        net = torch_func.relu(self.bn1(self.conv1D_1(input)))
        net = torch_func.relu(self.bn2(self.conv1D_2(net)))
        net = torch_func.relu(self.bn3(self.conv1D_3(net)))
        net = nn.MaxPool1d(net.size(-1))(net)
        net = nn.Flatten(1)(net)
        net = torch_func.relu(self.bn4(self.fc1(net)))
        net = torch_func.relu(self.bn5(self.fc2(net)))

        #Init
        T_bais = torch.eye(self.input_size, requires_grad=True).repeat(batch_size,1,1)
        if net.is_cuda:
            T_bais = T_bais.cuda()

        T_transform = self.fc3(net).view(-1,self.input_size,self.input_size) + T_bais

        return T_transform


class PointNet_Classic(nn.Module):
    def __init__(self,numOfClasses=40, reg_alpha= 0.001):

        super(PointNet_Classic, self).__init__()
        self.reg_alpha = reg_alpha
        self.numOfClasses = numOfClasses

        self.input_transform = Tnet(input_size=3).to(DEVICE)
        self.feature_transform = Tnet(input_size=64).to(DEVICE)

        self.conv1D_1 = nn.Conv1d(3,64,1)
        self.conv1D_2 = nn.Conv1d(64, 128, 1)
        self.conv1D_3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, self.numOfClasses)

        self.dropout = nn.Dropout(p=0.3)
        self.logSoftMax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        #Input Transform
        T_mat_3x3 = self.input_transform(input.transpose(1,2))
        net = torch.bmm(input,T_mat_3x3)

        #MLP 64x64
        net = torch_func.relu(self.bn1(self.conv1D_1(net.transpose(1,2))))

        # Feature Transform
        T_mat_64x64 = self.feature_transform(net)
        net = torch.bmm(net.transpose(1,2), T_mat_64x64)

        #MLP 64x128x1024
        net = torch_func.relu(self.bn2(self.conv1D_2(net.transpose(1,2))))
        net = torch_func.relu(self.bn3(self.conv1D_3(net)))

        #Max Polling
        net = nn.MaxPool1d(net.size(-1))(net)
        net = nn.Flatten(1)(net)

        #FC
        net = torch_func.relu(self.bn4(self.fc1(net)))
        net = torch_func.relu(self.bn5(self.dropout(self.fc2(net))))
        output = self.fc3(net)

        return self.logSoftMax(output), T_mat_3x3, T_mat_64x64 #We want loss on all componnets including Tnets

    def get_model_loss(self, predicts, labels, T_mat_3x3, T_mat_64x64):
        nll_loss = torch.nn.NLLLoss()
        batch_size = predicts.size(0)

        #Tnets Loss
        T_mat_3x3_diff = torch.eye(3,requires_grad=True).to(DEVICE).repeat(batch_size,1,1)\
                         - torch.bmm(T_mat_3x3, T_mat_3x3.transpose(1,2))
        T_mat_64x64_diff = torch.eye(64, requires_grad=True).to(DEVICE).repeat(batch_size, 1, 1) \
                         - torch.bmm(T_mat_64x64,T_mat_64x64.transpose(1, 2))
        tnet_reg_loss = torch.norm(T_mat_3x3_diff) + torch.norm(T_mat_64x64_diff)

        return nll_loss(predicts,labels.long()) + self.reg_alpha * tnet_reg_loss

class Momenet(nn.Module):
    def __init__(self,numOfClasses=40, k=0, use_normals=False,moment_order=2, use_lifting=False):
        super(Momenet, self).__init__()
        self.moment_input_increment_dict = {2:6, 3:16} #{moment_order:input_increment_value}
        self.use_normals = use_normals
        self.numOfClasses = numOfClasses
        self.input_size = 6 if self.use_normals else 3 # 6 channels for normals else 3
        self.k_nn = k
        self.use_knn = True if k >0 else False
        self.moment_order = moment_order
        self.use_lifting = use_lifting
        self.lifting_size = 1 if self.use_lifting else 0


        self.input_transform = Tnet(input_size=3).to(DEVICE)

        #Second Order Layer Components - use in case of use KNN only
        self.conv2D_1 = nn.Conv2d(self.input_size + self.moment_input_increment_dict[moment_order] + self.lifting_size + 3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(64)

        #Main net layers
        self.conv1D_1 = nn.Conv1d(64 if self.use_knn else self.input_size + self.moment_input_increment_dict[moment_order] + self.lifting_size, 64, 1) #Case using knn or otherwise
        self.bn2 = nn.BatchNorm1d(64)
        self.conv1D_2 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv1D_3 = nn.Conv1d(64, 64, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv1D_4 = nn.Conv1d(64, 128, 1)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv1D_5 = nn.Conv1d(128, 1024, 1)
        self.bn6 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.bn8 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.numOfClasses)
        self.logSoftMax = nn.LogSoftmax(dim=1)

    def forward(self,input):

        # Input Transform
        if self.use_normals: #With normals - applay same transform on normals
            input_xyz, input_norms = torch.split(input, 3, dim=2)
            T_mat_3x3 = self.input_transform(input_xyz.transpose(1, 2))
            xyz_tfromed = torch.bmm(input_xyz,T_mat_3x3)
            normals_tformed = torch.bmm(input_norms,T_mat_3x3)
            input_tformed = torch.cat((xyz_tfromed, normals_tformed), dim=2)

        else: #Without normals
            T_mat_3x3 = self.input_transform(input.transpose(1, 2))
            input_tformed = torch.bmm(input, T_mat_3x3)

        #Second order layer
        input_lifted = provider.createUnifiedDescriptor(input_tformed, moment_order=self.moment_order, k=self.k_nn,use_Knn=self.use_knn) #[bs x N x k x 12/15] 12/15 w/o normals
        if self.use_lifting:
            curv_desc = provider.get_patch_evd(input_lifted[:,:,0:3]).detach()
            input_lifted = torch.cat((input_lifted, curv_desc), dim=2).contiguous()


        if self.use_knn:
            ##MLP 64x64
            net = torch_func.relu(self.bn1(self.conv2D_1(input_lifted.transpose(1, 3)))) #out [bs x N x 64 x k]
            ##Max polling
            #net = nn.MaxPool1d(net.size(-1))(net) #out [bs x N x 64 x k]
            net = net.max(dim=2, keepdim=False)[0]
        else:
            net = input_lifted.transpose(1,2)

        #Main Momenet
        #MLP 64x64
        net = torch_func.relu(self.bn2(self.conv1D_1(net))) #out [bs x 64 x N ]
        #MLP 64x64
        net = torch_func.relu(self.bn3(self.conv1D_2(net))) #out [bs x 64 x N ]
        # MLP 64x64
        net = torch_func.relu(self.bn4(self.conv1D_3(net)))  #out [bs x 64 x N ]
        # MLP 64x128
        net = torch_func.relu(self.bn5(self.conv1D_4(net)))  #out [bs x 128 x N ]
        # MLP 128x1024
        net = torch_func.relu(self.bn6(self.conv1D_5(net)))  #out [bs x 1024 x N ]
        #Max Polling
        net = nn.MaxPool1d(net.size(-1))(net)
        net = nn.Flatten(1)(net)

        # FC
        net = torch_func.relu(self.bn7(self.fc1(net)))
        net = torch_func.relu(self.bn8(self.dropout(self.fc2(net))))
        output = self.fc3(net)

        return self.logSoftMax(output)

    def get_model_loss(self, predicts, labels):
        nll_loss = torch.nn.NLLLoss()

        return nll_loss(predicts,labels.long())









































