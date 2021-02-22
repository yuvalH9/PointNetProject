import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

#Parameters
NumOfEpocs = 100
#train_file = os.path.join(BASE_DIR, 'train.py')
train_file = 'train.py'

#Momenet Classic (No Knn) - 2 order moments + Normals
print('momenet_with_normals With {} Epoces'.format(NumOfEpocs))
os.system("python " + train_file + " --max_epoch {} --model \"{}\" --use_knn {} --test_name \"{}\"".
          format(NumOfEpocs,"momenet_with_normals",0,"Momenet_Test_NoKnn_WithNormals"))

#Momenet Classic (No Knn) - 3 order moments
print('momenet_cls 3 Order With {} Epoces'.format(NumOfEpocs))
os.system("python " + train_file + " --max_epoch {} --model \"{}\" --use_knn {} --test_name \"{}\" --moment_order {}".
          format(NumOfEpocs,"momenet_cls",0,"Momenet_Test_NoKnn_3Order", 3))

#Momenet Classic (No Knn) - 3 order moments + Normals
print('momenet_with_normals With 3 Order {} Epoces'.format(NumOfEpocs))
os.system("python " + train_file + " --max_epoch {} --model \"{}\" --use_knn {} --test_name \"{}\" --moment_order {}".
          format(NumOfEpocs, "momenet_with_normals", 0, "Momenet_Test_NoKnn_WithNormals_3Order", 3))

#Momenet Classic (No Knn) - 2 order moments
print('momenet_cls With {} Epoces'.format(NumOfEpocs))
os.system("python " + train_file + " --max_epoch {} --model \"{}\" --use_knn {} --test_name \"{}\"".
          format(NumOfEpocs,"momenet_cls",0,"Momenet_Test_NoKnn"))