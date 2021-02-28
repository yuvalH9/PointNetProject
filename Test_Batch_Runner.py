import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

#Parameters
NumOfEpocs = 100
#train_file = os.path.join(BASE_DIR, 'train.py')
train_file = 'train.py'

#Momenet With Knn
run_title1 = "Momenet With Knn"
run1 = {'max_epoch': NumOfEpocs,
        'model': "momenet_cls",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn",
        'moment_order': 2,
        'use_lifting': 0}

#Momenet With Knn With Normals
run_title2 = "Momenet With Knn"
run2 = {'max_epoch': NumOfEpocs,
        'model': "momenet_with_normals",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn_WithNormals",
        'moment_order': 2,
        'use_lifting': 0}

#Momenet With Knn 3Order
run_title3 = "Momenet With Knn 3 Order"
run3 = {'max_epoch': NumOfEpocs,
        'model': "momenet_cls",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn_3Order",
        'moment_order': 3,
        'use_lifting': 0}

#Momenet With Knn With Normals 3 Order
run_title4 = "Momenet With Knn With Normals 3 Order"
run4 = {'max_epoch': NumOfEpocs,
        'model': "momenet_with_normals",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn_WithNormals_3Order",
        'moment_order': 3,
        'use_lifting': 0}

#Momenet With Knn With Curvature
run_title5 = "Momenet With Knn With Curvature"
run5 = {'max_epoch': NumOfEpocs,
        'model': "momenet_cls",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn_WithCurv",
        'moment_order': 2,
        'use_lifting': 1}

#Momenet With Knn With Normals With Curvature
run_title6 = "Momenet With Knn With Normals With Curvature"
run6 = {'max_epoch': NumOfEpocs,
        'model': "momenet_with_normals",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn_WithNormals_WithCurv",
        'moment_order': 2,
        'use_lifting': 1}

#Momenet With Knn With Normals 3 Order With Curvature
run_title7 = "Momenet With Knn With Normals 3 Order With Curvature"
run7 = {'max_epoch': NumOfEpocs,
        'model': "momenet_with_normals",
        'use_knn': 1,
        'test_name': "Momenet_WithKnn_WithNormals_WithCurv",
        'moment_order': 3,
        'use_lifting': 1}

run_list = [run1, run2, run3, run4, run5, run6, run7]
run_title_list = [run_title1, run_title2,run_title3, run_title4, run_title5, run_title6, run_title7]

for run_id in range(len(run_list)):
    print(run_title_list[run_id] + "|Num Of Epoces: {}".format(NumOfEpocs))
    os.system(
        "python " +\
        train_file +\
        " --max_epoch {max_epoch} --model \"{model}\" --use_knn {use_knn} --test_name \"{test_name}\" --moment_order {moment_order} --use_lifting {use_lifting}".\
        format(**run_list[run_id]))
