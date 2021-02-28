# PointNetProject
Final project of Geometric Learning Course - Technion 048865


##Introducion
This project was implemented by PyTorch. Models are traind on ModelNet40 dataset. A parser is use for the train and eval input (insturction below).
Some new lifting enhancmented are suggested as part of this project, including: use of point normals, use 3 order moments and use of points curvature.
Two basic models are available:
1. PointNet cls 
2. Momenet cls

Use `Train.py` to train the desired model. Use `evaluate.py` to test a trained model.

## Train Running Instructions
Available models:
1. `pointnet_cls` - Basic PointNet classification network [1]
2. `momenet_cls` - Basic Momenet classification network [2]
3. `momenet_with_normals` - Momenet classification network using normals as well.

###Train Input Params

| Parameter Name | Description | Possible values
| --- | --- | --- |
| `model`     | used model type         | `pointnet_cls \   momenet_cls \    momenet_with_normals `    |
| `test_name` | save name for logs | string|
|`log_dir` | path to log dir | string
|`num_point` | Num of point in point cloud | `[256/512/1024/2048] [default: 1024]`|
|`max_epoch` | Number of epochs | int|
|`batch_size` | train batch size | `int [default: 32]`
|`use_knn` | Use Knn | `0- False 1 - True`|
|`moment_order`| Maximal moment order| `1 / 2 / 3`|
|`use_lifting`| Use curvature lifting | `0- False 1 - True`|


### Train Usage Example
`python train.py --max_epoch 100 --model "momenet_with_normals" --use_knn 1 --test_name "TstName" --use_lifting 1  --moment_order 3`

## Evaluate Running Instructions
You can choose one of the pretrained models saved as `.pth` files.

###Evaluate Input Params

| Parameter Name | Description | Possible values
| --- | --- | --- |
| `model`     | used model type         | `pointnet_cls \   momenet_cls \    momenet_with_normals `    |
| `test_name` | save name for logs | string|
|`log_dir` | path to log dir | string
|`num_point` | Num of point in point cloud | `[256/512/1024/2048] [default: 1024]`|
|`use_knn` | Use Knn | `0- False 1 - True`|
|`moment_order`| Maximal moment order| `1 / 2 / 3`|
|`use_lifting`| Use curvature lifting | `0- False 1 - True`|
|`model_path`| Path to a `.pth` file | string


### Evaluate Usage Example
`python train.py --model_path "weights/momenet_3Order.pth" --model "momenet_cls"`

### List Of Saved Modles
| Model Weights File | Description |
| --- | --- |
| a | b|

## Accuracy Comparison
## Refrences
[1] Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas, "Pointnet: Deep learning on point sets for 3d classification and segmentation".Proceedings of the IEEE conference on computer vision and pattern recognition, p652--660,2017.

[2] Joseph-Rivlin, Mor and Zvirin, Alon and Kimmel, Ron, "Momen (e) t: Flavor the moments in learning to classify shapes",Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops,2019
 
