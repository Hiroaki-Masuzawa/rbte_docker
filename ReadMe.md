# rBTE docker

# 準備
## 本リポジトリのクローン
```
git clone --recurcive https://github.com/Hiroaki-Masuzawa/rbte_docker.git
```

## サンプル画像ダウンロード
```
./download_image.sh
```

## 各モデルweightのダウンロード
- SEは以下コマンドでダウンロードする
    ```
    ./download_se_model.sh
    ```
- bdcn_modelについては[ここ](https://drive.google.com/file/d/1CmDMypSlLM6EAvOt5yjwUQ7O5w-xCm1n/view?usp=sharing)からダウンロードしてbdcn_modelフォルダ内に置く．
- nms用のモデルは以下コマンドでダウンロードする．
    ```
    wget http://ptak.felk.cvut.cz/im4sketch/Models/opencv_extra.yml.gz -O ./Pretrained_Models/opencv_extra.yml.gz
    ```

## docker image build
```
cd docker
./build.sh
```
# 実行
## exec sample
```
cd docker
./run.sh
```
```
python3 script/edge_detection_test.py --input /userdir/images/color/Lenna.bmp
```


# データセット作成
```
cd docker
./run.sh
```
```
mkdir dataset
python3 script/gen_edge_image.py /dataset/coco/annotations/instances_train2017.json --input_dir /dataset/coco/train2017 --device cuda --data_type train > dataset/train.csv
python3 script/gen_edge_image.py /dataset/coco/annotations/instances_val2017.json --input_dir /dataset/coco/val2017 --device cuda --data_type val > dataset/val.csv
mkdir dataset/test
```
dataset/testにテスト画像を入れる．

# 学習
```
python3 script/training_edge_image.py --model resnet50 --batchsize 64 --lr 1e-4 --ep 50  --output output/resnet50_tiny_w_aug
python3 script/training_edge_image.py --model resnet50 --batchsize 64 --lr 1e-4 --ep 50  --output output/resnet50_tiny_wo_aug --use_geometric false --use_thinnms false
python3 script/training_edge_image.py --model mvitv2_tiny --batchsize 64 --lr 5e-5 --ep 50  --output output/mvitv2_tiny_w_aug
python3 script/training_edge_image.py --model mvitv2_tiny --batchsize 64 --lr 5e-5 --ep 50  --output output/mvitv2_tiny_wo_aug --use_geometric false --use_thinnms false
python3 script/training_edge_image.py --model vit_small_patch16_224 --batchsize 64 --lr 5e-5 --ep 50 --output output/vit_small_w_aug 
python3 script/training_edge_image.py --model vit_small_patch16_224 --batchsize 64 --lr 5e-5 --ep 50 --output output/vit_small_wo_aug --use_geometric false --use_thinnms false
```
# 評価
```
python3 script/pred_edge_image.py --modelfile output_compare/resnet50_tiny_w_aug/model_final.pth 
python3 script/pred_edge_image.py --modelfile output_compare/resnet50_tiny_wo_aug/model_final.pth 
python3 script/pred_edge_image.py --modelfile output_compare/mvitv2_tiny_w_aug/model_final.pth 
python3 script/pred_edge_image.py --modelfile output_compare/mvitv2_tiny_wo_aug/model_final.pth 
python3 script/pred_edge_image.py --modelfile output_compare/vit_small_wo_aug/model_final.pth 
python3 script/pred_edge_image.py --modelfile output_compare/vit_small_w_aug/model_final.pth 
```
