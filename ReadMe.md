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
    ./donwload_se_model.sh
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
python3 script/edge_detection_test.py 
```
