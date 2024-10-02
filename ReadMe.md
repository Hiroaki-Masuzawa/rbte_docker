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
```
./donwload_se_model.sh
```
bdcn_modelについては[ここ](https://drive.google.com/file/d/1CmDMypSlLM6EAvOt5yjwUQ7O5w-xCm1n/view?usp=sharing)からダウンロードしてbdcn_modelフォルダ内に置く．

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
