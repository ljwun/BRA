# BRA
> 全名為行為風險評估（behavioral risk assessment），提供了一套管線（pipeline）將作為輸入的影像，轉換為各項評估資料並輸出，目前支援(1).人數(2).社交距離、(3).口罩檢查、(4).清潔行為檢查。

---
## Third Party Reference
+ **[Megvii-BaseDetection](https://github.com/Megvii-BaseDetection) / [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)**
+ **[ifzhang](https://github.com/ifzhang) / [ByteTrack](https://github.com/ifzhang/ByteTrack)**
+ **[JDAI-CV](https://github.com/JDAI-CV) / [fast-reid](https://github.com/JDAI-CV/fast-reid)**
+ **[NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT) / [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)**

---
## Prerequisites
+ Python >=3.8 `Tested on 3.8 and 3.9`
+ CUDA `Tested on 10.2 and 11.3`
+ FFmpeg or Gstreamer
  
if need
+ TensorRT8 `Tested on v8.2 and v8.4`

---
## Usage Step
#### 1.下載
```bash
git clone --recurse-submodules https://github.com/YyuK-Liao/BRA
```
#### 2.準備Python所需環境
```bash
cd BRA
# 建立一個python虛擬環境
# 「.dev-env」是虛擬環境的名字，可以自己改
python -m venv .dev-env

# 啟用虛擬環境
# windows
.dev-env/Scripts/activate
# GNU/Linux
source .dev-env/bin/activate

# 安裝套件
python -m pip install -r requirement.txt
```
#### 3.準備需要的文件
+ 模型的權重
  > 可以參考如下的預訓練權重
  + 行人
    + [ByteTrack](https://github.com/ifzhang/ByteTrack/#Model-zoo)
    + [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/#Benchmark)
  + 口罩（[Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)）
    + [預訓練YOLOX-m](https://drive.google.com/file/d/1qOaoiGO9il8XAiR9j__pJKmISfwWWIf7/view?usp=sharing)
    + [預訓練YOLOX-nano](https://drive.google.com/file/d/1gyzccGYATmPAnF8SiT8_Qe-twUd56ki4/view?usp=sharing)
    + 使用tools/train.py來訓練，可參考exps/FaceMask_nano.py來訓練COCO格式的資料、參考exps/FaceMask_m.py來訓練VOC格式的資料。
  + reid
    + [fast-reid](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md/MSMT17-Baseline)
+ view-config
  > 設定文件可參考work/xxx_conf.yaml，記錄了計算社交距離所需要的座標轉換參數，以及清潔行為檢查事先做的設定，包含定義區域、定義檢查時長。
  
  這份設定文件可以透過tools/IConFiguration.py來進行設定，操作步驟可參考[這裡](view_config.md)。
+ worker-config
  > 設定文件可參考worker/worker3_conf.yaml，記錄了worker需要的參數，包含了主要管線的各式開關參數，以及模型或演算法所需參數、權重檔案等。

  這份設定檔案需要手動撰寫，建議複製worker/worker3_conf.yaml後再修改：
  ```bash
  cp worker/worker3_conf.yaml worker/custom_worker_conf.yaml
  vim worker/custom_worker_conf.yaml
  ```

#### 3.EX 如果需要TensorRT
<details><summary>[詳細]</summary>

  > 要以TensorRT來執行BRA的推論工作的話，需要TensorRT Python API和torch2trt的python模組。
  
  TensorRT的安裝教學可以直接參考[NVIDIA DOCUMENT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)；若是使用Jetson，安裝JetPack即可，JetPack已經附帶了包含TensorRT在內的相關資料，但若需要以不同於JetPac自帶的Python版本的話，需要重新對TensorRT做跨語言的API綁定，具體流程可參考[這裡](https://github.com/mlcommons/inference_results_v2.0/issues/2#issuecomment-1133845520)和[這裡](https://github.com/NVIDIA/TensorRT/tree/main/python)
  ```bash
  export EXT_PATH=~/external
  mkdir -p $EXT_PATH && cd $EXT_PATH
  git clone https://github.com/pybind/pybind11.git -b v2.6.2 --depth=1

  # python原始碼可以從https://www.python.org/downloads/source/ 找
  # 或是用以下的指令來下載，3.x.y需替換成現在用的版本
  wget https://www.python.org/ftp/python/3.x.y/Python-3.x.y.tgz
  tar xvzf Python-3.x.y.tgz
  mkdir python3.x
  cp -r Python-3.x.t/Include/ python3.x/include
  # 如果沒有的話，可以用find指令來找pyconfig.h
  cp /usr/include/aarch64-linux-gnu/python3.x/pyconfig.h python3.8/include/

  # 複製TensorRT OSS，版本和內建JetPack的TensorRT版本一樣就好了
  git clone https://github.com/NVIDIA/TensorRT -b X.Y.Z --depth=1
  cd TensorRT/python
  # 直接執行build.sh即可建立.whl檔案
  # 需要確認一下參數有沒有起作用，沒作用的話可以先編輯build.sh
  CPATH=$EXT_PATH/pybind11/include TRT_OSSPATH=$EXT_PATH/TensorRT \
  PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=8 TARGET=aarch64 \
  ./build.sh
  # 如果編譯過程有跳缺少標頭檔的錯誤。可以先用find指令來找，在複製進TensorRT/include對應的資料夾底下，再重新執行build.sh

  # 需要先啟動python虛擬環境在安裝
  python -m pip install build/dist/tensorrt-*.whl

  # 可以使用python import來測試
  python -c "import tensorrt"
  ```
  確認TensorRT Python API後就可以安裝torch2trt了，這邊提醒一點，目前測試下來 **[NVIDIA-AI-IOT](https://github.com/NVIDIA-AI-IOT) / [torch2trt@5405207](https://github.com/NVIDIA-AI-IOT/torch2trt/tree/540520700f969e13b921be1bb944c44d299ff406)** 不支持對YOLOX進行動態batch_size的轉換，建議使用 **[jaybdub](https://github.com/jaybdub) / [torch2trt@tensor_shape_support](https://github.com/jaybdub/torch2trt/tree/tensor_shape_support)**，python模組安裝流程如下：
  ```bash
  # 老樣子，需要先啟動python的虛擬環境
  git clone -b tensor_shape_support --depth=1 https://github.com/jaybdub/torch2trt
  cd torch2trt
  python ./setup.py install
  ```
  確認所有工具鏈都安裝好後，就可以透過tools/trt.py來對YOLOX模型進行轉換
  ```bash
  # 啟動python的虛擬環境
  # 要注意的是-C 並不會幫忙創建資料夾，需要手動建立
  python ./tools/trt.py -f $EXP位置 -c $權重位置 -b $最大批次大小 -P $生成檔案的前綴名稱 -C 儲存資料夾 -w 工作區域大小
  # 工作區域大小是1的向左偏移數量，也就是說30是1G=1<<30，32是4G
  # 最後會在儲存資料夾底下生成兩個檔案，分別為前綴.engine和前綴.pth
  # engine是給原始的tensorrt模組調用的，pth是給torch2trt調用的，torch2trt可以保證推論流程和pytorch一樣
  ```
</details>

#### 4.執行
```bash
cd work
python ./work_dev.py $ARGS
```
ARGS的選項有很多，我們可以類型來分類
##### + 輸入輸出
|短前綴|長前綴|參數|意義|
|------|------|------|------|
|-vin |--video_input  |輸入源||
|-vout|--video_output |輸出源||
|-s   |--output_scale |(寬度):(長度)|輸出的影像尺寸，使用-1可以自斷推斷|
|-so  |--stream_output|串流輸出源||
|-csv |--write_to_csv |用來儲存每個時間的計算結果的csv檔||

##### + 日誌
|短前綴|長前綴|參數|意義|
|------|------|------|------|
|-vlog|--vout_log|輸出源日誌檔案||
|-slog|--stream_log|串流輸出源日誌檔案||

##### + 流程的控制參數
|短前綴|長前綴|參數|意義|
|------|------|------|------|
||--legacy||只處理一次輸入源就結束|
|-b   |--batch_size   |最大的一次執行批次數 ||
|-fps |--fps          |浮點數               |預設是從輸入源的header||
|-ss  |--start_second |hh:mm:ss.ms 或秒數   |輸入源的起始位置||
|-d   |--duration     |hh:mm:ss.ms 或秒數   |預計的處理長度||

##### + 需要的設定檔
|短前綴|長前綴|參數|意義|
|------|------|------|------|
|-vc  |--view_config|view_config設定文件||
|-wc  |--worker_config|worker_config設定文件||
|-worker|--worker_file|worker文件|默認是使用worker/worker3.py||

##### + 其他
|短前綴|長前綴|參數|意義|
|------|------|------|------|
|-en|--output_encoder|ffmpeg編碼器|可以使用`ffmpeg -codecs來查詢`|
||--log_level|ERROR、WARN、INFO、DEBUG、TRACE|work的輸出日誌層級|
||--io_backend|FFMPEG、GSTREAMER|輸入和輸出所要使用的後端|
|-h|--help||列出所有參數|
