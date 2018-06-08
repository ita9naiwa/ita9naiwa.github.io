---
layout: article
title:  "Tensorflow C++ 동적 링크드 라이브러리(stanalone C++ project) 만들기"
date:   2017-11-26 17:02:34 +0900
categories: tensorflow
tag : tensorflow
---

[이 글의 번역!](https://tuatini.me/building-tensorflow-as-a-standalone-project/)입니다.

가끔 텐서플로우를 C++에서 deploy해야 하는 경우가 있습니다. 주로, Tensorflow로 training한 모델을 python을 지원하지 않는 환경에서 사용하고 싶을 때 사용하는 것 같아요.
이 경우, 매번 bazel에서 빌드하지 않고 library를 빌드해 .so 파일로 만들어 사용하는 방법에 대해 다룹니다.
자세한 텐서플로우를 C++에서 사용하는 방법에 대한 가이드는 [https://www.tensorflow.org/api_docs/cc/](https://www.tensorflow.org/api_docs/cc/)을 참조해주세요.

[원본](https://tuatini.me/building-tensorflow-as-a-standalone-project/)에서는 Raspberry Pi에서의 설정까지 다루고 있지만, 제가 라즈베리 파이를 갖고 있지 않아서 실제로 실행을 못해보기 때문에 Ubuntu 환경에서 build하는 방법만 정리합니다. 또한, 원본 글에서는 Tensorflow **1.3.0** 버전을 사용하지만, 이 글에서는 Tensorflow **1.5.0** 버전을 사용합니다. 따라서, 사용하는 dependency가 조금 바뀝니다.

**Ubuntu 17.04, GCC 6.3.0에서 테스트했습니다.**
**원본 글은 Raspberry Pi, tensorflow 1.3.0 버전에서 테스트했지만, 저는 ubuntu 17.04 버전,tensorflow 1.5.0에서 테스트했습니다.**

--------

개요
이 문서는 Tensorflow를 리눅스 환경(ubuntu)에서 빌드하고, C++ interface를 사용하는 방법에 대해 다룬다.

## Overview
이하 설치 명령어 등은 다음과 같은 환경에서 테스트되었습니다. (본문과 살짝 다릅니다!)

- 우분투 17.04 x86_64 machine
- Python 3.6
- GCC(g++) 6.3.0
- CUDA 8.0에서 CPU, GPU 환경에서 실험을 진행하고, 작동을 확인했다.

## 빌드하기 위해 필요한 다른 프로그램 설치
먼저, 패키지 매니저 ```apt-get``` 를 업데이트한다. 우분투에서는 ```apt``` 를 사용하지만, 다른 리눅스 라이브러리에서는 ```yum``` 등 다른 명령어를 사용해야 한다.
```
sudo apt-get update
```
다음으로 linux 환경에서 C++ 프로그램을 빌드하기 위한 기본적인 프로그램을 설치한다.

Bazel (구글에서 관리하는 소프트웨어 설치 도구. 텐서플로우를 설치하는 데 사용한다) 의존성
```
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip default-jdk autoconf automake libtool
```


Bazel 설치

[https://github.com/bazelbuild/bazel/releases?after=0.8.1](https://github.com/bazelbuild/bazel/releases?after=0.8.1)
여기서 자신의 플랫폼에 맞는 Bazel 다운로드 후 설치해주세요.
(0.8.0 버전에서 테스트 되었습니다.)




Tensorflow 의존성
```
sudo apt-get install python3-pip python3-numpy swig python3-dev
sudo pip3 install wheel
sudo apt-get install tensorflow
```
## 텐서플로우 빌드
github에서 tensorflow source code를 가져온다.

```
git clone --recurse-submodules -b v1.5.0 https://github.com/tensorflow/tensorflow.git tensorflow
cd tensorflow
```

그 다음, 텐서플로우의 옵션을 설정한다.

** 1.5.0 버전에선 살짝 바뀌었지만, 적당히 Configuration을 참고해서 입력해주시면 됩니다!**
```
Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N]
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N]
Do you wish to build TensorFlow with VERBS support? [y/N]
Do you wish to build TensorFlow with OpenCL support? [y/N]
Do you wish to build TensorFlow with CUDA support? [y/N] Y
Do you want to use clang as CUDA compiler? [y/N]
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: 8.0
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 6
Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.0
Do you wish to build TensorFlow with MPI support? [y/N]
```

MKL 옵션은 intel CPU에서 부동소수점 빠른 연산을 지원하며,
OpenCl, CUDA 등은 GPU 연산을 지원하는 옵션이다.
자세한 옵션에 대해서는 https://www.tensorflow.org/install/install_sources#ConfigureInstallation 항목을 참조.
또한, GPU(CUDA) 설정에 관해서는 https://www.tensorflow.org/install/install_sources 을 참조.

모든 설정을 마쳤으면, 이제 텐서플로우를 빌드할 수 있다.

```
bazel build -c opt --verbose_failures //tensorflow::libtensorflow_cc.so
```

## Protobuf, Eigen, Nsync 설치
텐서플로우를 독립적으로 실행하기 위해서는 추가적으로 두개의 라이브러리를 더 설치해야 한다.

Protobuf  빌드
```
mkdir /tmp/proto
tensorflow/contrib/makefile/download_dependencies.sh
cd tensorflow/contrib/makefile/downloads/protobuf/
//이 과정은 외부 dependency를 설치하는 과정이며, "tensorflow/contrib/makefile/downloads"에 갖가지 라이브러리가 저장되어 있습니다.
./autogen.sh
./configure --prefix=/tmp/proto/
make
make install
```

Eigen 빌드
```
mkdir /tmp/eigen
cd ../eigen
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
make install
cd ../../../../../..
```

Nsync 빌드
```
mkdir /tmp/nsync
cd ../tensorflow/contrib/downloads/nsync
cmake -DCMAKE_INSTALL_PREFIX=/tmp/nsync
make install
```


빌드가 완료된 라이브러리 파일들을 옮겨준다.
```
#프로젝트 디렉토리 생성
mkdir ../tf_test/

mkdir ../tf_test/lib
mkdir ../tf_test/include
#tensorflow
cp bazel-bin/tensorflow/libtensorflow_cc.so ../tf_test/lib/
cp bazel-bin/tensorflow/libtensorflow_framework.so ../tf_test/lib

#protobuf
cp /tmp/proto/lib/libprotobuf.a ../tf_test/lib/

#nsync
cp -r /tmp/nsync/lib/libnsync.a ../tf_test/lib/
```

다음의 include file도 옮긴다.
```

cp -r bazel-genfiles/* ../tf_test/include/
cp -r tensorflow/cc ../tf_test/include/tensorflow
cp -r tensorflow/core ../tf_test/include/tensorflow
cp -r third_party ../tf_test/include
cp -r /tmp/proto/include/* ../tf_test/include
cp -r /tmp/eigen/include/eigen3/* ../tf_test/include
cp -r /tmp/nsync/include/* ../tf_test/include

```
설치 완료. 다음의 테스트 코드를 만들어 실행해본다.

## 테스트
헤더, 라이브러리 등을 옮긴 디렉토리에서, 테스트 코드를 만든다.
```
cd ../tf_test
vi test.cpp
```
이러한 ```test.cpp```파일을 만들어 컴파일 후 작동을 확인한다.
```
//test.cpp
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f}});
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f}});
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}
```
마지막으로 작성한 test.cpp을 컴파일하고, 실행을 확인한다.
```
g++ -std=c++11 -Wl,-rpath=lib -Iinclude -Llib -ltensorflow_framework test.cpp -ltensorflow_cc -ltensorflow_framework -o exec
./exec
```
작동이 잘 된다면, 성공한 것이다.


### References
[Building TensorFlow 1.3.0 as a standalone project (Raspberry pi 3 included)](http://tuatini.me/building-tensorflow-as-a-standalone-project/)
- 사실 여기 있는 글을 그대로 번역한 것이다. 으음 부끄러워;;
