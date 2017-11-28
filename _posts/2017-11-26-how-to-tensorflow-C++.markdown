---
layout: post
title:  "Tensorflow C++ 동적 링크드 라이브러리(stanalone C++ project) 만들기"
date:   2017-11-26 17:02:34 +0900
categories: tensorflow
tag : tensorflow
---


개요
이 문서는 Tensorflow를 리눅스 환경(ubuntu)에서 빌드하고, C++ interface를 사용하는 방법에 대해 다룬다.



컨텐츠

* TOC
{:toc}

## Overview
이하 설치 명령어 등은 우분투 17.04 x86_64 machine, Python 3.6, CUDA 8.0에서 CPU, GPU 환경에서 실험을 진행하고, 작동을 확인했다.

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
```
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel

```

Tensorflow 의존성
```
sudo apt-get install python3-pip python3-numpy swig python3-dev
sudo pip3 install wheel
sudo apt-get install tensorflow
```
## 텐서플로우 빌드
github에서 tensorflow source code를 가져온다.
```
git clone https://github.com/tensorflow/tensorflow tensorflow
cd tensorfloworflow
```

그 다음, 텐서플로우의 옵션을 설정한다.
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

## Protobuf, Eigen 설치
텐서플로우를 독립적으로 실행하기 위해서는 추가적으로 두개의 라이브러리를 더 설치해야 한다.

Protobuf  빌드
```
mkdir /tmp/proto
tensorflow/contrib/makefile/download_dependencies.sh
cd tensorflow/contrib/makefile/downloads/protobuf/
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

빌드가 완료된 라이브러리 파일들을 옮겨준다.
```
#프로젝트 디렉토리 생성
mkdir ../project_directory/

mkdir ../project_directory/lib
mkdir ../project_directory/include
#tensorflow
cp bazel-bin/tensorflow/libtensorflow_cc.so ../project_directory/lib/

#protobuf
cp /tmp/proto/lib/libprotobuf.a ../project_directory/lib/
```

다음의 include file도 옮긴다.
```

cp -r bazel-genfiles/* ../project_directory/include/
cp -r tensorflow/cc ../project_directory/include/tensorflow
cp -r tensorflow/core ../project_directory/include/tensorflow
cp -r third_party ../project_directory/include
cp -r /tmp/proto/include/* ../project_directory/include
cp -r /tmp/eigen/include/eigen3/* ../project_directory/include
```
설치 완료. 다음의 테스트 코드를 만들어 실행해본다.

## 테스트
헤더, 라이브러리 등을 옮긴 디렉토리에서, 테스트 코드를 만든다.
```
cd ../project_directory
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
g++ -std=c++11 -Wl,-rpath='$ORIGIN/lib' -Iinclude -Llib test.cpp -ltensorflow_cc -o exec
./exec
```
작동이 잘 된다면, 성공한 것이다.


### References
[](http://tuatini.me/building-tensorflow-as-a-standalone-project/)
- 사실 여기 있는 글을 그대로 번역한 것이다. 으음 부끄러워;;