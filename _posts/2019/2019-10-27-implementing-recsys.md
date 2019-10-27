---
layout: article
title: "ALS, BPR을 직접 구현해봤다."
category: "recommender systems"
tag: "recommender systems"
mathjax: true
---

## 잡담
~~유명~~ 추천 시스템 라이브러리 2개에 involve되어 있긴 한데, 생각해보니 추천 시스템 알고리즘을 스크래치부터 구현해 본 적이 거의 없는 것 같다. Logistic matrix factorization을 implicit에 구현해보긴 했는데...

그래서, 과연 나는 추천 시스템 알고리즘을 충분히 빠르게 구현할 수 있을까? 하는 생각이 들었다. 생각한 김에 바로 해봤다. 주말에 할 게 없기도 했었고...


## 사용할 언어/프레임워크 선택

#### 언어
인터페이스 부분은 파이썬으로 진행하기로 결정했다. Numpy Array가 무척 편하기도 하고, scipy.sparse_matrix를 그대로 이용할 수 있다는 장점이 있기 때문이다.

사실, 중요한건 Backend였는데, 내가 약간이라도 경험해 본 기술을 사용하는게 유리할 것 같아서, 선택지는 사실 두 가지 밖에 없었다.

1. Cython
2. C++ with Cython wrapper

효율적으로 구현했다면, C++이 당연히 속도가 빠르긴 할 것 같다. 다만, 내 C++ 숙련도보다, Cython을 좀 더 편하게 사용할 수 있을 것 같긴 했다. 하지만 이쯤 해서 C++ class나 function을 python에서 불러보는 걸 해보고 싶어졌고, C++를 다 까먹어버린 것 같아서 한번 C++을 리프레시 한다는 느낌으로 C++ with Cython wrapper로 진행하기로 했다.

#### 수치 연산 라이브러리
[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)이 가장 좋다고 알고 있었으므로, Eigen을 사용하기로 했다. 코드를 짜면서 찾아봤는데, [blaze-lib](https://bitbucket.org/blaze-lib/blaze)같은 옵션도 있었다고 한다. 다만 Eigen이 더 보편적인 것 같아 Eigen을 선택. 사실 대부분 수치 연산 라이브러리는 크게 차이나지 않을 것 같아 어떤 걸 선택해도 그리 큰 차이는 없을 것 같다.

> Eigen에 Conjugate gradient가 있어서 편리했다.

## 구현

구현하다 생긴 몇가지 귀찮은 점을 정리해봤다. 나중에 비슷한 일을 하게 된다면 이 점을 참고해서 빠르게 할 수 있을 것 같다.

### 1. Wrapping의 번거로움
[Cython 튜토리얼](https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html)에서 C++ class를 가져와 사용할 때, 권장되는 방법은 다음과 같았다.

```python
import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "src/cals.hpp" namespace "Algo":
    cdef cppclass CALS:
        CALS(float, float, float, int, int)
        void init_params(float*, float*, int, int, int)
        void update(int*, int*, float*, bool)
cdef class ALS_solver:
    cdef CALS *obj

    def __cinit__(self, alpha, reg_u, reg_i, num_threads, seed):
        self.obj = new CALS(alpha, reg_u, reg_i, num_threads, seed)

    def init_params(self, float[:, :] U, float[:, :] I,):
        n_users = U.shape[0]
        n_factors = U.shape[1]
        n_items = I.shape[0]
        self.obj.init_params(&U[0, 0], &I[0, 0], n_users, n_items, n_factors)

    def update(self,
               np.ndarray[int, ndim=1] indices,
               np.ndarray[int, ndim=1] indptr,
               np.ndarray[float, ndim=1] data,
               bool user_side):

        return self.obj.update(&indices[0], &indptr[0], &data[0], user_side)

    def __dealloc__(self):
        del self.obj
```

1. C++ class를 extern으로 선언해준 뒤에,
2. 이 class를 property로 갖는 wrapper class를 하나 선언해서, 변수 타입 변환 등 귀찮은 일을 처리한다.

근데, **엄청 귀찮은게** 내가 C++ 클래스/함수를 수정하려고 하면 다음과 같은 과정을 거쳐야 한다는 점이었다.

1. 헤더에서 함수 시그니쳐를 바꿔줌
2. 코드에서 함수 내용을 고침
3. Cython warpping 부분을 고침
4. python에서 함수 호출하는 부분을 고침

한가지 고치고 싶은데, 네번이나 비슷한 작업을 해야해서 짜증났다.

### 2. Package/ Package directory
어느 디렉토리에 설치되고, 어떤 shared object File로 저장되는지 아직도 헷갈려 미치겠다.
그냥 될 때까지 trail and error식으로 고치고 고치고 고치다가 겨우 되는 걸 하나 (우연히) 찾은 것 같다. 이 부분은 아직 잘 모르겠는데... 도저히 어딜 봐야 참고가 될 지 도무지 잘 이해가 가지 않는다.

> 시간이 되면 implicit이나, buffalo의 setup.py 부분을 잘 이해해두자. 이런거 한번 대충 됬다고 그냥 넘어가는 습관이 내 안좋은 점인 것 같다.

어떤 파일을 어디에 두는지... 이 부분도 어려웠던 것 같다. 새 토이 프로젝트를 진행할 때, 디렉토리 구조를 잘 잡아두는게 현명하다는 사실을 알게 되었다. 막상 큰 규모의 프로젝트는 정말 경험이 없었기 때문에, 짧은 기간이지만 많은 도움이 되었던 것 같다. 사실, 이런 부분은 한번 만들어놓고 잘 고치지 않기 때문에... 주로 다른 사람이 해놓은 걸 사용하기 때문에, 스크래치부터 만들어 볼 기회가 전혀 없었어서...

### 3. openMP
너무 편했다. thread 간 overwrite가 없다고 해야 하나(이걸 뭐라고 부르는지 까먹었다). 암튼, 같은 Object에 write하지 않는 부분은 한번에 병렬화가 되는 점이 너무 편했다.

ALS 알고리즘에서는, user $u$와 $u'$의 업데이트에 겹치는 parameter가 없기 때문에, 동시에 업데이트를 진행할 수 있다. 이 부분을 for loop로 구현하고, openMP parallel을 선언만 해주면, 이 부분을 알아서 병렬처리 해준다. 완전 신기했다. 이런 거 쓸 일이 별로 없었어서, 이런 기능이 있다는게 너무 신기했다...

ALS update loop는 다음과 같다.
```
    #pragma omp for schedule(dynamic, 4)
    for(int u = 0; u < n_users; ++u){
        auto num_pos_items = indptr[u + 1] - indptr[u];
        if(0 == num_pos_items)
            continue;

        rowMatrix temp(num_pos_items, n_factors);
        rowMatrix temp2(num_pos_items, n_factors);
        colVector y(n_factors, 1);

        temp.setZero();
        temp2.setZero();
        y.setZero();

        for(int idx = indptr[u], t = 0; idx < indptr[u + 1]; ++idx, ++t){
            int i = indices[idx];
            float val = data[idx];
            temp.row(t) = val * I.row(i);
            temp2.row(t) = I.row(i);
            y.noalias() += (I.row(i).transpose() * (1.0 + val * _alpha));
        }

        rowMatrix YtCY = ItI + (temp.transpose() * temp2 * _alpha);
        for(int d = 0; d < n_factors; ++d)
            YtCY(d, d) += reg;
        cg.setMaxIterations(3).compute(YtCY);
        U.row(u).noalias() = cg.solve(y);
    }

```

### 느낀 점
예전에 비슷한 시도를 한 번 해봤던 것 같은데, 그 때는 무참히 실패했었는데, 지금은 어떻게 구현할 수 있게 된 것 같다. 실력이 는 것일까, 끈기가 는 것일까. 가끔 이렇게 성장했다는 사실을 확인하는 기회가 생겨서 기쁘다. 앞으로도 더 좋은 개발자가 될 수 있게 노력해야겠다.

구현체는 [요기](https://github.com/ita9naiwa/cython-prac) 있다.
