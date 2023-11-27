# 천각 블로그
천재개발자들의 모각코 모임, 천각의 블로그입니다. 

## requirement 
 - https://rubyinstaller.org/downloads/
 - gem install jekyll bundler
 - bundle install
 - bundle exec jekyll serve

## 테스팅
```
bundle exec jekyll serve
```
## 빌드 
```
bundle exec jekyll build
``` 
## 본문 작성 가이드
### Front Matter 작성

<Details>
    <Summary>자세히 보기</Summary>

콘텐츠의 속성 정보를 YAML 형식에 따라 작성합니다.
    
| 이름 | 필수 | 설명 | 비고 |
| - | - | - | - |
| `layout` | O | 레이아웃 종류 | 논문(papers): `post`, 딥다이브(deepdive): `post-deepdive` |
| `use-katex` | X | 본문에서 `KaTeX` 사용 여부 | `true`, `false`(기본값) |
| `research-area` | O | 연구 분야 | (NLP, COMPUTER VISION, SPEECH/AUDIO) |
| `title` | O | 콘텐츠 타이틀 | |
| `slug` | O | 콘텐츠가 게시 될 경로 | 다른 콘텐츠의 `slug`와 중복될 수 없음. |
| `description` | O | 콘텐츠 한 줄 요약 | |
| `published-date` | O | 학회 시작일 또는 ArXiv 업로드 날짜 |  |
| `publisher` | O | 학회 또는 저널 이름 |  |
| `publisher-fullname` | O | 학회, 저널, 또는 워크샵 풀네임 |  |
| `authors` | O | 저자 및 소속 이름 | `저자명`:`(논문에 기재된)소속`의 리스트로 기재. 카카오 공동체 소속은 크루 이름, 그 외 소속은 한글 실명 |
| `paper` | X | 논문 링크 |  |
| `code` | X | GitHub Repo 링크 |  |
| `tag` | X | 관련 키워드 | 리스트로 기재. |

**예시**

```yaml
---
layout: post
use-katex: true
research-area: COMPUTER VISION
title: "A Plug-in Method for Representation Factorization in Connectionist Models"
slug: ieee2020-fden
description: "딥러닝 모델에서 추출한 임베딩 벡터를 독립 요인으로 분해하는 기법 ‘FDEN’ 제안"
published-date: 2021-02-10
publisher: IEEE Transactions on Neural Networks and Learning Systems
authors:
  - 윤재석:고려대학교
  - joshua:카카오엔터프라이즈
  - 석흥일:고려대학교
paper: https://arxiv.org/pdf/1905.11088.pdf
code: https://github.com/wltjr1007/Factors-Decomposer-Entangler-Network
tag:
  - disentangled representation
  - factorial representation
  
---
```
    
</Details>

### 이미지 삽입

<Details>
    <Summary>자세히 보기</Summary>

1. `assets/img/{작성 중인 마크다운 파일 이름}` 디렉토리를 생성합니다.
2. 생성된 디렉토리 안에 이미지를 `001.jpg`, `002.jpg`, ...의 이름으로 위치시킵니다.
3. 콘텐츠 본문에서 이미지 삽입을 원하는 위치에 다음과 같이 기술합니다.
  `{% include image.html name="{이미지 파일명}" width="{이미지 너비}" align="{좌우정렬 위치}" %}` 

**예시**
```liquid
{% include image.html name="002.png" width="70%" align="center" %}
```
    
</Details>

### 주석 삽입

<Details>
    <Summary>자세히 보기</Summary>

1. 콘텐츠 본문에서 주석 삽입을 원하는 위치에 `[^{주석번호}]` 형식으로 마킹합니다.
2. 본문에서 원하는 위치에 (문단 바로 아래 또는 본문 최하단) 주석번호 별 설명을 작성합니다. `[^1]: 시스템 설계자가 미리...`
3. 본문 최하단에 다음과 같이 주석 영역을 만듭니다.
    
```markdown
-----

### Footnotes
```   
    
</Details>

----

# License
This software is licensed under the Apache 2 license, quoted below.

Copyright 2021 Kakao Enterprise Corp. http://www.kakaoenterprise.com

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Contact

ai-research@kakaoenterprise.com
