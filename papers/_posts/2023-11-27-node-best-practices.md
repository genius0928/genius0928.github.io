---
layout: post
research-area: BE/node.js
title: "Node.js와 MongoDB의 프로젝트팁"
slug: 블로그
description: 
publisher: 주홍철
publisher-fullname: 나와유 배방점 모임 / 2023.12.02
authors:
  - 주홍철:어비스  
paper: https://arxiv.org/pdf/2305.10823.pdf
code: 
tag:
---

# 서론

MongoDB와 Node.js를 이용해서 서비스를 만들 때 코드리팩토링은 반드시 필요하다. 그렇다면 어떻게 해야할까?  요새 주먹구구식으로 만들다가 열심히 리팩토링 중인데 죽겠다 ㅠㅠ  
![Alt text](/assets/img/001.png)  
# 본론
오류의 연속.. ㅠ  
> 1. 코드 모듈화
모델, 컨트롤러, 유틸 등 폴더기반 모듈화를 해야 함.  라우터 분리  