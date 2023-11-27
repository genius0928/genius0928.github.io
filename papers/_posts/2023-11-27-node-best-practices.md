---
layout: post
research-area: BE/node.js
title: "Node.js Best Practices"
slug: 블로그 
publisher: 주홍철 
publisher-fullname: 천각 - 나와유 배방점 / 2023.12.02
authors:
  - 주홍철:어비스  
  - 주홍철:어비스  
paper: https://arxiv.org/pdf/2305.10823.pdf
code: https://github.com/wnghdcjfe
tag: node.js, mongodb
---

Node.js를 기반으로 앱을 만들 때 어떤 것을 지켜야할까?   
모듈화, 오류처리, 공격방어 등에 대해 알아보자.  
## 공격방어  
실제로 이런식으로 .env 등에 대한 파일을 스크래핑 하려고 많은 노력을 당하게 된다. 
이를 위해서 cloudfront를 앞단에 두어야 합니다.
![공격당하는중](/assets/img/20231127/공격당하는중.png)

## 안정적인 종료 
DB - 서버 종료시에 다음과 같이 안정적으로 종료를 해야 합니다.
```js
const gracefulShutdown = () => {
    console.log('Shutting down gracefully');
    server.close(() => {
        console.log('Server closed');
        client.close().then(() => {
            console.log('Database connection closed');
            process.exit(0);
        });
    });
};

// Handle termination signals
process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);
```

