---
layout: post
research-area: BE/node.js
title: "Node.js Best Practices"
slug: 블로그 
publisher: zagabi 
published-date: 천각8회차 / 2023.12.02
authors:
  - zagabi   
code: https://github.com/wnghdcjfe
tag: node.js, mongodb
---

Node.js로 서버를 구축할 때 어떤 것을 지키면 좋을까? Best Practices에 대해 알아봅시다.  

## 모듈화
data, db, routes, test, utils 등으로 모듈화를 해야 합니다.

```shell
myapp/
|-- node_modules/
|-- routes/
|   |-- index.js
|   |-- user.js
|-- controllers/
|   |-- userController.js
|-- models/
|   |-- userModel.js
|-- middleware/
|   |-- authMiddleware.js
|-- utils/
|   |-- index.js
|-- config/
|   |-- db.js
|-- test/
|   |-- user.test.js
|-- dist/ 
|-- views/
|   |-- front-end/ 
|-- app.js
|-- package.json

```

db 모듈
```js
const db = require('./db');
```

express 예시
```js
// userRoutes.js
const express = require('express');
const router = express.Router();
const { getUsers, addUser } = require('./userController');

router.get('/users', getUsers);
router.post('/users', addUser);

module.exports = router;

```

## 공격방어  
실제로 이런식으로 .env 등에 대한 파일을 스크래핑 하려고 많은 노력을 당하게 됩니다.
이를 위해서 cloudfront를 앞단에 두어야 합니다.
![공격당하는중](/assets/img/20231127/공격당하는중.png)

### cloudfront
Amazon CloudFront는 Amazon Web Services(AWS)에서 제공하는 콘텐츠 전송 네트워크(Content Delivery Network, CDN) 서비스입니다. 이 서비스는 전 세계에 분산된 서버 네트워크를 통해 사용자에게 웹 콘텐츠와 애플리케이션을 빠르고 안정적으로 전달합니다. 

 - 저지연 및 빠른 콘텐츠 전달: CloudFront는 콘텐츠를 사용자에게 더 빠르게 전달하기 위해 지능적으로 콘텐츠를 **캐시**합니다. 이는 웹 사이트나 애플리케이션의 응답 시간을 개선합니다.
 - 다양한 콘텐츠 지원: 정적 및 동적 웹 콘텐츠, 스트리밍 비디오, API 호출 등 다양한 유형의 콘텐츠를 지원합니다.
 - 보안: CloudFront는 SSL/TLS를 사용하여 데이터 전송 중인 콘텐츠의 보안을 강화합니다. 또한 AWS Shield Standard와 함께 DDoS 공격으로부터 보호합니다. 
 

## 안정적인 종료(graceful shutdown)
보통의 서버는 다음과 같이 구축이 되어있습니다. 
ctrl + c 등을 통해 서버가 종료될 때 클라이언트의 요청을 모두 처리한 이후에 종료하는 것이 좋습니다. 만약 이를 안한다면 서버를 종료할 경우 클라이언트가 응답을 받지 못합니다.  

> 서버 종료 -> DB 종료순으로.

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

## 테스트
단위테스트, 통합테스트, 린팅, 엔드투엔드 테스트가 대표적입니다. 
 - 단위 테스트: 비즈니스 논리에 맞는 단위 테스트를 작성하는 것.
 - 통합 테스트: 데이터베이스와의 상호 작용을 테스트하여 모든 것이 예상대로 작동하는지 확인하는 것.
 - 린팅: ESLint 또는 이와 유사한 도구를 사용하여 코드 품질과 일관성을 강화하는 것.

### 단위테스트
단위 테스트는 소프트웨어의 가장 작은 단위(주로 함수나 메소드)가 의도대로 작동하는지 확인하는 테스트입니다. JavaScript에서는 Jest, Mocha, Jasmine 등의 프레임워크를 사용하여 단위 테스트를 작성할 수 있습니다.
```js
// sum.js
function sum(a, b) {
  return a + b;
}

module.exports = sum;

// sum.test.js
const sum = require('./sum');

test('adds 1 + 2 to equal 3', () => {
  expect(sum(1, 2)).toBe(3);
});

```
### 통합테스트

통합 테스트(Integration Testing)는 여러 컴포넌트 또는 시스템의 서로 다른 부분이 함께 올바르게 작동하는지 확인하는 테스트 과정입니다. 단위 테스트가 개별 모듈의 기능을 검증하는 데 초점을 맞춘다면, 통합 테스트는 이러한 모듈들이 통합되었을 때 생기는 인터페이스와 흐름을 검증합니다.
```js
// test/user.test.js
const request = require('supertest');
const app = require('../app');

describe('GET /user/:id', () => {
  it('responds with json containing a single user', done => {
    request(app)
      .get('/user/1')
      .expect('Content-Type', /json/)
      .expect(200)
      .then(response => {
        expect(response.body).toEqual({ id: 1, name: 'John Doe' });
        done();
      });
  });

  it('responds with 404 not found for invalid user', done => {
    request(app)
      .get('/user/999')
      .expect(404, done);
  });
});
```


### 엔드투엔드 테스트
사용자의 관점에서 전체 애플리케이션의 흐름을 테스트하는 과정입니다. 이는 실제 사용자의 시나리오를 모방하여 시스템이 종단 간(end-to-end)으로 예상대로 작동하는지 확인합니다.

cypress를 이용한 테스팅
```js
describe('Login Test', () => {
    it('Visits the login page and logs in', () => {
        cy.visit('http://localhost:3000/login'); // 로그인 페이지 방문
        cy.get('input[name="username"]').type('user1'); // 사용자 이름 입력
        cy.get('input[name="password"]').type('password123'); // 비밀번호 입력
        cy.get('button[type="submit"]').click(); // 로그인 버튼 클릭

        cy.url().should('include', '/dashboard'); // 대시보드 페이지로 리디렉션되었는지 확인
    });
});
``` 
### 린팅
 - 코드 품질 향상: 린터는 코드의 일관성을 유지하고, 읽기 쉽고 유지보수하기 쉬운 코드를 작성하도록 도와줍니다.
 - 버그 예방: 초기 단계에서 잠재적인 오류나 버그를 찾아내어 추후 발생할 수 있는 큰 문제를 예방합니다.
 - 코드 스타일 일관성: 프로젝트 전체에서 일관된 코딩 스타일을 유지할 수 있게 도와줍니다. 이는 팀 작업에 특히 중요합니다.
 - 자동화된 코드 검토: 코드 리뷰 과정에서 사소한 스타일 이슈에 시간을 낭비하는 대신, 더 중요한 구조적 문제에 집중할 수 있게 해줍니다.


## 에러 핸들링
먼저 에러모니터링이 필요하다. winston 또는 Morgan과 같은 로깅 라이브러리를 사용하여 로깅하는데 미들웨어를 기반으로 하며 다음과 같이 promise나 async, await를 통해 에러핸들링하는게 좋습니다. 
```js
const log = console.log
const app = require('express')()

const wrapE = f => {
  return (req, res, next) =>{
    f(req, res, next).catch(next)
  }
} 
app.get("/a", wrapE(async(req, res, next) => {
  throw Error("a")
  res.status(200).json({"message" : "a"})
}))
app.get("/b", wrapE(async(req, res, next) => {
  throw Error("a")
  res.status(200).json({"message" : "b"})
})) 
app.use((error, req, res, next) => {
  log(error)
  res.status(400).json({"message" : "에러가 발생했습니다."})
}) 
app.listen(3000, () => log("서버시작, http://127.0.0.1:3000/a")) 
```


## 그외
### pm2
서버 모니터링과 안정적인 백그라운드 실행을 위해 pm2로 실행하는 것이 좋습니다. 
![pm2](https://pm2.io/images/home.gif) 

### DB 연결을 최소화해라
```
A -> DB
B -> DB
```

보다는

```
A -> C
B -> C
C -> DB
```
가 낫다.


### type 체킹
왠만하면 타입스크립트로 정적타입시스템을 도입하는게 좋다.

### date는 UTC 기준으로. 
date 같은 경우 배포하는 서버의 타임마다 다르다. UTC를 기준으로 로직을 설게하자. 