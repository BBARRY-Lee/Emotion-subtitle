# 5가지의 감정자막 자동 생성 'EmoS' 서비스

## 1. Introduction
fine-tuning한 KLUE\RoBERTa 모델과 OpenCV를 활용한 5가지의 감정자막 자동 생성 'EmoS' 서비스는 <br>
발화문을 AI 모델을 통해 분류한 감정에 따라 화자의 감정을 반영한 각각 다른 스타일의 자막을 생성하는 서비스입니다. <br>
본 서비스를 통해 음성을 들을 수 없는 상황이나 청각 장애를 가진 사람들도 화자의 감정을 생생하게 전달받을 수 있습니다. <br>
- ['EmoS' 서비스 시연 동영상](https://drive.google.com/file/d/1roGFsI-8gxlmMXDeHjcZsTYdy2Em3rTX/view?usp=share_link)
<br>
2023년 2월 17일, 본 프로젝트를 통해 프로그래머스 인공지능 데브코스 4기의 최우수 프로젝트 팀으로 선정되어 최우수상을 수상하였습니다 🙂 <br>
<br>
<img src=https://user-images.githubusercontent.com/99329555/219948186-04f92992-aa5f-4e4e-9cdb-b9a695181ef9.png width="100%" height="100%"/>



<br>

## 2. Development process
1. **데이터 수집 및 전처리** : AI-Hub의 감정 분류를 위한 음성데이터셋과 감성대화 말뭉치 사용
2. **Sentiment Analysis** : fine-tuning KLUE\RoBERTa 모델을 이용해 발화문을 5가지 감정(분노, 슬픔, 불안, 기쁨, 중립) 분류
3. **Caption Generation** : openCV와 PIL를 활용해 감정 Label 별 자막을 생성하고 Time stamp에 따라 영상과 자막 합성
4. **Frontend & Backend** : AWS, Django, CSS, HTML을 활용한 FE & BE 구축

<br>
<img src=https://user-images.githubusercontent.com/99329555/219366404-9b55feef-230e-4216-bfb1-9c16e973e859.png width="100%" height="100%"/>
<br>
  
## 3. Prototype
<img src=https://user-images.githubusercontent.com/99329555/219374999-d1817119-cc52-4247-8eca-7d5c22a0caa2.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219375043-6b872649-5aa2-4308-a8fe-4d27cfbfa03e.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219375094-f8b73e23-8496-45e6-9c96-48a3704c3556.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219375178-46bf9e37-9f91-494e-a96e-86c954379b5d.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219375231-05af846e-2d4c-44a2-bd63-30c86f0f0d58.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219373432-dbc8195a-f308-4513-8606-edd0fc00a8cb.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219375280-d2d3db7b-c144-433c-87ab-f39e1d8b9ed4.png width="100%" height="100%"/>
<img src=https://user-images.githubusercontent.com/99329555/219373100-094139c7-90c4-48bf-8d16-32028603ed64.png width="100%" height="100%"/>
<br>

## 4. Makers
프로그래머스 인공지능 데브코스 4기 B5팀의 팀원들과 함께 기획하고 개발하였습니다.
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable --> 

<table>
  <tr>
    <td align="center"><a href="https://github.com/seroak"><img src="https://user-images.githubusercontent.com/99329555/219380160-e69e840d-6064-44c2-b1a3-ebb59fcc51da.png" width="150" height="150"><br /><sub><b>이규열</b></sub></td>
    <td align="center"><a href="https://github.com/Bandi120424"><img src="https://user-images.githubusercontent.com/99329555/219380477-a9b9cc6c-836d-4c70-8707-b2128ee2bd5c.png" width="150" height="150"><br /><sub><b>한나영</b></sub></td>
    <td align="center"><a href="https://github.com/BBARRY-Lee"><img src="https://user-images.githubusercontent.com/99329555/219381993-1219c0ec-b038-4c7b-9182-28696e40eca4.png" width="150" height="150"><br /><sub><b>이지윤</b></sub></td>
    <td align="center"><a href="https://github.com/Jieun-Enna"><img src="https://user-images.githubusercontent.com/99329555/219381428-0ab5b2c4-e4bf-4d0b-acc2-0f3e19905fce.png" width="150" height="150"><br /><sub><b>박지은</b></sub></td>
  </tr>
</table>

<br>

## 5. Git Commit Message Convention
### Commit Message Format

#### [참고](https://underflow101.tistory.com/31)

- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: Document 수정
- `style`: 코드 formatting, 세미콜론(;) 누락, 코드 변경이 없는 경우 등
- `refactor`: 이미 있는 코드에 대한 리팩토링
- `test`: Test Code에 대한 commit
- `build`: 빌드 관련 파일 수정 (예시 scope: gulp, broccoli, npm)
- `perf`: 성능 개선사항
- `ci`: CI 설정 파일 수정 (예시 scope: Circle, BrowserStack, SauceLabs)
- `chore`: 그 외의 작은 수정 (빌드 업무 수정, 패키지 매니저 수정 등)

</div>
</details>
<br />

### Issue
- 해야 할 Task를 미리 Issue에 등록 후 개발

### Pull Request
- Issue에 올라온 Task를 끝내면, Pull Request를 통해 팀원들의 Review를 받은 후, develop 브랜치에 merge

### Branch Strategy

#### `main`
- 제품을 최종적으로 배포하는 브랜치 (develop 브랜치로부터 merge만 받는 브랜치)
- 배포에 사용

#### `develop`
- 아직 배포되지 않은 공용 브랜치
- feature 브랜치로부터 merge를 받아 개발 중 버그를 발견하면 이 브랜치에 직접 commit

#### `feature`
- 새로운 기능 개발을 하는 브랜치
  - 반드시 develop 로부터 시작되고, develop 브랜치에 머지함
  - **feature/기능이름**
    ex) `feature/new-feature`
    
#### `realease`
- 최종 배포전, QA를 진행하는 브랜치
- 아직 배포되지 않은 공용 브랜치
- develop 브랜치로부터 merge를 받음

#### `hotfix`
- 다음 배포 전까지 급하게 고쳐야되는 버그를 처리하는 브랜치
  - 배포 버전 심각한 버그 수정이 필요한경우, 버그 수정을 진행한뒤 main, develop 브랜치에 merge함
  - **hotfix/버그이름**
    ex) `hotfix/bugs`
