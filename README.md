# 프로그래머스 인공지능 데브코스 B5팀 Final Project

## 1. Introduction
Muti-modal 감정 분류를 통한 감정이 반영된 자막 합성 서비스
<br>
History : [Notion page](https://daisylee.notion.site/b5-4-2dfe4e8c3a7b48fabe3bf29f3de60076)

<br>

## 2. Makers
- 이지윤
- 박지은
- 한나영
- 이규열

<br>

## 3. Service
Muti-modal 감정 분류를 통한 감정이 반영된 자막 합성 서비스

<br>

## 4. Development process
1. Text data와 Audio data를 활용해 Muti-modal 감정 분류 모델 구현
2. Video에서 Audio data를 추출해 STT를 이용하여 발화에 따른 Timestamp와 Text data 추출
3. 추출한 Text data를 Muti-modal 감정 분류 모델에 Input으로 하여, 감정이 Labeling된 Output 출력
4. openCV와 PIL 라이브러리를 이용한 Text data → 감정 Label에 따른 자막 이미지 변환
5. Video에 Timestamp에 맞는 감정 Label에 따른 자막 이미지 합성

<br>

## 5. Tech Stacks
![Python](https://img.shields.io/badge/-Python-14354C?style=flat-square&logo=Python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat-square&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![openCV](https://img.shields.io/badge/opencv-5C3EE8?style=flat-square&logo=opencv&logoColor=black)
<br/>
<br>
  
## 6. Git Commit Message Convention
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
