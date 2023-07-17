# SATSegmentation

Satellite Image Building Area Segmentation

# Colab SSH

[참고](https://www.youtube.com/watch?v=oAKxxLy-G5g)

```
NGROK_TOKEN = '(1)'
PASSWORD = 'mypassword'
GITHUB_USER_NAME = '(2)'
GITHUB_EMAIL = '(3)'
GITHUB_ACCESS_TOKEN = '(4)'
DATA_DIR = '(5)'
```

1. ngrok 가입 이후 your authtoken 탭에서 auth token 복사, 이후 (1) 에 붙여넣기
2. (2), (3)에는 자기 github username, email
3. github token 발급
    - Contents 섹션 Read and Write
    - Administration 섹션 Read Only
    - 토큰 (4) 에 붙여넣기
4. (5)에 데이터셋 구글 드라이브 경로
    - ex) `/gdrive/Dacon/Datasets/`
    - [ ] EnvVar로 데이터셋 읽어오도록 코드 수정해야함
5. 코랩 실행하면 아래와 같이 나올텐데

```
  Host google_colab_ssh
  HostName 0.tcp.ngrok.io
  User root
  Port 12744
```

5. 이거 이용해서 VSCode SSH 로 접속 가능
   ex)

```
 ssh root@0.tcp.ngrok.io -p 12744
```
