# Agent Simulation을 이용한 추리 게임
사용 library/framework:   
langchain, streamlit, etc

### 구현된 기능:
-Open AI API 재시 후 사이트 열림 (입력 가려짐)   

-simulation 참가시킬 캐릭터 숫자 고름   
-키워드로 캐릭터 이름 -검색 + 뽑기  
-캐릭터 추가  
-캐릭터 전체 삭제  
-살인사건 피해자 지정  

-initialize   
-> 캐릭터 패르소나 (성격) 설정 지정됨 -eg 칸예웨스트- 래퍼 스타일, donald trump 오만,부자,etc   
-> 캐릭터간의 관계성 설정 지정됨   
-> 범인 캐릭터 지정  

-image initialize   
->캐릭터 외형 관련 검색/지정/statement   
->성별 (검색)  

-이미지 step 숫자 설정 고름  
-캐릭터 이미지 생성  

-start   
->각 agent에 캐릭터 설정 인격 부여  
->증거 생성 (탐정 agent만 이정보를 가지고있음)  

-번역 (한/영)  

-캐릭터 한명의 대화 + 캐릭터 행동 생성   
-사용자 대화 참여 - 영어 한국어 둘다 가능  
-캐릭터 행동 image 생성  
-캐릭터 말하기 -> 대화의 감정상태 분석 톤 지정 -> 지정된톤, 캐릭터 성별에 맞는 목소리로 캐릭터가 대화 첫문장을 립싱크로 읽음 (비디오 생성)  
  
-캐릭터들 끼리 투표로 범인 지적 (mafia게임같이)  
-사용자가 범인 고르기 -> 결과   


[presentation slides](https://github.com/sohneunsoo/nctfinal-ai-agents-simulations/blob/main/%EB%A7%88%EC%A7%80%EB%A7%89%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_3%EC%A1%B0_.pptx.pdf)
