# PDF 벡터 RAG 시스템

PDF 문서를 처리하고 쿼리하기 위한 Llama 3 모델 기반의 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템입니다. 이 시스템은 벡터 기반 검색을 사용하여 PDF 문서를 효율적으로 처리하고, 사용자 친화적인 GUI를 제공합니다.

## 주요 기능

- PDF 문서 처리 및 청킹
- Llama 3 모델을 이용한 벡터 임베딩
- 코사인 유사도 기반 의미론적 검색
- 문서 처리 및 쿼리를 위한 대화형 GUI
- 응답에 대한 출처 표시

## 사전 요구사항

- Python 3.8 이상
- PyTorch
- Transformers 라이브러리
- Hugging Face 계정 및 API 토큰
- 필요한 Python 패키지 (pip으로 설치):
  ```
  torch
  transformers
  numpy
  huggingface_hub
  pdfplumber
  tqdm
  scikit-learn
  tkinter
  ```

## 설치 방법

1. 저장소 복제
2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. Hugging Face 토큰을 코드에 설정하거나 환경 변수로 설정

## 사용 방법

### GUI 인터페이스

그래픽 인터페이스 실행:

```bash
python vectorrag_gui.py
```

GUI에서 제공하는 기능:
- PDF 파일 선택
- 여러 PDF 동시 처리
- 쿼리 입력 및 제출
- 출처가 포함된 응답 확인

### 명령줄 인터페이스

Python 코드에서 VectorRAG 클래스 직접 사용:

```python
from vectorrag import PDFLlama3VectorRAG

# 시스템 초기화
rag = PDFLlama3VectorRAG()

# PDF 문서 처리
documents = rag.read_pdf("문서/경로/파일명.pdf")
rag.index_documents(documents)

# 시스템 쿼리
results = rag.retrieve("검색어를 입력하세요")
response = rag.generate_response("검색어를 입력하세요", results)
```

## 시스템 구성 요소

### PDFLlama3VectorRAG (vectorrag.py)
- RAG 주요 구현체
- PDF 문서 처리
- 벡터 임베딩 생성 및 관리
- 코사인 유사도 기반 문서 검색
- Llama 3 기반 응답 생성

### PDFVectorRAGApp (vectorrag_gui.py)
- 그래픽 사용자 인터페이스
- 멀티 파일 선택 및 처리
- 대화형 쿼리 인터페이스
- 실시간 진행 상황 표시
- 응답 및 출처 표시

## 기술적 세부사항

### 문서 처리
- PDF를 구성 가능한 크기의 청크로 처리 (기본값 1000자)
- 각 청크는 페이지 번호 참조 유지
- Llama 3 모델을 사용한 텍스트 임베딩

### 검색 시스템
- 문서 검색에 코사인 유사도 사용
- 검색 문서 수 설정 가능 (기본값 k=5)
- 검색된 문서와 함께 관련성 점수 반환

### 응답 생성
- Llama 3를 사용한 문맥 기반 응답 생성
- 출처 표시 포함 (문서 이름 및 페이지 번호)
- 온도 제어된 생성 (기본값 0.7)

## 제한사항

- 대용량 문서의 경우 상당한 메모리 필요
- 처리 속도는 사용 가능한 GPU 리소스에 따라 달라짐
- Llama 3 모델에 의한 최대 문맥 길이 제한

## 기여하기

문제점을 제출하거나, 저장소를 포크하고, 개선사항에 대한 풀 리퀘스트를 생성하는 것을 환영합니다.

## 라이선스

MIT License

Copyright (c) 2024 PDF Vector RAG System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.