from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
from pydantic import BaseModel
from langchain_community.llms import OpenAI as LangOpen
from openai import OpenAI as Open
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
import os
import requests
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
from sqlalchemy import Column, String, Float, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text  
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler


DATABASE_URL = "mysql+pymysql://gunwoo2:woorifisa3!W@118.67.131.22:3306/gunwoo"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# Load environment variables
load_dotenv()
# 외화 보유 데이터를 저장하는 임시 딕셔너리 (키: user_id, 값: 외화 보유 리스트)
foreign_currency_data = {}


app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

client = Open(api_key=api_key)
# OpenAI model initialization

embedding = OpenAIEmbeddings(api_key=api_key)
# Load CSV data
csv_path = os.getenv("CSV_PATH", "woori_bank_savings.csv")
woori_bank_savings = pd.read_csv(csv_path).to_dict(orient='records')

# Store user asset information and responses
user_asset_info = {}
user_responses = {}
current_dir = os.path.dirname(os.path.abspath(__file__))
savings_vectorstore = Chroma("savings_vectorstore", embedding_function=embedding)
cards_vectorstore = Chroma("cards_vectorstore", embedding_function=embedding)
funds_vectorstore = Chroma("funds_vectorstore", embedding_function=embedding)
portfolio_vectorstore = Chroma("portfolio_vectorstore", embedding_function=embedding)

def load_and_embed_csv(file_path, vectorstore):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    vectorstore.add_documents(docs)

savings_csv_path = os.path.join(current_dir, "woori_bank_savings.csv")
cards_csv_path = os.path.join(current_dir, "cards.csv")
funds_csv_path = os.path.join(current_dir, "funds.csv")
portfolio_csv_path = os.path.join(current_dir, "finance.csv")

load_and_embed_csv(savings_csv_path, savings_vectorstore)
load_and_embed_csv(cards_csv_path, cards_vectorstore)
load_and_embed_csv(funds_csv_path, funds_vectorstore)
load_and_embed_csv(portfolio_csv_path, portfolio_vectorstore)

# 환율 API URL
API_URL = f"https://api.exchangerate-api.com/v4/latest/USD?apikey={os.getenv('EXCHANGE_API_KEY')}"


# 환율 관련 저장된 사용자 설정
user_rates = []

# Predefined questions
questions = [
    "예금 목적은 무엇인가요?",
    "적금 목표 금액은 얼마인가요?",
    "펀드에 대해 어떤 목표를 가지고 계신가요?"
]

class UserRequest(BaseModel):
    question: str

class AssetInfo(BaseModel):
    deposit: str
    savings: str
    fund: str
    debt: str  # 부채 추가
    age: int
    monthly_income: str

class ResponseInfo(BaseModel):
    user_id: str
    response: str

class UserId(BaseModel):
    user_Id: str

class UserRateRequest(BaseModel):
    user_id: str  # 사용자 ID 추가
    currency: str  # 목표 환율 설정(엔, 달러, 유로 등)
    target_rate: float  # 목표 환율
    action: str  # "buy" 또는 "sell"
    amount: float  # 목표 금액


class UserForeignCurrency(BaseModel):
    user_id: str
    currency: str
    balance: float = 0.0
    total_spent: float = 0.0
    total_converted_krw: float = 0.0

class UserForeignCurrencyDB(Base):
    __tablename__ = "user_foreign_currency"

    id = Column(BigInteger, primary_key=True, autoincrement=True)  # 고유 ID
    user_id = Column(String(50), nullable=False)                  # 사용자 ID
    currency = Column(String(10), nullable=False)                 # 통화 (USD, EUR 등)
    balance = Column(Float, nullable=False, default=0.0)          # 보유 외화 잔액
    total_spent = Column(Float, nullable=False, default=0.0)      # 총 지출 금액 (KRW)
    total_converted_krw = Column(Float, nullable=False, default=0.0)  # 총 환전 금액 (KRW)

    def __repr__(self):
        return (
            f"<UserForeignCurrencyDB(user_id={self.user_id}, currency={self.currency}, "
            f"balance={self.balance}, total_spent={self.total_spent}, total_converted_krw={self.total_converted_krw})>"
        )


def initialize_user_deposit(user_id: str, db: Session):
    """
    사용자의 예금 잔액(deposit)을 DB에서 불러와 초기화합니다.
    """
    global user_asset_info

    # DB에서 사용자 예금 정보 가져오기
    result = db.execute(
        text("SELECT deposit_holdings FROM user_profile WHERE user_id = :user_id"),
        {"user_id": user_id}
    ).fetchone()

    if result:
        deposit = result[0]  # 실 단위로 저장된 예금 잔액 그대로 사용
        user_asset_info = {"deposit": deposit}
    else:
        # 사용자 정보가 없을 경우 기본값 설정
        user_asset_info = {"deposit": 0}

    print(f"[INFO] User {user_id} deposit initialized: {user_asset_info['deposit']}")


def get_current_exchange_rate(currency: str) -> float:
    response = requests.get(API_URL)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="환율 데이터를 가져오는 데 실패했습니다.")
    
    rates = response.json().get("rates", {})
    if currency not in rates:
        raise HTTPException(status_code=400, detail=f"{currency}에 대한 환율 정보를 찾을 수 없습니다.")
    
    return rates[currency]





#외화기능-------------------------------------------------------------------
@app.get("/foreign-currency/{user_id}/holdings", response_model=list[UserForeignCurrency])
def get_user_foreign_holdings(user_id: str, db: Session = Depends(get_db)):
    """
    특정 사용자의 외화 보유 정보를 조회합니다.
    """
    initialize_user_deposit(user_id, db)  # 데이터 초기화
    holdings = db.query(UserForeignCurrencyDB).filter(UserForeignCurrencyDB.user_id == user_id).all()

    if not holdings:
        raise HTTPException(status_code=404, detail="사용자의 외화 보유 데이터를 찾을 수 없습니다.")

    return [
        UserForeignCurrency(
            user_id=holding.user_id,
            currency=holding.currency,
            balance=holding.balance,
            total_spent=holding.total_spent,
            total_converted_krw=holding.total_converted_krw
        )
        for holding in holdings
    ]


@app.get("/foreign-currency/{user_id}/holdings", response_model=list[UserForeignCurrency])
def get_user_foreign_holdings(user_id: str, db: Session = Depends(get_db)):
    """
    특정 사용자의 외화 보유 정보를 조회합니다.
    """
    initialize_user_deposit(user_id, db)  # 데이터 초기화
    holdings = db.query(UserForeignCurrencyDB).filter(UserForeignCurrencyDB.user_id == user_id).all()

    if not holdings:
        raise HTTPException(status_code=404, detail="사용자의 외화 보유 데이터를 찾을 수 없습니다.")

    return [
        UserForeignCurrency(
            user_id=holding.user_id,
            currency=holding.currency,
            balance=holding.balance,
            total_spent=holding.total_spent,
            total_converted_krw=holding.total_converted_krw
        )
        for holding in holdings
    ]

@app.post("/foreign-currency/set-target")
def process_target_rate(user_rate: UserRateRequest, db: Session = Depends(get_db)):
    """
    외화 목표 거래를 처리합니다 (구매/판매).
    거래 체결 시 알림을 생성합니다.
    """
    # 사용자의 예금 잔액 초기화
    initialize_user_deposit(user_rate.user_id, db)

    # 예금 잔액 확인
    deposit = user_asset_info["deposit"]

    # 실시간 환율 가져오기
    current_rate = get_current_exchange_rate(user_rate.currency)

    # 거래 계산 (구매 시 실시간 환율로 계산)
    if user_rate.action == "buy":
        transaction_amount_krw = user_rate.amount * user_rate.target_rate
    elif user_rate.action == "sell":
        transaction_amount_krw = user_rate.amount * user_rate.target_rate
    else:
        raise HTTPException(status_code=400, detail="유효하지 않은 거래 유형입니다.")

    # 거래 유형 처리
    if user_rate.action == "buy":
        if deposit < transaction_amount_krw:
            raise HTTPException(status_code=400, detail="예금 잔액이 부족합니다.")

        # 예금 차감 및 DB 반영
        user_asset_info["deposit"] -= transaction_amount_krw
        db.execute(
            f"UPDATE user_profile SET deposit_holdings = {user_asset_info['deposit']} WHERE user_id = '{user_rate.user_id}'"
        )

        # 외화 추가
        holding = db.query(UserForeignCurrencyDB).filter(
            UserForeignCurrencyDB.user_id == user_rate.user_id,
            UserForeignCurrencyDB.currency == user_rate.currency
        ).first()

        if not holding:
            # 보유 외화 데이터가 없으면 새로 생성
            holding = UserForeignCurrencyDB(
                user_id=user_rate.user_id,
                currency=user_rate.currency,
                balance=0.0,
                total_spent=0.0,
                total_converted_krw=0.0,
            )
            db.add(holding)

        holding.balance += user_rate.amount
        holding.total_spent += transaction_amount_krw  # 총 지출 업데이트
        db.commit()

    elif user_rate.action == "sell":
        # 외화 잔액 차감 및 DB 반영
        holding = db.query(UserForeignCurrencyDB).filter(
            UserForeignCurrencyDB.user_id == user_rate.user_id,
            UserForeignCurrencyDB.currency == user_rate.currency
        ).first()

        if not holding or holding.balance < user_rate.amount:
            raise HTTPException(status_code=400, detail="외화 보유 잔액이 부족합니다.")

        holding.balance -= user_rate.amount

        # 예금 증가 (판매 금액을 기존 예금 잔액에 더하기)
        user_asset_info["deposit"] += transaction_amount_krw
        db.execute(
            f"UPDATE user_profile SET deposit_holdings = {user_asset_info['deposit']} WHERE user_id = '{user_rate.user_id}'"
        )
        db.commit()

    return {"message": f"{user_rate.currency} 거래가 성공적으로 처리되었습니다."}
#외화기능-------------------------------------------------------------------







@app.post("/set_asset_info")
def set_asset_info(asset_info: AssetInfo):
    global user_asset_info
    
    def clean_and_convert(value: str):
        """쉼표를 제거하고, 숫자로 변환하는 함수"""
        try:
            return int(value.replace(",", "")) if value != "정보 없음" else "정보 없음"
        except ValueError:
            return "정보 없음"  # 변환이 불가능한 값에 대해서는 "정보 없음" 반환


    deposit = clean_and_convert(asset_info.deposit)
    savings = clean_and_convert(asset_info.savings)
    fund = clean_and_convert(asset_info.fund)
    debt = clean_and_convert(asset_info.debt)
    monthly_income = clean_and_convert(asset_info.monthly_income)

    # 총 자산 계산 (예금 + 적금 + 펀드 + 부채)
    total_assets = deposit + savings + fund + debt

    user_asset_info = {
        "deposit": str(deposit) if deposit != "정보 없음" else "정보 없음",
        "savings": str(savings) if savings != "정보 없음" else "정보 없음",
        "fund": str(fund) if fund != "정보 없음" else "정보 없음",
        "debt": str(debt) if debt != "정보 없음" else "정보 없음",
        "totalAssets": str(total_assets) if total_assets != "정보 없음" else "정보 없음",  # 총 자산 추가
        "monthly_income": str(monthly_income) if monthly_income != "정보 없음" else "정보 없음",
        "age": asset_info.age
    }

    print(f"Received age: {asset_info.age}")  # FastAPI에서 수신된 age 값 로그
    print(f"Processed asset info: {user_asset_info}")  # 처리된 자산 정보 로그

    # 자산 정보를 챗봇에 학습시키기
    sendAssetInfoToChatbot(user_asset_info)
    
    return {"status": "success", "message": "User asset information updated successfully"}

def sendAssetInfoToChatbot(asset_info: dict):
    """자산 정보를 챗봇에 전달하는 메서드"""
    print(f"Sending asset info to chatbot: {asset_info}")  # 실제로는 Langchain 모델에 전달



# API 키 설정
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = Open(api_key=api_key)




@app.post("/recommend_savings")
def recommend_products(request: UserRequest):
    """
    사용자의 질문에 대해 적절한 금융 상품을 추천합니다.
    """
    try:
        question = request.question.strip()

        # 사용자 자산 정보 포함
        if user_asset_info:
            asset_info_str = (
                f"사용자의 자산 내역은 다음과 같습니다 - "
                f"예금: {user_asset_info['deposit']}원, "
                f"적금: {user_asset_info['savings']}원, "
                f"펀드: {user_asset_info['fund']}원, "
                f"부채: {user_asset_info['debt']}원, "
                f"총 자산: {user_asset_info['totalAssets']}원, "
                f"월 수입: {user_asset_info['monthly_income']}원, "
                f"나이: {user_asset_info['age']}세."
            )
        else:
            asset_info_str = ""


        # 금융 관련 질문 확인
        keywords = ["은행", "적금", "금리", "상품", "돈", "이자", "자산", "예금", "펀드", "투자", "저축", "포트폴리오", "주식", "달러", "환율", "저금", "수입", "소비"]
        is_financial_question = any(keyword in question.lower() for keyword in keywords)

        if is_financial_question:
            # 각 파일에서 질문과 관련된 정보 검색
            savings_results = savings_vectorstore.similarity_search(question, k=3)
            cards_results = cards_vectorstore.similarity_search(question, k=3)
            funds_results = funds_vectorstore.similarity_search(question, k=3)
            portfolio_results = portfolio_vectorstore.similarity_search(question, k=3)


            # 검색 결과 정리
            savings_info = "\n".join([f"적금 추천: {doc.page_content}" for doc in savings_results])
            cards_info = "\n".join([f"카드 추천: {doc.page_content}" for doc in cards_results])
            funds_info = "\n".join([f"펀드 추천: {doc.page_content}" for doc in funds_results])
            portfolio_info = "\n".join([f"자산 분배 원칙: {doc.page_content}" for doc in portfolio_results])


            # 금융 상품 정보 요약
            product_info_str = (
                f"적금 정보:\n{savings_info}\n\n카드 정보:\n{cards_info}\n\n펀드 정보:\n{funds_info}\n\n자산 분배 참고자료:\n{portfolio_info}"
            )

            # 1차 ChatCompletion 호출
            response1 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "너는 간결하고 빠른 금융 상담 답변을 제공하는 에이전트야. "
                        "모든 답변은 200자 이내로 작성하며, 사용자의 자산 정보를 바탕으로 재무 상태를 분석하고 "
                        "필요 시에 적절한 상품을 추천해줘. 필요하면 하나의 핵심 상품만 언급하고, 불필요한 정보는 포함하지 마."
                    )},
                    {"role": "user", "content": (
                        f"사용자의 자산 내역, 월수입, 나이는 다음과 같아: {asset_info_str}. "
                        f"금융 상품 정보와 자산 분배 참고자료는 다음과 같아: {product_info_str}. "
                        f"질문: {question}?"
                    )}
                ]
            )

            # 1차 응답 추출
            first_response = response1.choices[0].message.content

            # 2차 ChatCompletion 호출로 응답 보강
            response2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "너는 2차 검토를 통해 금융 상담 답변을 더욱 최적화하는 에이전트야. "
                        "아래는 사용자가 처음 받은 응답이야. 이를 바탕으로 추가적인 인사이트를 제공하거나, "
                        "불필요한 내용을 제거하여 더욱 정확하고 간결한 최종 답변을 작성해줘."
                        "만약 너가 필요한 정보가 있으면 사용자에게 질문해줘"
                    )},
                    {"role": "user", "content": (
                        "`*` 기호나 불필요한 특수 문자가 있으면 삭제해줘."
                        f"사용자의 질문: {question}\n"
                        f"사용자의 자산 정보: {asset_info_str}\n"
                        f"처음 받은 응답: {first_response}\n"
                        "이 응답을 보완해서 최종적으로 더 나은 답변을 작성해줘."
                    )}
                ]
            )

            # 2차 응답 추출
            final_response = response2.choices[0].message.content

        else:
            final_response = "죄송합니다. 은행 및 금융 관련 질문에만 답변을 제공할 수 있습니다."

        return {"response": final_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


    
@app.post("/analyze-user")
def analyze_user(user: UserId):
    # Database connection
    engine = create_engine('mysql+pymysql://gunwoo2:woorifisa3!W@118.67.131.22:3306/gunwoo')
    query = "SELECT * FROM user_profile"
    data = pd.read_sql(query, engine)

    if data.empty:
        raise HTTPException(status_code=404, detail="No profile data found.")
    
    # Preprocess data
    data = data.drop(['password', 'cluster'], axis=1, errors='ignore')
    data = data.replace('', np.nan).fillna(0)

    categorical_features = ['gender', 'employment_status']
    numerical_features = [col for col in data.columns if col not in categorical_features + ['user_id']]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    data_processed = pipeline.fit_transform(data)

    n_clusters = min(4, len(data_processed))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data_processed)
    data['cluster'] = clusters

    user_data = data[data['user_id'] == user.user_Id]
    user_processed = pipeline.transform(user_data.drop(['user_id'], axis=1))
    user_cluster = kmeans.predict(user_processed)[0]

    # 클러스터별 평균 순자산 및 부채 계산
    cluster_data = data[data['cluster'] == user_cluster]
    avg_net_assets = cluster_data['total_financial_assets'].mean() - cluster_data['debt_amount'].mean()
    avg_debt = cluster_data['debt_amount'].mean()
    user_net_assets = user_data['total_financial_assets'].values[0] - user_data['debt_amount'].values[0]
    user_debt = user_data['debt_amount'].values[0]

    return {
        "userType": f"Cluster {user_cluster}",
        "userNetAssets": user_net_assets,
        "userDebt": user_debt,
        "avgNetAssets": avg_net_assets,
        "avgDebt": avg_debt
    }

from decimal import Decimal

@app.get("/get_exchange_rate")
def get_exchange_rate():
    try:
        # API 호출
        response = requests.get(API_URL)
        data = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="환율 정보를 가져오는 데 실패했습니다.")

        rates = data["rates"]

        # 기준 환율 (1 USD -> KRW)
        usd_to_krw = rates.get("KRW")
        if not usd_to_krw:
            raise HTTPException(status_code=500, detail="KRW 환율 정보를 가져올 수 없습니다.")

        # USD를 기준으로 다른 통화 계산
        converted_rates = {
            "USD": round(usd_to_krw, 2),  # 1 USD = KRW
            "EUR": round(usd_to_krw / rates["EUR"], 2),  # 1 EUR = KRW
            "JPY": round(usd_to_krw / rates["JPY"]*100, 2)   # 1 JPY = KRW
        }

        return {"rates": converted_rates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")



@app.post("/set_target_rate")
def set_target_rate(user_rate: UserRateRequest, background_tasks: BackgroundTasks):
    try:
        if user_rate.target_rate <= 0 or user_rate.amount <= 0:
            raise HTTPException(status_code=400, detail="목표 환율과 금액은 0보다 커야 합니다.")
        
        if user_rate.currency not in ["USD", "EUR", "JPY"]:
            raise HTTPException(status_code=400, detail="지원하지 않는 통화입니다.")

        user_rates.append(user_rate)
        background_tasks.add_task(check_target_rate, user_rate)
        return {"message": "목표 환율이 성공적으로 설정되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 중 오류 발생: {str(e)}")


def check_target_rate(user_rate: UserRateRequest):
    try:
        response = requests.get(API_URL)
        data = response.json()

        if response.status_code != 200:
            raise Exception("환율 정보를 가져오는 데 실패했습니다.")

        rates = data["rates"]
        krw_rate = rates.get("KRW")
        if not krw_rate or krw_rate == 0:
            raise Exception("유효한 KRW 환율 정보를 가져올 수 없습니다.")

        current_rate = rates.get(user_rate.currency)
        if current_rate:
            # 현재 환율을 KRW 기준으로 변환
            converted_rate = current_rate * krw_rate

            # 구매일 경우: 목표 환율보다 떨어지면 알림
            if user_rate.action == "buy" and converted_rate <= user_rate.target_rate:
                notify_user(user_rate, converted_rate)

            # 판매일 경우: 목표 환율보다 오르면 알림
            elif user_rate.action == "sell" and converted_rate >= user_rate.target_rate:
                notify_user(user_rate, converted_rate)
        else:
            raise Exception("목표 환율에 해당하는 정보를 찾을 수 없습니다.")
    except Exception as e:
        print(f"Error while checking target rate: {str(e)}")



def notify_user(user_rate: UserRateRequest, current_rate: float):
    """
    거래 완료 시 알림 생성 및 전송.
    """
    try:
        notification = {
            "userId": user_rate.user_id,
            "message": f"거래 완료: {user_rate.amount} {user_rate.currency} 거래가 체결되었습니다! "
                       
        }
        response = requests.post("http://localhost:8001/notifications/create", json=notification)
        response.raise_for_status()  # 요청 실패 시 예외 발생
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 알림 전송 실패: {e}")



# APScheduler 설정
scheduler = BackgroundScheduler()

def monitor_exchange_rates():
    try:
        for user_rate in user_rates:
            check_target_rate(user_rate)
    except Exception as e:
        print(f"Monitor Error: {e}")

scheduler.add_job(monitor_exchange_rates, "interval", hours=24)  # 24시간마다 실행
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
