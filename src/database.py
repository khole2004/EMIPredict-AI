import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "applicants.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ApplicantRecord(Base):
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_name = Column(String, default="Applicant")
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    marital_status = Column(String, nullable=False)
    education = Column(String, nullable=False)
    monthly_salary = Column(Float, nullable=False)
    employment_type = Column(String, nullable=False)
    years_of_employment = Column(Float, nullable=False)
    company_type = Column(String, nullable=False)
    house_type = Column(String, nullable=False)
    monthly_rent = Column(Float, nullable=False)
    family_size = Column(Integer, nullable=False)
    dependents = Column(Integer, nullable=False)
    school_fees = Column(Float, default=0.0)
    college_fees = Column(Float, default=0.0)
    travel_expenses = Column(Float, default=0.0)
    groceries_utilities = Column(Float, default=0.0)
    other_monthly_expenses = Column(Float, default=0.0)
    existing_loans = Column(String, default="No")
    current_emi_amount = Column(Float, default=0.0)
    credit_score = Column(Float, nullable=False)
    bank_balance = Column(Float, default=0.0)
    emergency_fund = Column(Float, default=0.0)
    emi_scenario = Column(String, nullable=False)
    requested_amount = Column(Float, nullable=False)
    requested_tenure = Column(Integer, nullable=False)
    
    # Model predictions & decisions
    predicted_eligibility = Column(String, nullable=True)
    predicted_max_emi = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
