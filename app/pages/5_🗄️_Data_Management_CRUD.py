import streamlit as st
import pandas as pd
import io
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
from database import SessionLocal, ApplicantRecord, init_db

st.set_page_config(page_title="Data Management - EMIPredict AI", page_icon="🗄️", layout="wide")

st.title("🗄️ Applicant Financial Data Management (CRUD)")
st.markdown("Administrative underwriting portal for loan officers to Create, Read, Update, and Delete applicant records.")

init_db()
db = SessionLocal()

tab_read, tab_create, tab_update, tab_delete = st.tabs([
    "📋 View & Search Records (Read)",
    "➕ Add Applicant (Create)",
    "✏️ Edit Record (Update)",
    "🗑️ Delete Record (Delete)"
])

# 1. READ & SEARCH
with tab_read:
    st.subheader("Stored Loan Applications")
    records = db.query(ApplicantRecord).all()
    if not records:
        st.info("No applicant records found in database yet. Add one in the 'Add Applicant' tab or run a prediction!")
    else:
        df_rec = pd.DataFrame([{
            "ID": r.id,
            "Name": r.customer_name,
            "Age": r.age,
            "Gender": r.gender,
            "Salary": r.monthly_salary,
            "Credit Score": r.credit_score,
            "Scenario": r.emi_scenario,
            "Requested Amt": r.requested_amount,
            "Tenure": r.requested_tenure,
            "Eligibility": r.predicted_eligibility,
            "Max Safe EMI": r.predicted_max_emi
        } for r in records])

        search_query = st.text_input("🔍 Search by Applicant Name or Scenario:")
        if search_query:
            df_rec = df_rec[
                df_rec['Name'].str.contains(search_query, case=False, na=False) |
                df_rec['Scenario'].str.contains(search_query, case=False, na=False)
            ]

        st.dataframe(df_rec, use_container_width=True)

        # CSV Export
        csv_buf = io.StringIO()
        df_rec.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Export Applicants to CSV",
            data=csv_buf.getvalue(),
            file_name="stored_loan_applicants.csv",
            mime="text/csv"
        )

# 2. CREATE
with tab_create:
    st.subheader("New Loan Application")
    with st.form("create_applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Full Name", value="Priya Patel")
            age = st.number_input("Age", 21, 65, 29)
            gender = st.selectbox("Gender ", ["Female", "Male"])
            marital = st.selectbox("Marital Status ", ["Single", "Married"])
            edu = st.selectbox("Education Level", ["Graduate", "Post Graduate", "High School", "Professional"])
        with c2:
            salary = st.number_input("Monthly Salary", 15000.0, 500000.0, 60000.0, 5000.0)
            emp_type = st.selectbox("Employment", ["Private", "Government", "Self-employed"])
            exp = st.number_input("Experience (Yrs)", 0.0, 40.0, 3.0)
            comp = st.selectbox("Company", ["Mid-size", "Large", "Startup", "MNC", "Small"])
            credit = st.number_input("Credit Score", 300, 850, 720)
        with c3:
            scenario = st.selectbox("EMI Scenario", [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI"
            ])
            req_amt = st.number_input("Loan Amount", 5000.0, 2000000.0, 150000.0, 5000.0)
            tenure = st.number_input("Tenure (Mos)", 3, 84, 18)
            rent = st.number_input("Rent ", 0.0, 100000.0, 10000.0)
            cur_emi = st.number_input("Current EMI", 0.0, 100000.0, 0.0)

        create_btn = st.form_submit_button("💾 Save Applicant")
        if create_btn:
            new_rec = ApplicantRecord(
                customer_name=name,
                age=age,
                gender=gender,
                marital_status=marital,
                education=edu,
                monthly_salary=salary,
                employment_type=emp_type,
                years_of_employment=exp,
                company_type=comp,
                house_type="Rented",
                monthly_rent=rent,
                family_size=2,
                dependents=0,
                credit_score=credit,
                current_emi_amount=cur_emi,
                emi_scenario=scenario,
                requested_amount=req_amt,
                requested_tenure=tenure,
                predicted_eligibility="Pending Assessment",
                predicted_max_emi=0.0
            )
            db.add(new_rec)
            db.commit()
            st.success("New applicant record created successfully!")
            st.rerun()

# 3. UPDATE
with tab_update:
    st.subheader("Update Applicant Details")
    all_recs = db.query(ApplicantRecord).all()
    if all_recs:
        applicant_ids = [f"ID {r.id}: {r.customer_name}" for r in all_recs]
        selected_option = st.selectbox("Select Applicant to Modify", applicant_ids)
        sel_id = int(selected_option.split(":")[0].replace("ID ", ""))
        target = db.query(ApplicantRecord).filter(ApplicantRecord.id == sel_id).first()

        if target:
            with st.form("update_form"):
                u1, u2 = st.columns(2)
                with u1:
                    new_salary = st.number_input("Updated Monthly Salary", value=float(target.monthly_salary))
                    new_credit = st.number_input("Updated Credit Score", value=int(target.credit_score))
                with u2:
                    new_amount = st.number_input("Updated Requested Amount", value=float(target.requested_amount))
                    new_tenure = st.number_input("Updated Requested Tenure", value=int(target.requested_tenure))

                update_btn = st.form_submit_button("🔄 Apply Update")
                if update_btn:
                    target.monthly_salary = new_salary
                    target.credit_score = new_credit
                    target.requested_amount = new_amount
                    target.requested_tenure = new_tenure
                    db.commit()
                    st.success("Record updated successfully!")
                    st.rerun()
    else:
        st.info("No records to update.")

# 4. DELETE
with tab_delete:
    st.subheader("Delete Applicant Record")
    if all_recs:
        del_option = st.selectbox("Select Record to Remove", applicant_ids, key="del_select")
        del_id = int(del_option.split(":")[0].replace("ID ", ""))
        if st.button("❌ Confirm Deletion", type="primary"):
            del_target = db.query(ApplicantRecord).filter(ApplicantRecord.id == del_id).first()
            if del_target:
                db.delete(del_target)
                db.commit()
                st.warning(f"Record ID {del_id} deleted.")
                st.rerun()
    else:
        st.info("No records available to delete.")

db.close()
