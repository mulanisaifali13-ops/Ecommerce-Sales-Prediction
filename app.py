import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

st.set_page_config(page_title="E-Commerce Sales Dashboard", page_icon="🛒", layout="wide")

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv", parse_dates=["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = "Qtr " + df["Order Date"].dt.quarter.astype(str)
    df["MonthName"] = df["Order Date"].dt.strftime("%b")
    return df


def train_model(df):
    d = df[~df["is_outlier"]]
    agg = (d.groupby(["Year", "Month", "Category", "PaymentMode", "State"], as_index=False)
             .agg(Quantity=("Quantity", "sum"), Sales=("Amount", "sum")))
    X = agg[["Year", "Month", "Quantity", "Category", "PaymentMode", "State"]]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
                              ["Category", "PaymentMode", "State"])], remainder="passthrough")
    m = Pipeline([("pre", pre),
                  ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                       min_child_weight=5, subsample=0.9,
                                       colsample_bytree=0.9, random_state=42))])
    return m.fit(X, agg["Sales"])


@st.cache_resource
def load_model():
    try:
        return joblib.load("sales_model.pkl")
    except Exception:
        return train_model(load_data())


df = load_data()
model = load_model()

# ---------------- Sidebar filters ----------------
st.sidebar.title("🔎 Filters")
quarters = st.sidebar.multiselect("Quarter", sorted(df["Quarter"].unique()), default=sorted(df["Quarter"].unique()))
cats = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
states = st.sidebar.multiselect("State", sorted(df["State"].unique()), default=sorted(df["State"].unique()))
pays = st.sidebar.multiselect("Payment Mode", sorted(df["PaymentMode"].unique()), default=sorted(df["PaymentMode"].unique()))

excl = st.sidebar.checkbox("Exclude outliers", value=False)
base = df[~df["is_outlier"]] if excl else df
f = base[base["Quarter"].isin(quarters) & base["Category"].isin(cats) & base["State"].isin(states) & base["PaymentMode"].isin(pays)]

st.title("🛒 E-Commerce Business Intelligence & Sales Prediction")
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Predict Sales", "📈 Forecast"])

# ---------------- Tab 1: Dashboard ----------------
with tab1:
    if f.empty:
        st.warning("No data for the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Amount", f"₹{f['Amount'].sum():,.0f}")
        c2.metric("Total Quantity", f"{int(f['Quantity'].sum()):,}")
        c3.metric("Total Profit", f"₹{f['Profit'].sum():,.0f}")
        c4.metric("Avg Order Value", f"₹{f.groupby('Order ID')['Amount'].sum().mean():,.0f}")

        a, b = st.columns(2)
        m = f.groupby(["Year", "Month"], as_index=False)["Profit"].sum()
        m["Period"] = m["Year"].astype(str) + "-" + m["Month"].astype(str).str.zfill(2)
        a.plotly_chart(px.bar(m, x="Period", y="Profit", title="Profit by Month"), width="stretch")
        sc = f.groupby("Sub-Category", as_index=False)["Profit"].sum().sort_values("Profit")
        b.plotly_chart(px.bar(sc, x="Profit", y="Sub-Category", orientation="h", title="Profit by Sub-Category"),
                       width="stretch")

        a, b = st.columns(2)
        q = f.groupby("Category", as_index=False)["Quantity"].sum()
        a.plotly_chart(px.pie(q, names="Category", values="Quantity", hole=0.5, title="Quantity by Category"),
                       width="stretch")
        qp = f.groupby(["Category", "PaymentMode"], as_index=False)["Quantity"].sum()
        b.plotly_chart(px.bar(qp, x="Category", y="Quantity", color="PaymentMode", barmode="group",
                              title="Category × Payment Mode"), width="stretch")

        a, b = st.columns(2)
        s = f.groupby("State", as_index=False)["Amount"].sum().sort_values("Amount")
        a.plotly_chart(px.bar(s, x="Amount", y="State", orientation="h", title="Sales by State"),
                       width="stretch")
        cu = f.groupby("CustomerName", as_index=False)["Amount"].sum().nlargest(10, "Amount")
        b.plotly_chart(px.bar(cu, x="CustomerName", y="Amount", title="Top 10 Customers"),
                       width="stretch")

        with st.expander("View data"):
            st.dataframe(f, width="stretch")

# ---------------- Tab 2: Single prediction ----------------
with tab2:
    st.subheader("Predict monthly sales for a segment")
    c1, c2, c3 = st.columns(3)
    p_cat = c1.selectbox("Category", sorted(df["Category"].unique()))
    p_pay = c2.selectbox("Payment Mode", sorted(df["PaymentMode"].unique()))
    p_state = c3.selectbox("State", sorted(df["State"].unique()))
    c1, c2, c3 = st.columns(3)
    p_month = c1.selectbox("Month", range(1, 13), format_func=lambda x: MONTHS[x - 1])
    p_year = c2.number_input("Year", 2018, 2035, int(df["Year"].max()))
    p_qty = c3.number_input("Total quantity expected", 1, 5000, 20)

    if st.button("Predict", type="primary"):
        X = pd.DataFrame([{"Year": p_year, "Month": p_month, "Quantity": p_qty,
                           "Category": p_cat, "PaymentMode": p_pay, "State": p_state}])
        pred = max(float(model.predict(X)[0]), 0)
        st.success(f"Predicted sales: ₹{pred:,.0f}")

# ---------------- Tab 3: Forecast ----------------
with tab3:
    st.subheader("Forecast total sales for upcoming months")
    n = st.slider("Months ahead", 1, 12, 6)

    # typical monthly quantity for every Category/Payment/State segment
    seg = (df[~df["is_outlier"]].groupby(["Year", "Month", "Category", "PaymentMode", "State"], as_index=False)["Quantity"].sum()
             .groupby(["Category", "PaymentMode", "State"], as_index=False)["Quantity"].mean())
    last = df["Order Date"].max()
    rows = []
    for i in range(1, n + 1):
        d = (last + pd.DateOffset(months=i))
        x = seg.copy()
        x["Year"], x["Month"] = d.year, d.month
        x["Pred"] = np.maximum(model.predict(x[["Year", "Month", "Quantity", "Category", "PaymentMode", "State"]]), 0)
        rows.append({"Period": d.strftime("%Y-%m"), "Predicted Sales": x["Pred"].sum()})
    fc = pd.DataFrame(rows)

    clean = df[~df["is_outlier"]]
    hist = clean.groupby(clean["Order Date"].dt.to_period("M"))["Amount"].sum().reset_index()
    hist["Period"] = hist["Order Date"].astype(str)
    hist = hist.rename(columns={"Amount": "Sales"})[["Period", "Sales"]]
    hist["Type"] = "Actual"
    fc2 = fc.rename(columns={"Predicted Sales": "Sales"})
    fc2["Type"] = "Forecast"
    st.plotly_chart(px.line(pd.concat([hist, fc2]), x="Period", y="Sales", color="Type", markers=True),
                    width="stretch")
    st.dataframe(fc.style.format({"Predicted Sales": "₹{:,.0f}"}), width="stretch")
    st.caption("Assumes each segment sells its historical average monthly quantity.")
