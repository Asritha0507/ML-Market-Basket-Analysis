import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBRanker

st.set_page_config(
    page_title="Instacart Next Basket Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("Instacart Next Basket Prediction")
st.write(
    "Predict the products a user is most likely to purchase in their next "
    "order using an XGBoost ranking model, with department- and aisle-aware "
    "candidate generation and a SHAP explanation for every recommendation."
)

st.divider()

MODEL_PATH = "models/xgboost_next_basket_ranker.json"
DATA_DIR = "data"

# Pool sizes validated in notebook 04.
TOP_GLOBAL_CANDIDATES = 500
TOP_DEPARTMENT_CANDIDATES = 20
TOP_AISLE_CANDIDATES = 20
USER_TOP_DEPARTMENTS = 3
USER_TOP_AISLES = 3

FEATURE_COLS = [
    "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order",
    "times_purchased_before", "first_purchase_before", "last_purchase_before",
    "recency_before", "purchase_span_before", "was_previously_purchased",
    "aisle_id", "department_id", "previous_orders", "avg_basket_before",
    "max_basket_before", "total_items_before", "order_progress",
]


# ---------------------------------------------------------------------------
# Cached loaders (cache_resource: no copying of the large DataFrames per rerun)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = XGBRanker()
    model.load_model(MODEL_PATH)
    return model


@st.cache_resource
def load_raw_data():
    orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
    products = pd.read_csv(f"{DATA_DIR}/products.csv")
    prior = pd.read_csv(f"{DATA_DIR}/order_products_prior.csv")
    train = pd.read_csv(f"{DATA_DIR}/order_products_train.csv")
    return orders, products, prior, train


@st.cache_resource
def build_candidate_pools(_products, _prior):
    """Global / department / aisle popularity pools (notebook 04 style),
    computed from product-level counts so it stays cheap."""
    product_counts = (
        _prior.groupby("product_id").size().rename("count").reset_index()
        .merge(_products[["product_id", "aisle_id", "department_id"]], on="product_id")
        .sort_values("count", ascending=False)
    )

    global_candidates = product_counts.head(TOP_GLOBAL_CANDIDATES)["product_id"].tolist()

    dept_candidates = {
        int(d): g.head(TOP_DEPARTMENT_CANDIDATES)["product_id"].tolist()
        for d, g in product_counts.groupby("department_id")
    }
    aisle_candidates = {
        int(a): g.head(TOP_AISLE_CANDIDATES)["product_id"].tolist()
        for a, g in product_counts.groupby("aisle_id")
    }

    return {
        "global": global_candidates,
        "dept": dept_candidates,
        "aisle": aisle_candidates,
    }


# ---------------------------------------------------------------------------
# Candidates + features
# ---------------------------------------------------------------------------

def get_user_candidates(user_prior, products, pools):
    """History + global top-500 + top-20 of the user's top-3 departments
    and top-3 aisles. user_prior = rows from the user's previous orders."""
    candidates = set(pools["global"])

    if user_prior.empty:
        return candidates  # cold start: global pool only

    candidates.update(user_prior["product_id"].unique().tolist())

    with_cat = user_prior.merge(
        products[["product_id", "aisle_id", "department_id"]], on="product_id"
    )

    top_depts = with_cat.groupby("department_id").size().nlargest(USER_TOP_DEPARTMENTS).index
    for d in top_depts:
        candidates.update(pools["dept"].get(int(d), []))

    top_aisles = (
        with_cat.groupby("aisle_id")["product_id"].nunique()
        .nlargest(USER_TOP_AISLES).index
    )
    for a in top_aisles:
        candidates.update(pools["aisle"].get(int(a), []))

    return candidates


def build_features(candidate_ids, user_prior, previous_orders, latest_order, products):
    """Vectorized version of the per-product feature calculation
    (same 17 features, same definitions as before)."""
    feats = pd.DataFrame({"product_id": list(candidate_ids)})

    if user_prior.empty:
        stats = pd.DataFrame(
            columns=["times_purchased_before", "first_purchase_before", "last_purchase_before"]
        )
        order_sizes = pd.Series(dtype=float)
    else:
        stats = user_prior.groupby("product_id")["order_number"].agg(
            times_purchased_before="count",
            first_purchase_before="min",
            last_purchase_before="max",
        )
        order_sizes = user_prior.groupby("order_id").size()

    feats = feats.merge(stats, left_on="product_id", right_index=True, how="left")
    for c in ["times_purchased_before", "first_purchase_before", "last_purchase_before"]:
        feats[c] = feats[c].fillna(0)

    bought = feats["times_purchased_before"] > 0
    feats["recency_before"] = np.where(
        bought, latest_order["order_number"] - feats["last_purchase_before"], 0
    )
    feats["purchase_span_before"] = np.where(
        bought, feats["last_purchase_before"] - feats["first_purchase_before"], 0
    )
    feats["was_previously_purchased"] = bought.astype(int)

    feats = feats.merge(
        products[["product_id", "aisle_id", "department_id"]], on="product_id", how="left"
    )

    n_prev = len(previous_orders)
    feats["order_number"] = latest_order["order_number"]
    feats["order_dow"] = latest_order["order_dow"]
    feats["order_hour_of_day"] = latest_order["order_hour_of_day"]
    feats["days_since_prior_order"] = latest_order["days_since_prior_order"]
    feats["previous_orders"] = n_prev
    feats["avg_basket_before"] = order_sizes.mean() if len(order_sizes) else 0
    feats["max_basket_before"] = order_sizes.max() if len(order_sizes) else 0
    feats["total_items_before"] = order_sizes.sum() if len(order_sizes) else 0
    feats["order_progress"] = (
        n_prev / latest_order["order_number"] if latest_order["order_number"] > 0 else 0
    )

    return feats


def explain_recommendation(shap_row, n=3):
    """Top-n |SHAP| features split by direction (notebook 13 logic)."""
    top = shap_row.abs().sort_values(ascending=False).head(n).index
    positive = [f for f in top if shap_row[f] > 0]
    negative = [f for f in top if shap_row[f] < 0]
    return positive, negative


def reorder_signal(row):
    """Plain-language label from recency/frequency (patterns from notebook 12).
    A heuristic display label, not a separate model."""
    recency, times = row["recency_before"], row["times_purchased_before"]
    if times == 0:
        return "New to this user"
    if recency <= 1 and times >= 3:
        return "Likely due: bought recently and often"
    if recency <= 1:
        return "Bought recently"
    if times >= 10:
        return "Frequently reordered"
    return "Occasional purchase"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

model = load_model()
orders, products, prior_products, train_products = load_raw_data()
pools = build_candidate_pools(products, prior_products)

st.subheader("User Selection")
user_id = st.number_input("Enter User ID", min_value=1, step=1)

st.subheader("User Information")
user_orders = orders[orders["user_id"] == user_id].sort_values("order_number")
total_orders = len(user_orders)

col1, col2 = st.columns(2)
with col1:
    st.metric("Selected User ID", user_id)
with col2:
    st.metric("Total Orders", total_orders)

st.subheader("Order History")
with st.expander("View Order History", expanded=False):
    st.dataframe(
        user_orders.sort_values("order_number", ascending=False),
        use_container_width=True, hide_index=True,
    )

st.subheader("Latest Order")
with st.expander("View Latest Order", expanded=True):
    if total_orders:
        latest = user_orders.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Order Number", int(latest["order_number"]))
        c2.metric("Day of Week", int(latest["order_dow"]))
        c3.metric("Order Hour", f'{int(latest["order_hour_of_day"])}:00')
        st.dataframe(user_orders.tail(1), use_container_width=True, hide_index=True)
    else:
        st.info("No orders found for this user.")

st.subheader("Latest Order Products")
if st.button("View Latest Order Products"):
    if total_orders == 0:
        st.warning("No orders found for this User ID.")
    else:
        latest_order_id = user_orders.iloc[-1]["order_id"]
        latest_products = train_products[train_products["order_id"] == latest_order_id].merge(
            products[["product_id", "product_name"]], on="product_id", how="left"
        )
        with st.expander("View Products in Latest Order", expanded=True):
            if not latest_products.empty:
                st.metric("Products in Latest Order", len(latest_products))
                st.dataframe(latest_products, use_container_width=True, hide_index=True)
            else:
                st.info("No products found for the latest order.")

st.divider()
top_k = st.selectbox("Number of recommendations", [5, 10, 20], index=0)

if st.button("Predict Next Basket", type="primary"):
    if total_orders == 0:
        st.warning("No orders found for this User ID.")
    else:
        latest_order = user_orders.iloc[-1]
        previous_orders = user_orders[user_orders["order_number"] < latest_order["order_number"]]

        # Filter the big prior table once, not once per candidate.
        user_prior = prior_products[
            prior_products["order_id"].isin(previous_orders["order_id"])
        ].merge(previous_orders[["order_id", "order_number"]], on="order_id")

        if previous_orders.empty:
            st.info("No prior history for this user: using globally popular products (cold start).")

        candidate_ids = get_user_candidates(user_prior, products, pools)
        feats = build_features(candidate_ids, user_prior, previous_orders, latest_order, products)

        st.subheader("Candidate Products")
        c1, c2 = st.columns(2)
        c1.metric("Candidate Pool Size", len(feats))
        c2.metric(
            "Pool Sources",
            "History + Global + Dept + Aisle" if not previous_orders.empty else "Global (cold start)",
        )

        X = feats[FEATURE_COLS].astype(float)
        feats["score"] = model.predict(X)

        # Original row index is kept through sorting so Top-K rows match their own feature rows for SHAP.
        ranked = feats.sort_values("score", ascending=False)
        names = products.set_index("product_id")["product_name"]
        ranked["product_name"] = ranked["product_id"].map(names)

        top = ranked.head(top_k)
        X_top = X.loc[top.index]

        explainer = shap.TreeExplainer(model)
        shap_df = pd.DataFrame(
            explainer.shap_values(X_top), columns=FEATURE_COLS, index=X_top.index
        )

        st.subheader(f"Top-{top_k} Recommended Products")
        for rank, (idx, row) in enumerate(top.iterrows(), start=1):
            positive, negative = explain_recommendation(shap_df.loc[idx])
            with st.container(border=True):
                a, b, c = st.columns([3, 1, 2])
                a.markdown(f"**#{rank}. {row['product_name']}**")
                a.caption(f"Product ID {int(row['product_id'])}")
                b.metric("Score", f"{row['score']:.3f}")
                c.markdown(f"*{reorder_signal(row)}*")
                parts = []
                if positive:
                    parts.append("Pushed up by: " + ", ".join(positive))
                if negative:
                    parts.append("Pulled down by: " + ", ".join(negative))
                st.caption(" · ".join(parts) if parts else "No strong drivers")

        with st.expander("View Full Ranked Candidates"):
            st.dataframe(
                ranked[["product_id", "product_name", "score"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

        with st.expander("View Model Feature Data"):
            st.dataframe(feats[FEATURE_COLS + ["score"]], use_container_width=True, hide_index=True)

        st.subheader("User History Summary")
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Previous Orders", len(previous_orders))
        h2.metric("Unique Products Bought", int(user_prior["product_id"].nunique()))
        h3.metric("Total Products Purchased", int(len(user_prior)))