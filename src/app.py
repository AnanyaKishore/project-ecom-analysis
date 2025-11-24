import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests
import warnings
import os
from itertools import product
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Writing the plots to ../assets/ as HTML files to host with GitHub
if not os.path.exists(r'../assets'):
    os.makedirs(r'../assets')

print("Downloading Brazil GeoJSON...")
try:
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    response = requests.get(geojson_url)
    response.raise_for_status()
    geojson = response.json()
    print("GeoJSON downloaded successfully.")
except Exception as e:
    print(f"Error downloading GeoJSON: {e}")
    geojson = None

df = pd.read_parquet(r"../data/merged_info_after_impute.parquet")
state_map = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AM": "Amazonas",
    "AP": "Amapá",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MG": "Minas Gerais",
    "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso",
    "PA": "Pará",
    "PB": "Paraíba",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "PR": "Paraná",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RO": "Rondônia",
    "RR": "Roraima",
    "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina",
    "SE": "Sergipe",
    "SP": "São Paulo",
    "TO": "Tocantins"
}

df['customer_state_full'] = df['customer_state'].map(state_map)
df["price_with_freight_charges"] = df["price"] + df["freight_value"]

df["diff_delivered_carrier"] = df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
df["diff_delivered_estimated"] = df["order_delivered_customer_date"].dt.normalize() - df["order_estimated_delivery_date"]
df["diff_carrier_limit"] = df["order_delivered_carrier_date"] - df["shipping_limit_date"]
df["diff_delivered_ordered"] = df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
df["diff_carrier_ordered"] = df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]

for col in ["diff_delivered_carrier", "diff_delivered_estimated", "diff_carrier_limit", "diff_delivered_ordered", "diff_carrier_ordered"]:
    df[col] = df[col].dt.days

# Logistics
unique_sellers = df.seller_id.nunique()
unique_customers = df.customer_unique_id.nunique()
unique_cities = df.customer_city.nunique()
unique_states = df.customer_state.nunique()
unique_regions = df.customer_zip_code_prefix.nunique()
unique_orders = df.order_id.nunique()
unique_products = df.product_id.nunique()

# Time-Based
def save_clean_fig(fig, filename, title, xaxis, yaxis):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        title_x=0.5,
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50))
    fig.update_traces(hovertemplate="Days: %{x}<br>Frequency: %{y}<extra></extra>")
    fig.write_html(fr"../assets/{filename}", include_mathjax=False, include_plotlyjs='cdn')
    print(f"Saved assets/{filename}")

# --- fig1 ---
fig1 = px.histogram(df.query("diff_delivered_carrier <= 91.0"), x="diff_delivered_carrier")
save_clean_fig(fig1, "fig1.html", "How Long Until Your Order Arrives After Shipping?", "Days", "Frequency")

# --- fig2 ---
fig2 = px.histogram(df.query("-60.0 <= diff_delivered_estimated <= 60.0"), x="diff_delivered_estimated")
save_clean_fig(fig2, "fig2.html", "How Early are Orders Delivered?", "Days (- / +)", "Frequency")

# --- fig3 ---
fig3 = px.histogram(df.query("-20.0 <= diff_carrier_limit <= 60.0"), x="diff_carrier_limit")
save_clean_fig(fig3, "fig3.html", "How Early Are Orders Shipped?", "Days (- / +)", "Frequency")

# --- fig4 ---
fig4 = px.histogram(df.query("diff_delivered_ordered <= 75.0"), x="diff_delivered_ordered")
save_clean_fig(fig4, "fig4.html", "How Long Does Delivery Take?", "Days", "Frequency")

# --- fig5 ---
fig5 = px.histogram(df.query("diff_carrier_ordered <= 50.0"), x="diff_carrier_ordered")
save_clean_fig(fig5, "fig5.html", "How Long Does It Take to Ship?", "Days", "Frequency")

# --- fig1_choropleth ---
# Average delivery time after shipping by state
pio.templates.default = "plotly_white"

state_delivered_after_shipping = (
    df.groupby(['customer_state'], observed=False)
      .agg(avg_delivered_after_ship=('diff_delivered_carrier', 'mean'))
      .reset_index()
)

# Add full state names
state_delivered_after_shipping["customer_state_full"] = \
    state_delivered_after_shipping["customer_state"].map(state_map)

# Choropleth
fig1_choropleth = px.choropleth(
    state_delivered_after_shipping,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="avg_delivered_after_ship",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="State-Wise: How Long Until Your Order Arrives After Shipping?",
    hover_name="customer_state_full",
    hover_data={},  
)

# Hover formatting
fig1_choropleth.update_traces(
    customdata=state_delivered_after_shipping[['avg_delivered_after_ship']],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +
        "Avg Delivery After Shipping: %{customdata[0]:.2f} days<br>" +
        "<extra></extra>"
    )
)

# Map bounds
fig1_choropleth.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)
fig1_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig1_choropleth.update_geos(fitbounds="locations", visible=False)
# Layout + legend title
fig1_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Delay (Days)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5
    )

# --- fig2_choropleth ---
# Average delivery timing vs estimated delivery date by state
pio.templates.default = "plotly_white"

state_delivery_est = (
    df.groupby(['customer_state'], observed=False)
      .agg(avg_diff_estimated=('diff_delivered_estimated', 'mean'))
      .reset_index()
)

# Add full state names
state_delivery_est["customer_state_full"] = state_delivery_est["customer_state"].map(state_map)

# Choropleth
fig2_choropleth = px.choropleth(
    state_delivery_est,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="avg_diff_estimated",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="State-Wise: How Early are Orders Delivered?",
    hover_name="customer_state_full",
    hover_data={},  
)

# Hover formatting
fig2_choropleth.update_traces(
    customdata=state_delivery_est[['avg_diff_estimated']],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +
        "Avg Difference: %{customdata[0]:.2f} days<br>" +
        "<extra></extra>"
    )
)

# Map bounds
fig2_choropleth.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

# Layout + legend rename
fig2_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Diff (Days)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5
    )
fig2_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig2_choropleth.update_geos(fitbounds="locations", visible=False)

# --- fig3_choropleth ---
# Average shipping timing vs shipping-limit by state
pio.templates.default = "plotly_white"

state_shiplimit = (
    df.groupby(['customer_state'], observed=False)
      .agg(avg_shiplimit_diff=('diff_carrier_limit', 'mean'))
      .reset_index()
)

# Add full state names
state_shiplimit["customer_state_full"] = state_shiplimit["customer_state"].map(state_map)

# Choropleth
fig3_choropleth = px.choropleth(
    state_shiplimit,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="avg_shiplimit_diff",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="State-Wise: How Early Are Orders Shipped?",
    hover_name="customer_state_full",
    hover_data={},  
)

# Hover formatting
fig3_choropleth.update_traces(
    customdata=state_shiplimit[['avg_shiplimit_diff']],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +
        "Avg Difference: %{customdata[0]:.2f} days<br>" +
        "<extra></extra>"
    )
)

# Map bounds
fig3_choropleth.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

# Layout + legend rename
fig3_choropleth.update_layout(
    title_x=0.5,
    coloraxis_colorbar=dict(title="Avg Diff (Days)")
)
fig3_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Diff (Days)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5
    )
fig3_choropleth.update_geos(fitbounds="locations", visible=False)

# --- fig4_choropleth ---
# Average delivery time (order → delivered) by state
pio.templates.default = "plotly_white"

state_delivery = (
    df.groupby(['customer_state'], observed=False)
      .agg(avg_delivery_time=('diff_delivered_ordered', 'mean'))
      .reset_index()
)

# Add full state names
state_delivery["customer_state_full"] = state_delivery["customer_state"].map(state_map)

# Choropleth
fig4_choropleth = px.choropleth(
    state_delivery,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="avg_delivery_time",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="State-Wise: How Long Does Delivery Take?",
    hover_name="customer_state_full",
    hover_data={},  
)

# Hover formatting
fig4_choropleth.update_traces(
    customdata=state_delivery[['avg_delivery_time']],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +
        "Average Delivery Time: %{customdata[0]:.2f} days<br>" +
        "<extra></extra>"
    )
)

# Map bounds
fig4_choropleth.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

# Layout + legend rename
fig4_choropleth.update_layout(
    title_x=0.5,
    coloraxis_colorbar=dict(title="Avg Time (Days)")
)
fig4_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Time (Days)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5
    )
fig4_choropleth.update_geos(fitbounds="locations", visible=False)

# --- fig5_choropleth ---
# Average shipping delay (order → carrier pickup) by state
pio.templates.default = "plotly_white"

# Compute state averages
state_delay = (
    df.groupby(['customer_state'], observed=False)
      .agg(avg_shipping_delay=('diff_carrier_ordered', 'mean'))
      .reset_index()
)

# Add full state names
state_delay["customer_state_full"] = state_delay["customer_state"].map(state_map)

# Choropleth
fig5_choropleth = px.choropleth(
    state_delay,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="avg_shipping_delay",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="State-Wise: How Long Does It Take to Ship?",
    hover_name="customer_state_full",
    hover_data={},  
)

# Hover formatting
fig5_choropleth.update_traces(
    customdata=state_delay[['avg_shipping_delay']],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +
        "Average Shipping Delay: %{customdata[0]:.2f} days<br>" +
        "<extra></extra>"
    )
)

# Map bounds
fig5_choropleth.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

# Layout + legend rename
fig5_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Delay (Days)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5)
fig5_choropleth.update_geos(fitbounds="locations", visible=False)

# Customers
# --- fig6 ---
# Number of customers per state
pio.templates.default = "plotly_white"

state_summary = df.groupby(['customer_state'], observed=False).agg(
    customer_count=('customer_unique_id', 'nunique'), city_count = ("customer_city", "nunique"), zip_count = ('customer_zip_code_prefix', "nunique")).reset_index()
state_summary["customer_state_full"] = state_summary["customer_state"].map(state_map)

fig6 = px.choropleth(
    state_summary,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="customer_count",
    color_continuous_scale="Turbo",
    scope="world",
    title="Customer Demographics by State, City & Region",
    hover_name="customer_state_full",
    hover_data={},  
)

fig6.update_traces(
    customdata = state_summary[['customer_count', 'city_count', 'zip_count']],
    hovertemplate=(
        "<b>%{hovertext}</b><br>" +
        "Number of Customers: %{customdata[0]:,}<br>" +
        "Cities: %{customdata[1]:,}<br>" +
        "Unique Regions: %{customdata[2]:,}<br>" +
        "<extra></extra>"
    )
)

fig6.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

fig6.update_layout(title_x=0.5, coloraxis_colorbar=dict(title="Customer Count"))

fig6.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Customer Count", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5
    )
fig6.update_geos(fitbounds="locations", visible=False)

# --- fig7 ---
df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
df['hour_of_day'] = df['order_purchase_timestamp'].dt.hour
days_of_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hourly_activity = df.groupby(['day_of_week', 'hour_of_day'])['order_id'].nunique().reset_index()
hourly_pivot = hourly_activity.pivot(
    index='day_of_week',
    columns='hour_of_day',
    values='order_id'
)
hourly_pivot = hourly_pivot.reindex(days_of_week_order)
fig7 = px.imshow(
    hourly_pivot,
    title="Order Activity: When Do Customers Shop?",
    labels=dict(x="Hour of Day", y="Day of Week", color="Total Orders"),
    color_continuous_scale="Plasma"
)
fig7.update_xaxes(nticks=24)

fig7.update_layout(
    coloraxis_colorbar=dict(
        title=dict(
            text="Total Orders", 
            side="right",              
            font=dict(size=12)
        )
    ), title_x=0.5
    )

# State-Wise
# --- fig8 ---
orders_per_state = (
    df.groupby('customer_state_full', observed=False)['order_id']
    .nunique()
    .reset_index()
)
orders_per_state.columns = ['customer_state_full', 'num_orders']

# Keep only rows with valid delivery and estimated delivery dates
df_valid = df.dropna(subset=['order_estimated_delivery_date', 'order_delivered_customer_date'])

# Create late indicator
df_valid['is_late'] = df_valid['order_delivered_customer_date'] > df_valid['order_estimated_delivery_date']

# Group late deliveries per state
late_deliveries = (
    df_valid.query("is_late")
    .groupby('customer_state_full', observed=False)['order_id']
    .nunique()
    .reset_index()
)
late_deliveries.columns = ['customer_state_full', 'late_orders']

# Compute percentage
late_deliveries['late_orders_percentage'] = (
    late_deliveries['late_orders'] / orders_per_state['num_orders']
)

# Sort by percentage
late_deliveries = late_deliveries.sort_values('late_orders_percentage', ascending=False)

# Plot
fig8 = px.bar(
    late_deliveries,
    x='customer_state_full',
    y='late_orders_percentage',
    color='late_orders_percentage',
    color_continuous_scale="Turbo",
    title='Percentage of Late Deliveries per State',
    labels={
        'customer_state_full': 'State',
        'late_orders_percentage': 'Percentage of Late Orders'
    }
)

# Center title and adjust layout
fig8.update_layout(
    title={'x': 0.5},
    xaxis=dict(categoryorder='total descending', tickangle = -45,  tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white',
    margin=dict(t=45),
    coloraxis_colorbar=dict(
        title=dict(
            text="Percentage of Late Orders", 
            side="right",              
            font=dict(size=12)
        ),
    )
)

# Annotate bars with values rounded to 2 decimals
fig8.update_traces(
    text=late_deliveries['late_orders_percentage'].round(2),
    textposition='outside'
)

# --- fig9 ---
# Avg sales price by state
pio.templates.default = "plotly_white"

state_summary = df.groupby(
    ['customer_state', 'customer_state_full'], observed=False
).agg(
    average_sales=('price', 'mean'),
    customer_count=('customer_unique_id', 'nunique')
).reset_index()

fig9 = px.choropleth(
    state_summary,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="average_sales",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="Average Sales Price by State",
    hover_name="customer_state_full",
    hover_data={},  
)

fig9.update_traces(
    customdata=state_summary[['average_sales']],
    hovertemplate=
        "<b>%{hovertext}</b><br>" +
        "Average Sales: $%{customdata[0]:,.2f}<br>" +
        "<extra></extra>"
)

fig9.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

fig9.update_geos(fitbounds="locations", visible=False)
fig9.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Sales ($)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5
    )

# --- fig10 ---
# Avg freight price by state
pio.templates.default = "plotly_white"

state_freight = df.groupby(
    ['customer_state', 'customer_state_full'], observed=False
)['freight_value'].mean().reset_index()

fig10 = px.choropleth(
    state_freight,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="freight_value",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="Average Freight Cost by State",
    hover_name="customer_state_full",
    hover_data={},       # remove automatic extras
)

# Match the custom hovertemplate style
fig10.update_traces(
    customdata=state_freight[['freight_value']],
    hovertemplate=
        "<b>%{hovertext}</b><br>" +
        "Average Freight: $%{customdata[0]:,.2f}<br>" +
        "<extra></extra>"
)

# Same geo formatting
fig10.update_geos(
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

# Match title + legend formatting
fig10.update_layout(title_x=0.5,
    coloraxis_colorbar=dict(
    title=dict(
        text="Avg Freight ($)", 
        side="right",              
        font=dict(size=12)),
    x = 0.85
    )
)
fig10.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig10.update_geos(fitbounds="locations", visible=False)

# Trends
# --- fig11 ---
# Avg Delivery Time per Month
df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

delivery_trend = (
    df.groupby(df['order_purchase_timestamp'].dt.to_period('M'))['delivery_time_days']
    .mean()
    .reset_index()
)
delivery_trend['order_purchase_timestamp'] = delivery_trend['order_purchase_timestamp'].astype(str)

fig11 = px.line(
    delivery_trend,
    x='order_purchase_timestamp',
    y='delivery_time_days',
    title='Average Delivery Time (days) By Month',
    markers = True,
    labels={'order_purchase_timestamp': 'Month', 'delivery_time_days': 'Avg Delivery Time (days)'}
)

fig11.update_traces(marker=dict(color="#ed1b76"))
fig11.update_layout(title_x = 0.5)

# --- fig12 ---
# Monthly Orders
orders_monthly = df.groupby(df['order_purchase_timestamp'].dt.to_period('M')).size().reset_index(name='num_orders')
orders_monthly['order_purchase_timestamp'] = orders_monthly['order_purchase_timestamp'].astype(str)

fig12 = px.bar(
    orders_monthly,
    x='order_purchase_timestamp',
    y='num_orders',
    title='Monthly Orders Trend',
    labels={'order_purchase_timestamp': 'Month', 'num_orders': 'Orders'},
    color='num_orders',
    color_continuous_scale='Plasma',
    text='num_orders'                     # show numbers
)

fig12.update_traces(
    texttemplate='%{text}',
    textposition='outside'                # place above bars
)

fig12.update_layout(
    title_x=0.5,
    uniformtext_minsize=8,
    uniformtext_mode='hide',              # avoid overlap
    margin=dict(t=50),
    coloraxis_colorbar=dict(
        title=dict(
            text="Orders", 
            side="right",              
            font=dict(size=12)
        )
    )
    )

# Dynamic Plots
# --- fig13 ---
# Monthly Sales by State (Using customer_state_full)

df['month_year'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

# Use the full state name column
all_states = df['customer_state_full'].unique()
all_months = df['month_year'].unique()
all_months.sort()

scaffold = pd.DataFrame(list(product(all_states, all_months)), 
                        columns=['customer_state_full', 'month_year'])

monthly_data = df.groupby(['customer_state_full', 'month_year'], observed=False).agg(
    monthly_sales=('price', 'sum')
).reset_index()

padded_data = pd.merge(scaffold, monthly_data,
                       on=['customer_state_full', 'month_year'],
                       how='left')

padded_data['monthly_sales'] = padded_data['monthly_sales'].fillna(1)
padded_data = padded_data[padded_data['monthly_sales'] > 0]

fig13 = px.bar(
    padded_data,
    x="customer_state_full",
    y="monthly_sales",
    color="customer_state_full",
    animation_frame="month_year",
    animation_group="customer_state_full",
    title="Monthly Sales by State (Log Scale)",
    log_y=True,
    labels={
        "customer_state_full": "Customer State",
        "monthly_sales": "Monthly Sales"
    }
)

fig13.update_layout(
    yaxis_range=[np.log10(1), np.log10(padded_data['monthly_sales'].max()) * 1.05],
    title_x=0.5,
    xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
    showlegend = False,
    xaxis_title = None,
)

fig13.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': '',
            'font': {'size': 12}
        }
    }]
)

# --- fig14 ---
# Cumulative Customer Growth
first_purchase = df.groupby('customer_unique_id', observed=True).agg(
    first_purchase_date=('order_purchase_timestamp', 'min'),
    customer_state_full=('customer_state_full', 'first')
).reset_index()

first_purchase['acquisition_month'] = first_purchase['first_purchase_date'].dt.to_period('M').astype(str)
all_months = sorted(first_purchase['acquisition_month'].unique())
all_states = df['customer_state_full'].unique()
all_states_df = pd.DataFrame(all_states, columns=['customer_state_full'])

cumulative_data_list = []
for month in all_months:
    temp_df = first_purchase[first_purchase['acquisition_month'] <= month]
    monthly_snapshot = temp_df.groupby('customer_state_full', observed=True)['customer_unique_id'].nunique().reset_index()
    monthly_snapshot.columns = ['customer_state_full', 'cumulative_customers']
    padded_snapshot = pd.merge(all_states_df, monthly_snapshot, on='customer_state_full', how='left')
    padded_snapshot['cumulative_customers'] = padded_snapshot['cumulative_customers'].fillna(1)
    padded_snapshot['month_year'] = month
    cumulative_data_list.append(padded_snapshot)

cumulative_data = pd.concat(cumulative_data_list, ignore_index=True)
cumulative_data = cumulative_data.sort_values(by=['month_year', 'cumulative_customers'])

fig14 = px.bar(
    cumulative_data,
    x='cumulative_customers',
    y='customer_state_full',
    orientation='h',
    color='customer_state_full',
    animation_frame='month_year',
    animation_group='customer_state_full',
    title='Cumulative Customer Growth by State (Log Scale)',
    log_x=True,
    labels={'cumulative_customers': 'Cumulative Customers (Log Scale)', 'customer_state_full': 'Customer State'}
)
fig14.update_layout(
    xaxis_range=[np.log10(1), np.log10(cumulative_data['cumulative_customers'].max()) * 1.05],
    showlegend = False,
    xaxis_title = None,
    xaxis=dict(showticklabels=False)
)
fig14.update_layout(title_x=0.5)
fig14.update_layout(
    sliders=[{
        'currentvalue': {
            'prefix': '',
            'font': {'size': 12}
        }
    }]
)

# -- fig15 --
items_per_order = df.groupby('order_id')['order_item_id'].max().reset_index()
items_per_order = items_per_order.rename(columns={'order_item_id': 'num_items'})
df_with_num_items = pd.merge(df, items_per_order, on='order_id')
avg_price_data = df_with_num_items.groupby('num_items')['price'].mean().reset_index()

# Create formatted label for display only
avg_price_data['price_label'] = avg_price_data['price'].round(2).apply(lambda x: f"${x}")

fig15 = px.bar(
    avg_price_data,
    x='num_items',
    y='price',
    color='price',
    color_continuous_scale='Plasma',
    text='price_label',               # display label with dollar sign
    hover_data={'price_label': False, 'price': ':.2f'},  # hide label; format price nicely
    title='Average Item Price Based On Number Of Items Purchased',
    labels={'num_items': 'Number of Items Purchased', 'price': 'Average Item Price ($)'}
)

fig15.update_traces(textposition='outside')
fig15.update_xaxes(type='category')

fig15.update_layout(
    title_x=0.5,
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    margin=dict(t=75),
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg Item Price ($)",
            side="right",
            font=dict(size=12)
        )
    )
)

# -- fig16 --
status_df = (
    df.query("order_status != 'delivered'")
      .assign(order_status_cap=lambda x: x['order_status'].str.capitalize())
      .reset_index(drop=True)
)

fig16 = px.pie(
    status_df,
    names='order_status_cap',
    title='Order Status Distribution Excluding Delivered Orders',
    color='order_status_cap',
    color_discrete_sequence=px.colors.qualitative.Bold
)

fig16.update_traces(
    textposition='inside',
    textinfo='label+percent',
    textfont=dict(size=11),                        # smaller label text
    pull=0.03,
    rotation=90,
    hovertemplate="Order Status: %{label}<br>Percent: %{percent}<extra></extra>"
)

fig16.update_layout(
    title_x=0.5,
    margin=dict(t=80, b=30, l=30, r=30),
    showlegend=False
)

# -- fig17 --
# Compute CLV per customer
clv_df = (
    df.groupby("customer_unique_id", observed=False)
      .agg(lifetime_value=("price_with_freight_charges", "sum"),
           state=("customer_state", "first"))
      .reset_index()
)

# Aggregate to state-level CLV
clv_state = (
    clv_df.groupby("state", observed=False)
           .lifetime_value
           .mean()
           .reset_index()
           .rename(columns={"lifetime_value": "avg_clv"})
)
clv_state["customer_state_full"] = clv_state["state"].map(state_map)
fig17 = px.choropleth(
    clv_state,
    geojson=geojson,
    locations="state",
    featureidkey="properties.sigla",
    color="avg_clv",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="Average Customer Lifetime Value by State",
    hover_name="customer_state_full",
    hover_data={},
)

fig17.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>Average CLV: $%{customdata[1]:,.2f}<extra></extra>",
    customdata=clv_state[["customer_state_full", "avg_clv"]].to_numpy()
)

fig17.update_geos(
    fitbounds="locations",
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

fig17.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Avg CLV ($)", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5)

# -- fig18 --
order_df = df.groupby("customer_state", observed = False).agg(orders_count = ("order_id", "nunique")).reset_index()
order_df["customer_state_full"] = order_df["customer_state"].map(state_map)
fig18 = px.choropleth(
    order_df,
    geojson=geojson,
    locations="customer_state",
    featureidkey="properties.sigla",
    color="orders_count",
    color_continuous_scale="RdBu_r",
    scope="world",
    title="Total Orders by State",
    hover_name="customer_state_full",
    hover_data={},
)

fig18.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>Orders Count: %{customdata[1]}<extra></extra>",
    customdata=order_df[["customer_state_full", "orders_count"]].to_numpy()
)

fig18.update_geos(
    fitbounds="locations",
    visible=False,
    lataxis_range=[-38, 10],
    lonaxis_range=[-78, -30]
)

fig18.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(
        title=dict(
            text="Total Orders", 
            side="right",              
            font=dict(size=12)
        ), x = 0.85
    ), title_x=0.5)

figures_to_save = {
    "fig1_choropleth.html": fig1_choropleth,
    "fig2_choropleth.html": fig2_choropleth,
    "fig3_choropleth.html": fig3_choropleth,
    "fig4_choropleth.html": fig4_choropleth,
    "fig5_choropleth.html": fig5_choropleth,
    "fig6.html": fig6,
    "fig7.html": fig7,
    "fig8.html": fig8,
    "fig9.html": fig9,
    "fig10.html": fig10,
    "fig11.html": fig11,
    "fig12.html": fig12,
    "fig13.html": fig13,
    "fig14.html": fig14,
    "fig15.html": fig15,
    "fig16.html": fig16,
    "fig17.html": fig17,
    "fig18.html": fig18
}

# Save them
for filename, fig in figures_to_save.items():
    # This keeps the graph interactive but removes the heavy modebar to look cleaner
    fig.write_html(fr"../assets/{filename}", config={'displayModeBar': False})
    print(f"Saved assets/{filename}")