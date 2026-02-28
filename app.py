import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import optimize

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# --- PORTADA / ENCABEZADO ---
st.title("📊 My Portfolio Management")
st.markdown("### Reporte de rendimiento & risk assessment")
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.divider()

# --- FUNCIONES FINANCIERAS ---
def xirr_core(cashflows, dates):
    if len(cashflows) < 2: return 0.0
    try:
        start_date = min(dates)
        def npv(rate):
            return sum([cf / (1 + rate) ** ((d - start_date).days / 365.0) for cf, d in zip(cashflows, dates)])
        return optimize.brentq(npv, -0.99, 10.0)
    except:
        return 0.0

def obtener_metricas_riesgo(serie_cartera, serie_spy, rf=0.04):
    if len(serie_cartera) < 5: return [0]*5
    rets = serie_cartera.pct_change().dropna()
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - rf) / vol if vol > 0 else 0
    sortino = (rets.mean() * 252 - rf) / (rets[rets<0].std() * np.sqrt(252)) if len(rets[rets<0])>0 else 0
    beta = 1.0
    if not serie_spy.empty:
        common = pd.concat([rets, serie_spy.pct_change().dropna()], axis=1).dropna()
        if len(common) > 5:
            beta = np.cov(common.iloc[:,0], common.iloc[:,1])[0][1] / np.var(common.iloc[:,1])
    var_95 = np.percentile(rets, 5)
    return vol, sharpe, sortino, beta, var_95

@st.cache_data(show_spinner=False)
def simular_cartera_final(df_movs, symbol_map):
    if df_movs.empty: return None, None, None
    df = df_movs.copy()
    df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
    start_date = df['fecha'].min()
    end_date = pd.Timestamp.now().normalize()
    
    activos = [i for i in df['instrumento'].unique() if i != 'CASH']
    precios = pd.DataFrame()
    with st.spinner('Descargando precios de mercado...'):
        for act in list(set(activos + ['SPY'])):
            tk = symbol_map.get(act, f"{act}.US")
            try:
                tmp = web.DataReader(tk, 'stooq', start=start_date - timedelta(days=15), end=end_date)
                precios[act] = tmp['Close']
            except: pass
    
    precios = precios.sort_index().ffill().bfill()
    fechas_rango = pd.date_range(start_date, end_date)
    precios = precios.reindex(fechas_rango).ffill().bfill()
    
    hist_cartera, hist_spy = [], []
    posiciones = {a: 0.0 for a in activos}
    caja, pos_spy = 0.0, 0.0
    
    for dia in fechas_rango:
        m_dia = df[df['fecha'] == dia]
        p_spy = precios.loc[dia, 'SPY'] if 'SPY' in precios.columns else 1.0
        for _, m in m_dia.iterrows():
            m_tipo, m_monto = str(m['tipo']).upper(), abs(m['monto'])
            if m_tipo == 'DEPOSITO':
                caja += m_monto
                pos_spy += m_monto / p_spy
            elif m_tipo == 'RETIRO':
                caja -= m_monto
                pos_spy -= m_monto / p_spy
            elif m_tipo == 'COMPRA':
                caja -= m_monto
                posiciones[m['instrumento']] += abs(m.get('cantidad', 0))
            elif m_tipo == 'VENTA':
                caja += m_monto
                posiciones[m['instrumento']] -= abs(m.get('cantidad', 0))
            elif m_tipo == 'DIVIDENDO':
                caja += m_monto

        val_mkt = sum(posiciones[a] * precios.loc[dia, a] for a in activos if a in precios.columns)
        hist_cartera.append(val_mkt + caja)
        hist_spy.append(pos_spy * p_spy)

    return pd.Series(hist_cartera, index=fechas_rango), pd.Series(hist_spy, index=fechas_rango), precios

# --- INTERFAZ ---
tab1, tab2, tab3 = st.tabs(["🏠 Principal", "📥 Carga de Movimientos", "📉 Métricas & Riesgo"])

with tab2:
    st.subheader("Carga de Datos")
    archivo = st.file_uploader("Subir Excel o CSV", type=['xlsx', 'csv'])
    if archivo:
        df_c = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        df_c.columns = df_c.columns.str.lower().str.strip()
        df_c['fecha'] = pd.to_datetime(df_c['fecha'])
        st.session_state.df_movimientos = df_c
    elif 'df_movimientos' not in st.session_state:
        st.session_state.df_movimientos = pd.DataFrame([
            {'fecha': pd.to_datetime('2023-01-01'), 'tipo': 'DEPOSITO', 'instrumento': 'CASH', 'monto': 10000.0, 'cantidad': 0},
            {'fecha': pd.to_datetime('2023-01-05'), 'tipo': 'COMPRA', 'instrumento': 'AAPL', 'monto': 5000.0, 'cantidad': 35}
        ])
    st.session_state.df_movimientos = st.data_editor(st.session_state.df_movimientos, num_rows="dynamic", use_container_width=True)

# PROCESAMIENTO
mapa_tickers = {"AAPL":"AAPL.US", "GOVT":"GOVT.US", "SPY":"SPY.US", "MELI":"MELI.US", "MSFT":"MSFT.US"}
curva, spy_c, df_p = simular_cartera_final(st.session_state.df_movimientos, mapa_tickers)

if curva is not None:
    with tab1:
        v_act = curva.iloc[-1]
        df_m = st.session_state.df_movimientos
        neto = df_m[df_m['tipo'].str.upper()=='DEPOSITO']['monto'].sum() - df_m[df_m['tipo'].str.upper()=='RETIRO']['monto'].sum()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Total", f"$ {v_act:,.2f}", f"{(v_act/neto-1):.2%}" if neto > 0 else "0%")
        c2.metric("TIR Cartera", f"{xirr_core(df_m[df_m['tipo'].isin(['DEPOSITO','RETIRO'])]['monto'].tolist() + [-v_act], [pd.to_datetime(d) for d in df_m[df_m['tipo'].isin(['DEPOSITO','RETIRO'])]['fecha'].tolist()] + [pd.Timestamp.now().normalize()]):.2%}")
        c3.metric("Bench SPY", f"$ {spy_c.iloc[-1]:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curva.index, y=curva, name="Mi Cartera", line=dict(color="#00CC96", width=2)))
        fig.add_trace(go.Scatter(x=spy_c.index, y=spy_c, name="SPY Bench", line=dict(color="#EF553B", dash="dot")))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # 1. MÉTRICAS DE RIESGO
        vol, sha, sor, bet, var = obtener_metricas_riesgo(curva, spy_c)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Volatilidad", f"{vol:.2%}"); m2.metric("Sharpe", f"{sha:.2f}"); m3.metric("Sortino", f"{sor:.2f}"); m4.metric("Beta", f"{bet:.2f}"); m5.metric("VaR 95%", f"{var:.2%}")
        
        st.divider()
        
        # 2. TIR POR INSTRUMENTO
        st.subheader("Rendimiento por Instrumento")
        resumen_activos = []
        for act in [a for a in st.session_state.df_movimientos['instrumento'].unique() if a != 'CASH']:
            m_act = st.session_state.df_movimientos[st.session_state.df_movimientos['instrumento'] == act].copy()
            p_act = df_p[act].iloc[-1]
            cant = m_act[m_act['tipo'].str.upper()=='COMPRA']['cantidad'].sum() - m_act[m_act['tipo'].str.upper()=='VENTA']['cantidad'].sum()
            val_act = cant * p_act
            
            cfs = m_act.apply(lambda x: abs(x['monto']) if x['tipo'].upper() in ['VENTA','DIVIDENDO'] else -abs(x['monto']), axis=1).tolist() + [val_act]
            fechas = [pd.to_datetime(d) for d in m_act['fecha'].tolist()] + [pd.Timestamp.now().normalize()]
            
            resumen_activos.append({
                'Instrumento': act,
                'Inversión Neta': abs(sum([f for f in cfs[:-1] if f < 0])),
                'Valor Actual': val_act,
                'TIR Indiv.': xirr_core(cfs, fechas)
            })
        
        st.table(pd.DataFrame(resumen_activos).style.format({'Inversión Neta': '$ {:,.2f}', 'Valor Actual': '$ {:,.2f}', 'TIR Indiv.': '{:.2%}'}))
        
        st.divider()

        # 3. GRÁFICOS ADICIONALES
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**Drawdown Histórico**")
            st.area_chart((curva - curva.cummax()) / curva.cummax(), color="#ff4b4b")
        
        with col_right:
            st.write("**Matriz de Correlación**")
            if not df_p.empty:
                # Calculamos correlación de los retornos diarios de los activos
                corr_matrix = df_p.pct_change().corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig_corr, use_container_width=True)