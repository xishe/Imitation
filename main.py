import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import random
import requests
import time
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Имитация пассажирских маршрутов Оренбурга")

@st.cache_data(show_spinner=False)
def get_osm_routes_overpass():
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = '''
    [out:json][timeout:60];
    area["name"="Оренбург"]->.a;
    (
      relation["route"="bus"](area.a);
      relation["route"="share_taxi"](area.a);
    );
    out body;
    >;
    out skel qt;
    '''
    response = requests.get(overpass_url, params={'data': query})
    if response.status_code != 200:
        return None
    data = response.json()
    nodes = {el['id']: el for el in data['elements'] if el['type']=='node'}
    routes = []
    for rel in data['elements']:
        if rel['type'] != 'relation':
            continue
        if 'members' not in rel or 'tags' not in rel:
            continue
        stops = []
        for m in rel['members']:
            if m['role'] in ('stop','platform','stop_entry_only','stop_exit_only') and m['type']=='node' and m['ref'] in nodes:
                node = nodes[m['ref']]
                stop_name = node['tags']['name'] if ('tags' in node and 'name' in node['tags']) else f"ID {node['id']}"
                stops.append({'name': stop_name, 'lat': node['lat'], 'lon': node['lon']})
        if len(stops)<2:
            continue
        routes.append({
            'number': rel['tags'].get('ref', rel['tags'].get('name','?')),
            'type': rel['tags'].get('route','bus'),
            'from': rel['tags'].get('from', stops[0]['name']),
            'to': rel['tags'].get('to', stops[-1]['name']),
            'stops': stops
        })
    return sorted(routes, key=lambda r: r['number'])

with st.spinner("Загрузка маршрутов с OpenStreetMap..."):
    osm_routes = get_osm_routes_overpass()
if osm_routes is None or len(osm_routes) == 0:
    st.error("Ошибка загрузки маршрутов. Попробуйте позже.")
    st.stop()

# ==== СЛАЙДБАР ПАРАМЕТРЫ ====
with st.sidebar:
    st.header("Параметры симуляции")
    rnames = [f"{r['number']}: {r['from']} — {r['to']}" for r in osm_routes]
    idx = st.selectbox("Маршрут:", range(len(osm_routes)), format_func=lambda x: rnames[x])
    route = osm_routes[idx]
    n_stops = len(route['stops'])
    n_buses = st.slider("Автобусов", 1, 40, min(12, n_stops))
    seats = st.slider("Мест в автобусе", 6, 40, 20)
    start_time = st.slider("Время начала движения", 4, 12, 6, format="%02d:00")
    end_time = st.slider("Время окончания движения", 12, 24, 21, format="%02d:00")
    highlight_thr = st.slider("Порог проблемных остановок", 3, 40, 15)
    anim_speed_x = st.slider("Скорость анимации (x, шагов/сек)", 1, 250, 25)
    COST_PER_PASSENGER = st.slider("Стоимость проезда (руб.)", 10, 100, 35)
    MEAN_PAX_INTERVAL_DAY = st.slider("Интервал прихода пассажира днем (мин)", 0.2, 10.0, 2.0, step=0.1)
    MEAN_PAX_INTERVAL_PEAK = st.slider("Интервал прихода пассажира в час-пик (мин)", 0.05, 5.0, 0.7, step=0.05)
    PEAK_HOURS = st.multiselect("Часы-пик (час, 24ч)", [i for i in range(6, 23)], [7, 8, 9, 17, 18, 19])
    st.markdown("---")
    auto_anim = st.checkbox("Авто-анимация (запуск/пауза)", value=False)
    step_btn = st.button("Следующий шаг")
    run_to_end_btn = st.button("Имитация до конца (моментально)")
    update_analytics = st.button("Обновить графики/аналитику")
    run_sim = st.button("▶ Новый запуск", key="run")
    stop_sim = st.button("■ Стоп", key="stop")

def safe_idx(lst, i, default=0):
    try:
        return lst[i]
    except IndexError:
        return default

def safe_list_idx(lst, i, fallback=None):
    if i < len(lst):
        return lst[i]
    elif len(lst) > 0:
        return lst[-1]
    else:
        return fallback

def init_sim_state(route, n_buses, seats, start_time, end_time):
    n_stops = len(route['stops'])
    start = datetime(2023,1,1,start_time,0)
    times = [start + timedelta(minutes=i) for i in range((end_time-start_time)*60)]
    # Корректное распределение по остановкам, не выходя за пределы массива
    bus_init_pos = [int(round(i * (n_stops-1) / max(n_buses-1,1))) for i in range(n_buses)]
    return {
        'bus_positions': bus_init_pos,
        'bus_free': [seats]*n_buses,
        'stop_queues': [0]*n_stops,
        'bus_trail': [[bus_init_pos[i]] for i in range(n_buses)],
        'bus_pax': [0]*n_buses,
        'bus_stop_names': [route['stops'][bus_init_pos[i]]['name'] for i in range(n_buses)],
        'bus_stop_pax': [0]*n_buses,
        'served_total': 0,
        'highlight': set(),
        'step': 0,
        'wait_hist': [[] for _ in range(n_stops)],
        'queue_hist': [[] for _ in range(n_stops)],
        'profit_hist': [],
        'max_queue_hist': [],
        'avg_queue_hist': [],
        'avg_busload_hist': [],
        'max_busload_hist': [],
        'free_seats_hist': [],
        'status':'Ожидание старта',
        'log':[],
        'times': times,
        'run': False,
        'stopped': False,
    }

def is_peak_step(cur_step, times, peak_hours):
    if cur_step >= len(times): return False
    hour = times[cur_step].hour
    return hour in peak_hours

def simulate_step(state, route, n_buses, seats, pax_prob, highlight_thr, peak, COST_PER_PASSENGER):
    n_stops = len(route['stops'])
    total_pass = 0
    for i in range(n_stops):
        prob = pax_prob * (2.5 if peak else 1)
        if random.random() < prob:
            state['stop_queues'][i] += 1
            state['wait_hist'][i].append(0)
        state['wait_hist'][i] = [w+1 for w in state['wait_hist'][i]]

    total_profit = 0
    busloads = []
    for b in range(n_buses):
        pos = safe_idx(state['bus_positions'], b, 0)
        exited = random.randint(0, seats - state['bus_free'][b])
        state['bus_free'][b] += exited
        can_board = min(safe_idx(state['stop_queues'], pos, 0), state['bus_free'][b])
        state['stop_queues'][pos] = max(0, safe_idx(state['stop_queues'], pos, 0) - can_board)
        state['bus_free'][b] -= can_board
        total_pass += can_board
        total_profit += can_board * COST_PER_PASSENGER
        state['bus_pax'][b] = seats - state['bus_free'][b]
        busloads.append(state['bus_pax'][b])
        state['wait_hist'][pos] = state['wait_hist'][pos][can_board:]
        next_stop = (pos + 1) % n_stops
        state['bus_positions'][b] = next_stop
        state['bus_trail'][b].append(next_stop)
        state['bus_stop_names'][b] = route['stops'][next_stop]['name']
        state['bus_stop_pax'][b] = state['stop_queues'][next_stop]
    state['step'] += 1
    state['served_total'] += total_pass
    queues = [len(q) for q in state['wait_hist']]
    state['queue_hist'] = [q+[len_q] for q, len_q in zip(state['queue_hist'], queues)]
    state['profit_hist'].append(total_profit)
    state['max_queue_hist'].append(max(queues) if queues else 0)
    state['avg_queue_hist'].append(np.mean(queues) if queues else 0)
    state['avg_busload_hist'].append(np.mean(busloads) if busloads else 0)
    state['max_busload_hist'].append(max(busloads) if busloads else 0)
    state['free_seats_hist'].append(np.mean([seats-x for x in busloads]))
    state['highlight'] = set([i for i, q in enumerate(state['stop_queues']) if q >= highlight_thr])
    time_str = state['times'][state['step']-1].strftime('%H:%M') if state['step']-1 < len(state['times']) else "--:--"
    state['status'] = f"Шаг: {state['step']}, Время: {time_str}, Обслужено: {state['served_total']}, Проблемных ост.: {len(state['highlight'])}"
    state['log'].append({
        'step': state['step'],
        'time': time_str,
        'served_total': state['served_total'],
        'max_queue': max(queues),
        'avg_queue': np.mean(queues),
        'profit': total_profit,
        'busload_avg': np.mean(busloads),
        'busload_max': max(busloads),
        'free_seats_avg': np.mean([seats-x for x in busloads]),
        'problems': len(state['highlight']),
        'total_pax_on_stops': sum(state['stop_queues'])
    })
    return state

# ======= SESSION CONTROL ===========
if 'state' not in st.session_state or run_sim:
    st.session_state.state = init_sim_state(route, n_buses, seats, start_time, end_time)
    st.session_state.running = False
    st.session_state.paused = False

if stop_sim:
    st.session_state.running = False
    st.session_state.paused = True

total_steps = len(st.session_state.state['times'])

def one_step():
    state = st.session_state.state
    if state['step'] < total_steps:
        peak = is_peak_step(state['step'], state['times'], PEAK_HOURS)
        pax_prob = 1.0/(MEAN_PAX_INTERVAL_PEAK if peak else MEAN_PAX_INTERVAL_DAY)
        simulate_step(state, route, n_buses, seats, pax_prob, highlight_thr, peak, COST_PER_PASSENGER)

def do_steps(N=1):
    for _ in range(N):
        if st.session_state.state['step'] >= total_steps:
            break
        one_step()

if auto_anim and st.session_state.state['step'] < total_steps:
    do_steps(anim_speed_x)
    time.sleep(1.0)

if step_btn and st.session_state.state['step'] < total_steps:
    one_step()

if run_to_end_btn and st.session_state.state['step'] < total_steps:
    do_steps(total_steps - st.session_state.state['step'])

st.markdown("*Автобусы стартуют с равномерно распределённых по маршруту остановок.*")
st.markdown(f"**Маршрут {route['number']}**: {route['from']} — {route['to']}")
state = st.session_state.state
col1, col2 = st.columns([1,2])
with col1:
    st.markdown('<h4 style="color:green;">' + ("🟢 Имитация идет" if auto_anim else "⏸ Пауза") + '</h4>', unsafe_allow_html=True)
    st.markdown(f"**Текущее время:** {safe_list_idx(state['times'], state['step']-1, fallback='--:--') if state['step'] > 0 else safe_list_idx(state['times'], 0, fallback='--:--')}")
    st.metric("Обслужено пассажиров", state['served_total'])
    st.metric("Пассажиров на остановках", sum(state['stop_queues']))
    st.metric("Проблемных остановок", len(state['highlight']))
    st.metric("Общий доход (руб.)", int(np.sum(state['profit_hist'])))
    st.metric("Средняя загрузка автобусов", f"{np.mean(state['avg_busload_hist']):.1f}" if state['step']>0 else 0)
    st.metric("Средняя очередь на остановке", f"{np.mean(state['avg_queue_hist']):.1f}" if state['step']>0 else 0)
    st.metric("Среднее время ожидания (шагов)", f"{np.mean([w for wh in state['wait_hist'] for w in wh]):.1f}" if state['step']>0 and sum(map(len, state['wait_hist']))>0 else 0)

with col2:
    st.subheader("Карта движения транспорта")
    m = folium.Map(location=[route['stops'][0]['lat'], route['stops'][0]['lon']], zoom_start=12)
    for idx, stop in enumerate(route['stops']):
        queue_val = state['stop_queues'][idx] if idx < len(state['stop_queues']) else 0
        color = 'red' if idx in state['highlight'] else 'blue'
        folium.Marker(
            [stop['lat'], stop['lon']],
            tooltip=f"{idx+1}. {stop['name']} (очередь: {queue_val})",
            icon=folium.Icon(color=color, icon='info-sign' if color=='blue' else 'alert', prefix='fa')
        ).add_to(m)
    folium.PolyLine([[s['lat'], s['lon']] for s in route['stops']], color="blue", weight=5, opacity=0.9).add_to(m)
    bus_colors = ['darkred','orange','green','purple','cadetblue','black','pink','lightgray']
    for b in range(n_buses):
        pos = safe_idx(state['bus_positions'], b, 0)
        bus_pax_val = safe_idx(state['bus_pax'], b, 0)
        folium.CircleMarker(
            [route['stops'][pos]['lat'], route['stops'][pos]['lon']],
            radius=12, color=bus_colors[b%len(bus_colors)], fill=True, fill_opacity=0.85,
            tooltip=(f"Автобус {b+1}: {route['stops'][pos]['name']}\n"
                     f"В салоне: {bus_pax_val} / {seats}\n"
                     f"Свободно: {seats-bus_pax_val}\n"
                     f"Очередь на ост.: {state['stop_queues'][pos] if pos < len(state['stop_queues']) else 0}")
        ).add_to(m)
    st_folium(m, width=700, height=540)

st.markdown("### Данные автобусов")
bus_pos = [safe_idx(state['bus_positions'], i, 0) for i in range(n_buses)]
bus_pax = [safe_idx(state['bus_pax'], i, 0) for i in range(n_buses)]
df_buses = pd.DataFrame({
    'Автобус': list(range(1, n_buses+1)),
    'Остановка': [safe_list_idx(route['stops'], p, {'name':'—'})['name'] for p in bus_pos],
    'В салоне': bus_pax,
    'Свободно мест': [seats-x for x in bus_pax],
    'Очередь на ост.': [safe_idx(state['stop_queues'], p, 0) for p in bus_pos]
})
st.dataframe(df_buses, use_container_width=True)

st.markdown("### Очереди на всех остановках")
df_stops = pd.DataFrame({
    'Остановка': [s['name'] for s in route['stops']],
    'Очередь': [safe_idx(state['stop_queues'], i, 0) for i in range(n_stops)]
})
st.dataframe(df_stops, use_container_width=True)

if update_analytics and state['step'] > 0:
    hist = state
    steps = range(1, len(hist['profit_hist'])+1)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(4,2, figsize=(18,22))
    axs[0,0].plot(np.cumsum(hist['profit_hist']), label='Доход накопительный', color='green', linewidth=2)
    axs[0,0].set_title('Доход (накопительно)')
    axs[0,0].legend()
    axs[0,1].plot(hist['served_total']*np.ones(len(steps)), label='Обслужено всего', color='blue')
    axs[0,1].set_title('Всего обслужено пассажиров')
    axs[0,1].legend()
    axs[1,0].bar(steps, hist['max_queue_hist'], color='red', alpha=0.6)
    axs[1,0].set_title('Максимальная очередь на остановках')
    axs[1,1].plot(hist['avg_queue_hist'], color='orange', linestyle='dashed', marker='o')
    axs[1,1].set_title('Средняя очередь по остановкам')
    axs[2,0].plot(hist['avg_busload_hist'], color='purple', linewidth=2)
    axs[2,0].set_title('Средняя загрузка автобусов')
    axs[2,1].bar(steps, hist['max_busload_hist'], color='navy', alpha=0.5)
    axs[2,1].set_title('Максимальная загрузка автобусов')
    axs[3,0].plot(hist['free_seats_hist'], color='brown', linestyle='--', marker='s')
    axs[3,0].set_title('Среднее число свободных мест')
    axs[3,1].plot([log['total_pax_on_stops'] for log in hist['log']], color='indigo', linewidth=2)
    axs[3,1].set_title('Пассажиров на всех остановках')
    for ax in axs.flat:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    df_log = pd.DataFrame(state['log'])
    summary = {
        "Всего обслужено пассажиров": [state['served_total']],
        "Общий доход (руб.)": [int(np.sum(state['profit_hist']))],
        "Макс. очередь (за всё время)": [np.max(state['max_queue_hist'])],
        "Средн. очередь": [np.mean(state['avg_queue_hist'])],
        "Макс. загрузка автобусов": [np.max(state['max_busload_hist'])],
        "Средн. загрузка автобусов": [np.mean(state['avg_busload_hist'])],
        "Макс. пассажиров на остановках": [np.max([log['total_pax_on_stops'] for log in state['log']])],
        "Средн. пассажиров на остановках": [np.mean([log['total_pax_on_stops'] for log in state['log']])],
        "Кол-во проблемных остановок (макс.)": [np.max([log['problems'] for log in state['log']])],
        "Кол-во проблемных остановок (средн.)": [np.mean([log['problems'] for log in state['log']])]
    }
    st.markdown("#### Итоговая таблица-отчет")
    st.dataframe(pd.DataFrame(summary).T, use_container_width=True)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_log.to_excel(writer, index=False, sheet_name='Лог')
        pd.DataFrame(summary).T.to_excel(writer, sheet_name='Сводка')
    st.download_button("Скачать лог симуляции (Excel)", data=output.getvalue(), file_name="simulation_log.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown(f"**Обслужено пассажиров:** {state['served_total']} | Проблемных ост.: **{len(state['highlight'])}**")
else:
    st.info("Нажмите кнопку 'Обновить графики/аналитику' для построения графиков")
