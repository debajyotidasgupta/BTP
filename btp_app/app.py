from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy

from rl_agent.policy import load_policy
from rl_agent.environment import SmartApplEnv, Job

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="centered")
st.header("Smart Application Web App")
st.write('\n')


st.subheader('Input Applications and Jobs')


def input_applications(machine):
    with st.expander(machine, expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write(machine)
            file = "_".join(machine.lower().split())
            st.image(f'./assets/{file}.jpg', use_column_width=True)

        with col2:
            num_machines = st.slider(
                f'Number of {machine.lower()}', 1, 10, 7, key=f'mc_{file}')
            st.write(f'Number of washing {machine.lower()}:', num_machines)

            st.write('\n')

            num_jobs = st.slider(f'Number of jobs', 1,
                                 40, 10, key=f'jb_{file}')
            st.write(f'Number of jobs:', num_jobs)

        st.markdown("---")
    return num_machines, num_jobs


wm_machines, wm_jobs = input_applications('Washing Machine')
mo_machines, mo_jobs = input_applications('Microwave Oven')
ac_machines, ac_jobs = input_applications('Air Conditioner')

power = {
    240:  {"cost": 1, "limit": 110, "penalty": 4.5},
    480:  {"cost": 3, "limit": 110, "penalty": 2.9},
    720:  {"cost": 8, "limit": 200, "penalty": 5.0},
    960:  {"cost": 4, "limit": 170, "penalty": 1.8},
    1200: {"cost": 2, "limit": 140, "penalty": 3.8},
    1440: {"cost": 1, "limit": 230, "penalty": 5.5}
}


machine1 = [
    {"cycle": "Water Fill", "power": 0,      "time": 10},
    {"cycle": "Agitation",  "power": 3.97,   "time": 10},
    {"cycle": "Wash",       "power": 70.05,  "time": 15},
    {"cycle": "Drain",      "power": 0,      "time": 5},
    {"cycle": "Spin",       "power": 11.53,  "time": 5},
    {"cycle": "Water Fill", "power": 0,      "time": 10},
    {"cycle": "Rinse",      "power": 21.14,  "time": 5},
    {"cycle": "Drain",      "power": 0,      "time": 5},
    {"cycle": "Spin",       "power": 28.09,  "time": 10}
]
machine2 = [{"cycle": "microwave", "power": 50, "time": 10, }]
machine3 = [{"cycle": "AC", "power": 100, "time": 25, }]

st.subheader("Rate Charts")
with st.expander("Price Chart", expanded=True):
    st.json(power)
with st.expander("Power Chart [Washing Machine]"):
    st.json(machine1)
with st.expander("Power Chart [Microwave Oven]"):
    st.json(machine2)
with st.expander("Power Chart [Air Conditioner]"):
    st.json(machine3)

st.subheader("Schedule")


def scheduler(gym_env, random_policy=False):
    py_env = gym_wrapper.GymWrapper(gym_env)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    if random_policy:
        policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                 tf_env.action_spec())
    else:
        policy = load_policy('policy_18000')

    time_step = tf_env.reset()
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = tf_env.step(action_step.action)

    return py_env.bill, py_env.schedule_list, py_env.power_usage, py_env.cost_chart


UNIT = 5
TIME = 2000
N_MACHINES = [7, 7, 7]
N_JOBS = [10, 10, 10]

env = SmartApplEnv(number_of_machines=N_MACHINES,
                   number_of_jobs=N_JOBS,
                   distinct_machines=3,
                   power_rate_chart=power,
                   full_day=TIME // UNIT,
                   time_per_unit=UNIT,
                   cycle_stat=[machine1, machine2, machine3]
                   )

bill, schedule, power_usage, cost_chart = scheduler(env)
r_bill, r_schedule, r_power_usage, r_cost_chart = scheduler(env, True)

df = pd.DataFrame(schedule)
df['type'] = df['type'].apply(
    lambda x: ["Washing Machine", "Microwave Oven", "Air Conditioner"][x])
df['time'] = df['time'].apply(
    lambda x: "{:02d}:{:02d}".format((x * UNIT)//60, (x * UNIT) % 60))
df = df.sort_values(['time', 'type', 'mc_id']).reset_index(drop=True)
st.dataframe(df)

st.markdown("---")
st.markdown("### Power Usage")
st.area_chart(pd.DataFrame(
    np.array([power_usage[:40], r_power_usage[:40]]).T, columns=["policy", "random"]))
st.markdown("### Cost per Unit Time")
st.line_chart(pd.DataFrame(
    np.array([cost_chart[:40], r_cost_chart[:40]]).T, columns=["policy", "random"]))

st.subheader("Bill Amount")
with st.expander('Policy', expanded=True):
    st.write("Total Cost : ", bill)
with st.expander('Random', expanded=True):
    st.write("Total Cost : ", r_bill)
