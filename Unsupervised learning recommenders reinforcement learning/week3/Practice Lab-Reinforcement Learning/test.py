import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 模拟数据
batch_size = 5
state_size = 4
action_size = 2

states = np.random.rand(batch_size, state_size)
actions = np.random.randint(0, action_size, size=batch_size)
rewards = np.random.rand(batch_size)
next_states = np.random.rand(batch_size, state_size)
done_vals = np.random.randint(0, 2, size=batch_size)

experiences = (states, actions, rewards, next_states, done_vals)

# 模拟网络
q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(action_size)
])

target_q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(action_size)
])

# 计算最大Q值
max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

# 计算目标值
gamma = 0.99
y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

# 获取当前Q值
q_values = q_network(states)
q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))

# 计算损失
loss = tf.keras.losses.MeanSquaredError()(y_targets, q_values)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 可视化Q值
ax[0].bar(range(batch_size), q_values.numpy(), label='Q(s,a)')
ax[0].bar(range(batch_size), y_targets.numpy(), alpha=0.5, label='y_targets')
ax[0].set_xlabel('Sample Index')
ax[0].set_ylabel('Q Value')
ax[0].legend()
ax[0].set_title('Q Values vs Targets')

# 可视化损失
ax[1].bar(range(batch_size), (y_targets - q_values).numpy()**2, label='Squared Error')
ax[1].set_xlabel('Sample Index')
ax[1].set_ylabel('Squared Error')
ax[1].legend()
ax[1].set_title('Squared Error for Each Sample')

plt.show()