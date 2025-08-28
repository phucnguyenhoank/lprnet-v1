#%%
import tensorflow as tf

batch_size = 2
time_steps = 71 # num frames
num_labels = 37
max_label_length = 10


logits = tf.random.normal([batch_size, time_steps, num_labels])
# list of logit length for each example in the batch[batch_size]
logit_length = tf.constant([time_steps] * batch_size, dtype=tf.int32)

y_true = tf.ragged.constant([
    [1, 2, 5, 6, 1],
    [2, 1, 3, 4, 3, 7, 2],
], dtype=tf.int32)

labels = y_true.to_sparse()

loss = tf.nn.ctc_loss(
    labels=labels,
    logits=logits,
    label_length=None, # auto inferred from the data (labels)
    logit_length=logit_length,
    logits_time_major=False,  # our logits are (batch, time, num_labels)
    blank_index=0
)

print("CTC Loss:", loss)
