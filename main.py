import tensorflow as tf

batch_size = 1
time_steps = 71 # num frames
num_classes = 37
max_label_length = 10

# (batch, time, num_classes)
logits = tf.random.normal([batch_size, time_steps, num_classes])

# target label (say "1234" → mapped to int ids [1, 2, 3, 4])
labels = tf.constant([[1, 2, 3, 4]], dtype=tf.int32)

# label lengths (number of valid tokens in each target sequence)
label_length = tf.constant([4], dtype=tf.int32)

# logit lengths (sequence length output from model)
logit_length = tf.constant([time_steps], dtype=tf.int32)

loss = tf.nn.ctc_loss(
    labels=labels,
    logits=logits,
    label_length=label_length,
    logit_length=logit_length,
    logits_time_major=False,  # our logits are (batch, time, classes)
    blank_index=num_classes - 1
)

print("CTC Loss:", loss.numpy())
