import numpy as np
import tensorflow as tf
import gensim
import json

#model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)  

model =dict()

with open('../data/labs_2', 'r') as f:
    labels_org = f.read()

with open('../data/sentences_2', 'r') as f:
    sentences = f.read()

with open('../data/claims_2', 'r') as f:
    claims = f.read()

from string import punctuation
	
all_text = ''.join([c for c in sentences if c not in punctuation])
sentences = all_text.split('\n')

all_text = ''.join([c for c in claims if c not in punctuation])
claims = all_text.split('\n')

all_text = ' '.join(claims)
all_text += ' '.join(sentences)
words = all_text.split()


#changing here
words = list(set(words))
vocab_to_int = dict()

for i in range(len(words)):
 vocab_to_int.update({words[i]:i})
#from collections import Counter
#counts = Counter(words)
#vocab = sorted(counts, key=counts.get, reverse=True)
#vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

sent_ints = []
for each in sentences:
    sent_ints.append([vocab_to_int[word] for word in each.split()])

claim_ints = []
for each in claims:
    claim_ints.append([vocab_to_int[word] for word in each.split()])


labels = np.array([1 if l == "positive" else 0 for l in labels_org.split()])

from collections import Counter
claim_lens = Counter([len(x) for x in claim_ints])
sent_lens = Counter([len(x) for x in sent_ints])
print("Zero-length sentences: {}".format(sent_lens[0]))
print("Maximum sentence length: {}".format(max(sent_lens)))

print("Zero-length claims: {}".format(claim_lens[0]))
print("Maximum claim length: {}".format(max(claim_lens)))

# Filter out that review with 0 length
claim_ints = [r[0:200] for r in claim_ints if len(r) > 0]
sent_ints = [r[0:200] for r in sent_ints if len(r) > 0]

from collections import Counter
claim_lens = Counter([len(x) for x in claim_ints])
print("Zero-length claims: {}".format(claim_lens[0]))
print("Maximum claim length: {}".format(max(claim_lens)))

sent_lens = Counter([len(x) for x in sent_ints])
print("Zero-length claims: {}".format(sent_lens[0]))
print("Maximum claim length: {}".format(max(sent_lens)))

claim_seq_len = 10
sent_seq_len = 200
claim_features = np.zeros((len(claim_ints), claim_seq_len), dtype=int)
sent_features = np.zeros((len(sent_ints), sent_seq_len), dtype=int)
# print(features[:10,:100])
for i, row in enumerate(claim_ints):
    claim_features[i, -len(row):] = np.array(row)[:claim_seq_len]

for i, row in enumerate(sent_ints):
    sent_features[i, -len(row):] = np.array(row)[:sent_seq_len]

#features[:10,:100]

split_frac = 0.8

split_index = int(split_frac * len(claim_features))

train_claim, val_claim = claim_features[:split_index], claim_features[split_index:]
train_sent, val_sent = sent_features[:split_index], sent_features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

split_frac = 0.5
split_index = int(split_frac * len(val_claim))

val_claim, test_claim = val_claim[:split_index], val_claim[split_index:]
val_sent, test_sent = val_sent[:split_index], val_sent[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]


#print("\t\t\tFeature Shapes:")
#print("Train set: \t\t{}".format(train_x.shape),
#      "\nValidation set: \t{}".format(val_x.shape),
#      "\nTest set: \t\t{}".format(test_x.shape))
#print("label set: \t\t{}".format(train_y.shape),
#      "\nValidation label set: \t{}".format(val_y.shape),
#      "\nTest label set: \t\t{}".format(test_y.shape))

lstm_size = 256
lstm_layers = 2
batch_size = 1
learning_rate = 0.003

n_words = len(vocab_to_int) + 1 # Add 1 for 0 added to vocab

# Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    #inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    claims_ = tf.placeholder(tf.int32, [None, None], name="claims")
    sentences_ = tf.placeholder(tf.int32, [None, None], name="sentences")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

embed_size = 300

w2v_embed = np.ndarray([n_words,embed_size])


for i in range(n_words-1):
 if words[i] not in model:
   w2v_embed[vocab_to_int[words[i]]] = np.array([0]*embed_size)
 else:
   w2v_embed[vocab_to_int[words[i]]] = model[words[i]]



with tf.name_scope("Embeddings"):
    #embedding = tf.constant(w2v_embed, dtype = tf.float32)
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    #embed = tf.nn.embedding_lookup(embedding, inputs_)
    claim_embed = tf.nn.embedding_lookup(embedding, claims_)
    sent_embed = tf.nn.embedding_lookup(embedding, sentences_)




def lstm_cell():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    # Add dropout to the cell
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

#with tf.name_scope("RNN2"):
    # Stack up multiple LSTM layers, for deep learning
#    cell2 = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

    # Getting an initial state of all zeros
#    initial_state2 = cell2.zero_state(batch_size, tf.float32)



with tf.name_scope("RNN1"):
    # Stack up multiple LSTM layers, for deep learning
    cell1 = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

    # Getting an initial state of all zeros
    initial_state1 = cell1.zero_state(batch_size, tf.float32)

with tf.name_scope("RNN1_forward"):
    outputs1, final_state1 = tf.nn.dynamic_rnn(cell1, claim_embed, initial_state=initial_state1)

with tf.name_scope("RNN2_forward"):
    outputs2, final_state2 = tf.nn.dynamic_rnn(cell1, sent_embed, initial_state=initial_state1)

with tf.name_scope('predictions'):
    predictions = tf.contrib.layers.fully_connected(outputs1[:,-1], 1 ,activation_fn=tf.sigmoid)
    #predictions = tf.multiply(tf.contrib.layers.fully_connected(outputs1[:,-1], 300 , activation_fn=tf.sigmoid),tf.contrib.layers.fully_connected(outputs2[:,-1], 300 , activation_fn=tf.sigmoid))
    #predictions = tf.sigmoid(tf.reduce_sum(tf.multiply(tf.contrib.layers.fully_connected(outputs1[:,-1], 300 , activation_fn=tf.sigmoid),tf.contrib.layers.fully_connected(outputs2[:,-1], 300 , activation_fn=tf.sigmoid)), axis = 1))
    tf.summary.histogram('predictions', predictions)

with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels_, predictions)
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

merged = tf.summary.merge_all()

with tf.name_scope('validation'):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_batches(c_x,s_x, y, batch_size=100):
    n_batches = len(c_x) // batch_size
    c_x,s_x, y = c_x[:n_batches * batch_size], s_x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(c_x), batch_size):
        yield c_x[ii:ii + batch_size], s_x[ii:ii + batch_size], y[ii:ii + batch_size]



epochs = 10

# with graph.as_default():
saver = tf.train.Saver()

with tf.Session() as sess:
    #sess.run(embedding)
    #print(sess.run(embedding[vocab_to_int[words[29]]]))
    #print(words[29])
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./logs/tb/train', sess.graph)
    test_writer = tf.summary.FileWriter('./logs/tb/test', sess.graph)
    iteration = 1
    max_till_now = -1
    for e in range(epochs):
        local_max  = 0
        state1 = sess.run(initial_state1)
        #state2 = sess.run(initial_state2)

        for ii, (c_x,s_x, y) in enumerate(get_batches(train_claim, train_sent, train_y, batch_size), 1):
            feed = {#inputs_: x,
                    sentences_: s_x,
                    claims_: c_x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state1: state1}
                    #initial_state2: state2}
            summary, loss, state, _ = sess.run([merged, cost, final_state1, optimizer], feed_dict=feed)
            #             loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            train_writer.add_summary(summary, iteration)

            if iteration % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration % 25 == 0:
                val_acc = []
                val_state1 = sess.run(cell1.zero_state(batch_size, tf.float32))
                #val_state2 = sess.run(cell2.zero_state(batch_size, tf.float32))
                for c_x,s_x, y in get_batches(val_claim, val_sent, val_y, batch_size):
                    feed = {#inputs_: x,
			    sentences_: s_x,
                            claims_: c_x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state1: val_state1}
                            #initial_state2: val_state2}
                    #                     batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    summary, batch_acc, val_state = sess.run([merged, accuracy, final_state1], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
                local_max = np.mean(val_acc)
            iteration += 1
            test_writer.add_summary(summary, iteration)
            if local_max > max_till_now:
               max_till_now = local_max
               saver.save(sess, "./sentiment_manish.ckpt")
    print(max_till_now)

exit(0)

test_acc = []
with tf.Session() as sess:
    saver.restore(sess, "./sentiment_manish.ckpt")
    test_state = sess.run(cell1.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state1: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state1], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

