import pandas as pd
import tensorflow as tf
import numpy as np

from finetune.encoding import TextEncoder
from finetune.optimizers import AdamWeightDecay
from finetune.utils import find_trainable_variables
from finetune.config import MAX_LENGTH, BATCH_SIZE

SHAPES_PATH = os.path.join(os.path.dirnam(__file__), '..', 'model', 'params_shapes.json')
PARAM_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'params_{}.npy')


def model(X, M, Y, train=False, reuse=False):
    with tf.variable_scope('model', reuse=reuse):
        we = tf.get_variable("we", [encoder.vocab_size + n_ctx, n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
        we = dropout(we, embd_pdrop, train)

        X = tf.reshape(X, [-1, n_ctx])
        M = tf.reshape(M, [-1, n_ctx])

        h = embed(X, we)
        for layer in range(n_layer):
            h = block(h, 'h%d'%layer, train=train, scale=True)

        lm_h = tf.reshape(h[:, :-1], [-1, n_embd])
        lm_logits = tf.matmul(lm_h, we, transpose_b=True)
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
        lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])

        # weird indexing on M is happening because of the _start_ token
        lm_losses = tf.reduce_sum(lm_losses * M[:, 1:], 1) / tf.reduce_sum(M[:, 1:], 1)

        clf_h = tf.reshape(h, [-1, n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32) * n_ctx + pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, n_embd])
        if train and clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, 1 - clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, n_embd])
        clf_logits = clf(clf_h, 1, train=train)
        clf_logits = tf.reshape(clf_logits, [-1, 2])

        clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
        return clf_logits, clf_losses, lm_losses


class LanguageModelClassifier(object):

    def __init__(self, *args, **kwargs):
        self.encoder = TextEncoder()

        # tf placeholders
        self.X = tf.placeholder(tf.int32,   [None, max_length]) # token idxs
        self.M = tf.placeholder(tf.float32, [None, max_length]) # sequence mask
        self.Y = tf.placeholder(tf.int32,   [None])             # classification targets

        # symbolic ops
        self.logits    = None # classification logits
        self.clf_loss  = None # cross-entropy loss
        self.lm_losses = None # language modeling losses
        self.train     = None # gradient + parameter update

    def finetune(self, X, Y, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
        """
        X: List / array of text
        Y: Class labels
        """
        token_idxs = self.encoder.encode_for_classification(X, max_length=max_length)
        x, mask = self._array_format(token_idxs)
        self._build_model()
        self._load_saved_params()

    def _array_format(self, token_idxs):
        """
        Returns numpy array of token idxs and corresponding mask
        """
        seq_lengths = [len(x) for x in token_idxs]
        x    = np.zeros((n, max_length), dtype=np.int32)
        mask = np.zeros((n, max_length), dtype=np.float32)
        for i, seq_length in enumerate(seq_lengths):
            x[:, :seq_length] = token_idxs[i]
            # value of 1 means "consider this in cross-entropy LM loss"
            mask[:, 1:seq_length] = 1
        return x, mask

    def _build_model(self):
        """
        Finetune language model on text inputs
        """
        gpu_ops = []
        gpu_grads = []
        xs = (tf.split(x, n_gpu, 0) for x in xs)
        for i, xs in enumerate(zip(*xs)):
            do_reuse = True if i > 0 else None
            device = tf.device(assign_to_gpu(i, "/gpu:0"))
            scope = tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse)
            with device, scope:
                clf_logits, clf_losses, lm_losses = model(self.X, self.M, self.Y, train=True, reuse=do_reuse)
                if lm_coef > 0:
                    train_loss = tf.reduce_mean(clf_losses) + lm_coef * tf.reduce_mean(lm_losses)
                else:
                    train_loss = tf.reduce_mean(clf_losses)
                params = find_trainable_variables("model")
                grads = tf.gradients(train_loss, params)
                grads = list(zip(grads, params))
                gpu_grads.append(grads)
                gpu_ops.append([clf_logits, clf_losses, lm_losses])

        self.logits, self.clf_losses, self.lm_losses = [tf.concat(op, 0) for op in zip(*gpu_ops)]
        grads = average_grads(gpu_grads)
        grads = [g for g, p in grads]
        self.train = AdamWeightDecay(
            params=params,
            grads=grads,
            lr=lr,
            schedule=partial(lr_schedules[lr_schedule], warmup=lr_warmup),
            t_total=n_updates_total,
            l2=l2,
            max_grad_norm=max_grad_norm,
            vector_l2=vector_l2,
            b1=b1,
            b2=b2,
            e=e
        )
        self.clf_loss = tf.reduce_mean(self.clf_losses)

    def _load_saved_params(self):
        params = find_trainable_variables('model')
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        shapes = json.load(open(SHAPES_PATH))
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load(PARAM_PATH.format(n)) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:n_ctx]
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(len(encoder.special_tokens), n_embd)*0.02).astype(np.float32), init_params[0]], 0)
        del init_params[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=4)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)

    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    df = pd.read_csv("data/AirlineNegativity.csv")
    model = LanguageModelClassifier()
    model.finetune(df.Text.values, df.Target.values)
