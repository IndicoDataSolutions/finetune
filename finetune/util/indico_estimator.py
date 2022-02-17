import numpy as np
import tensorflow as tf
import warnings

try:
    from tensorflow.neuron import trace as neuron_trace

    AWS_NEURON_AVAILABLE = True
except ImportError:
    AWS_NEURON_AVAILABLE = False


def placeholder_like(tensor):
    return tf.compat.v1.placeholder(tensor.dtype, shape=tensor.shape)


def parse_input_fn_result(result):
    iterator = tf.compat.v1.data.make_initializable_iterator(result)
    init = iterator.initializer
    result = iterator.get_next()
    return result, init


class IndicoEstimator(tf.estimator.Estimator):
    def __init__(self, *args, **kwargs):
        self.estimator_spec = None
        self.g = None
        self.features_real = None
        self.placeholder_feats = None
        self.predictions = None
        self.mon_sess = None
        self._cached_predict = False
        super().__init__(*args, **kwargs)

    def get_features_from_fn(self, input_fn, predict=True):
        with tf.Graph().as_default() as g:
            result = self._call_input_fn(input_fn, tf.estimator.ModeKeys.PREDICT)
            features, initializer = parse_input_fn_result(result)
            if type(features) == tuple and predict:
                features = features[0]
            with tf.compat.v1.Session(config=self._session_config) as sess:
                sess.run(initializer)
                output = []
                while True:
                    try:
                        output.append(sess.run(features))
                    except tf.errors.OutOfRangeError:
                        break
            return output, features

    def close_predict(self):
        self.estimator_spec = None
        self.features_real = None
        self.placeholder_feats = None
        self.predictions = None
        self.g = None
        if self.mon_sess is not None:
            self.mon_sess.close()
        self.mon_sess = None

    def inferentia_predict(
        self,
        input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True,
    ):
        if not AWS_NEURON_AVAILABLE:
            raise ValueError("AWS Neuron not available")
        # Check that model has been trained.
        self.g = self.g or tf.Graph()
        tf.compat.v1.set_random_seed(self._config.tf_random_seed)
        features_real, features = self.get_features_from_fn(input_fn) # Still use a tf-session for this.
        with self.g.as_default(): # TODO: probably remove this - I hope that tracing happens in it's own graph that is then properly cleaned up.
            if self.aws_neuron_model is None:
                example_feats = tf.nest.map_structure(tf.ones_like, features)

                def place_on_inferentia(nodedef):
                    # TODO: probably need to generically filter out any CRFs or similar funkiness.
                    return True

                # TODO: This does not include the pre-trained weights. Need to figure out how the tracing works and how to inject  the weights from the hooks.
                def get_prediction_graph(features):
                    all_hooks = hooks or []
                    all_hooks.extend(list(self.estimator_spec.prediction_hooks or []))
                    return self._extract_keys(
                        self._call_model_fn(
                            features, None, tf.estimator.ModeKeys.PREDICT, self.config
                        ).predictions,
                        predict_keys,
                    )

                self.aws_neuron_model = neuron_trace(
                    get_prediction_graph,
                    example_feats,
                    subgraph_builder_function=place_on_inferentia,
                )
                # TODO: save out using code below and attach to the saver and re-save with the build model. We can then try to only build if pre-build doesn't exist?
                #                model_dir = './model_neuron'
                #                model_neuron.save(model_dir)
                #                model_neuron_reloaded = tf.keras.models.load_model(model_dir)
            for feats in features_real:
                preds_evaluated = self.aws_neuron_model(feats)
                if not yield_single_examples:
                    yield preds_evaluated
                elif not isinstance(self.predictions, dict):
                    for pred in preds_evaluated:
                        yield pred
                else:
                    for i in range(self._extract_batch_length(preds_evaluated)):
                        yield {key: value[i] for key, value in preds_evaluated.items()}

    def cached_predict(
        self,
        input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_path=None,
        yield_single_examples=True,
    ):
        # Check that model has been trained.
        self.g = self.g or tf.Graph()
        tf.compat.v1.set_random_seed(self._config.tf_random_seed)
        features_real, features = self.get_features_from_fn(input_fn)
        with self.g.as_default():
            if self.estimator_spec is None:
                self._create_and_assert_global_step(self.g)
                if not checkpoint_path:
                    checkpoint_path = tf.train.latest_checkpoint(self._model_dir)
                if not checkpoint_path:
                    tf.compat.v1.logging.info(
                        "Could not find trained model in model_dir: {}, running "
                        "initialization to predict.".format(self._model_dir)
                    )

                self.placeholder_feats = tf.nest.map_structure(
                    placeholder_like, features
                )
                self.estimator_spec = self._call_model_fn(
                    self.placeholder_feats,
                    None,
                    tf.estimator.ModeKeys.PREDICT,
                    self.config,
                )
                # Call to warm_start has to be after model_fn is called.
                self._maybe_warm_start(checkpoint_path)

                self.predictions = self._extract_keys(
                    self.estimator_spec.predictions, predict_keys
                )
                all_hooks = hooks or []
                all_hooks.extend(list(self.estimator_spec.prediction_hooks or []))

                self.mon_sess = tf.compat.v1.train.MonitoredSession(
                    session_creator=tf.compat.v1.train.ChiefSessionCreator(
                        checkpoint_filename_with_path=checkpoint_path,
                        master=self._config.master,
                        scaffold=self.estimator_spec.scaffold,
                        config=self._session_config,
                    ),
                    hooks=all_hooks,
                )

            for feats in features_real:
                feed_dict = {self.placeholder_feats[k]: v for k, v in feats.items()}
                preds_evaluated = self.mon_sess.run(
                    self.predictions, feed_dict=feed_dict
                )
                if not yield_single_examples:
                    yield preds_evaluated
                elif not isinstance(self.predictions, dict):
                    for pred in preds_evaluated:
                        yield pred
                else:
                    for i in range(self._extract_batch_length(preds_evaluated)):
                        yield {key: value[i] for key, value in preds_evaluated.items()}
